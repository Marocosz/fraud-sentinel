import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

# ==============================================================================
# ARQUIVO: random_forest_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo de Random Forest.
#
# MELHORIAS (v2):
#   - RandomizedSearchCV ao inves de GridSearchCV (espaco maior sem explodir
#     computacao, referencia: Bergstra & Bengio, 2012).
#   - Espaco de busca enriquecido com max_features, min_samples_leaf,
#     bootstrap e class_weight variavel.
#   - Threshold calculado em hold-out de validacao (sem data leakage).
#   - Logging centralizado via threshold_utils.
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao (Model Training Stage).
#
# COMUNICACAO:
#   - Le: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/rf_best_model.pkl, models/rf_threshold.txt
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline
from src.models.threshold_utils import compute_optimal_threshold, save_threshold, log_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURACAO DO MODELO (PARAMETROS)
# ==============================================================================
MODEL_CONFIG = {
    "model_class": RandomForestClassifier,
    "model_params": {
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    },
    
    "smote_strategy": None,
    "cv_folds": 3,
    
    # Espaco de Busca ENRIQUECIDO (RandomizedSearchCV)
    "param_distributions": {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [5, 10, 15, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 5, 10],
        'model__max_features': ['sqrt', 'log2', 0.3, 0.5],  # Fracao de features por split
        'model__class_weight': ['balanced', 'balanced_subsample'],
    },
    
    "n_iter": 40,   # Combinacoes aleatorias
    "n_jobs": 1,    # Jobs do search (RF ja usa paralelismo interno)
    "verbose": 1
}

def train_random_forest():
    """
    Treina o modelo de Random Forest com otimizacao completa.
    
    METODOLOGIA:
    ----------------------
    1. Pipeline Completo: EDAFeatureEngineer -> ColumnTransformer -> RandomForest.
    2. RandomizedSearchCV com 40 iteracoes para espaco expandido.
    3. Threshold otimizado em hold-out de validacao (sem data leakage).
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento Random Forest (Run ID: {run_id})...")
    logger.info(f"ℹ️  Configuracao: SMOTE={MODEL_CONFIG['smote_strategy']}")
    
    # -------------------------------------------------------------------------
    # 1. CARGA DE DADOS
    # -------------------------------------------------------------------------
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino nao encontrados.")
        return

    logger.info("📂 Carregando dados de treino...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    logger.info(f"   Dimensoes: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")

    # -------------------------------------------------------------------------
    # 2. DEFINICAO DO PIPELINE
    # -------------------------------------------------------------------------
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("❌ SMOTE Desativado. Usando class_weight='balanced'.")
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    # -------------------------------------------------------------------------
    # 3. RANDOMIZED SEARCH CV
    # -------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=MODEL_CONFIG["param_distributions"],
        n_iter=MODEL_CONFIG["n_iter"],
        scoring='roc_auc', 
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"],
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    # -------------------------------------------------------------------------
    # 4. AMOSTRAGEM PARA BUSCA (Random Forest em 800k × 120 fits = muito lento)
    # -------------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    
    SAMPLE_SIZE = 200000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimizacao Acelerada: Usando amostra de {SAMPLE_SIZE} linhas para RandomizedSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train
    
    # -------------------------------------------------------------------------
    # 5. TREINAMENTO E OTIMIZACAO
    # -------------------------------------------------------------------------
    logger.info(f"⚙️  Otimizando Hiperparametros (RandomizedSearchCV: {MODEL_CONFIG['n_iter']} iteracoes)...")
    
    print(f"\n⚡ Iniciando treinamento...")
    random_search.fit(X_sample, y_sample)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info("✅ Treinamento Concluido!")
    logger.info(f"🏆 Melhor ROC-AUC Medio: {best_score:.4f}")
    logger.info(f"🔧 Melhores Parametros: {best_params}")
    
    # Diagnostico de Overfitting
    best_idx = random_search.best_index_
    train_score = random_search.cv_results_['mean_train_score'][best_idx]
    logger.info(f"   ROC-AUC (Treino): {train_score:.4f} | Gap: {train_score - best_score:.4f}")
    
    # -------------------------------------------------------------------------
    # 6. RETREINO COM DATASET COMPLETO
    # -------------------------------------------------------------------------
    logger.info("🚀 Retreinando modelo com TODOS os dados (800k+)...")
    best_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 7. PERSISTENCIA
    # -------------------------------------------------------------------------
    latest_model_path = MODELS_DIR / "rf_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_rf_{run_id}.pkl"
    joblib.dump(best_model, latest_model_path)
    joblib.dump(best_model, versioned_model_path)
    
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")
    
    # -------------------------------------------------------------------------
    # 6. THRESHOLD TUNING (Corrigido: Hold-out de Validacao)
    # -------------------------------------------------------------------------
    best_threshold, best_fbeta, best_model = compute_optimal_threshold(
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="rf",
        skip_final_refit=True  # RandomizedSearch ja treinou com todos os dados
    )
    
    save_threshold(best_threshold, "rf", MODELS_DIR)
    
    # -------------------------------------------------------------------------
    # 7. REGISTRO DO EXPERIMENTO
    # -------------------------------------------------------------------------
    log_experiment(
        run_id=run_id,
        model_type="RandomForestClassifier",
        best_params=best_params,
        best_cv_score=best_score,
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None,
        extra_data={
            "search_type": "RandomizedSearchCV",
            "n_iter": MODEL_CONFIG["n_iter"],
            "train_auc_gap": float(train_score - best_score)
        }
    )
    
    with open(MODELS_DIR / "rf_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_random_forest()
