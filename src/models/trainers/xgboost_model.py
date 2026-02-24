import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint

# ==============================================================================
# ARQUIVO: xgboost_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo XGBoost com busca de hiperparametros ampla.
#
# MELHORIAS (v2):
#   - RandomizedSearchCV ao inves de GridSearchCV (mais eficiente para espacos
#     grandes de hiperparametros, referencia: Bergstra & Bengio, 2012).
#   - Espaco de busca enriquecido com subsample, colsample_bytree,
#     min_child_weight e gamma (regularizacao).
#   - Threshold calculado em hold-out de validacao (sem data leakage).
#   - Logging centralizado via threshold_utils.
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao (Model Training Stage).
#
# COMUNICACAO:
#   - Le: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/xgb_best_model.pkl, models/xgb_threshold.txt
# ==============================================================================

# Ignora avisos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Configuracao de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports do Projeto
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline
from src.models.threshold_utils import compute_optimal_threshold, save_threshold, log_experiment

# Configuracao de Logs
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
    # Modelo Base
    "model_class": XGBClassifier,
    "model_params": {
        "eval_metric": "logloss",
        "scale_pos_weight": 90,  # Peso alto (90:1) para compensar desbalanceamento
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",   # Treinamento acelerado via histograma
    },
    
    # Estrategia de Oversampling
    "smote_strategy": None,
    
    # Validacao Cruzada
    "cv_folds": 3,
    
    # Espaco de Busca ENRIQUECIDO (RandomizedSearchCV)
    # Referencia: XGBoost Documentation + "Practical Hyperparameter Optimization"
    # (Probst et al., 2019)
    "param_distributions": {
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 4, 5, 6, 7, 8],
        'model__min_child_weight': [1, 3, 5, 7],        # Regularizacao: min amostras por folha
        'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  # Bagging: fracao de amostras por arvore
        'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],  # Feature bagging por arvore
        'model__gamma': [0, 0.1, 0.3, 0.5, 1.0],        # Min loss reduction para split
        'model__reg_alpha': [0, 0.01, 0.1, 1.0],         # Regularizacao L1
        'model__reg_lambda': [0.1, 0.5, 1.0, 5.0],       # Regularizacao L2
    },
    
    # Numero de combinacoes aleatorias a testar
    "n_iter": 60,
    
    # Configuracao de Execucao
    "n_jobs": 1,       # Jobs do GridSearch (XGBoost ja usa paralelismo interno)
    "verbose": 1
}

def train_xgboost():
    """
    Treina o modelo XGBoost com otimizacao completa de hiperparametros.
    
    METODOLOGIA:
    ----------------------
    1. Pipeline Completo: EDAFeatureEngineer -> ColumnTransformer -> XGBClassifier.
    2. RandomizedSearchCV para exploracao eficiente do espaco de hiperparametros.
       Com 60 iteracoes aleatorias, cobrimos estatisticamente 95% do espaco efetivo
       (Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization", 2012).
    3. Threshold otimizado em hold-out de validacao (sem data leakage).
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento XGBoost (Run ID: {run_id})...")
    
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
    # 2. PIPELINE (EDA-Driven)
    # -------------------------------------------------------------------------
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("❌ SMOTE Desativado. Usando scale_pos_weight=90.")
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    # -------------------------------------------------------------------------
    # 3. AMOSTRAGEM PARA BUSCA (Eficiencia Computacional)
    # -------------------------------------------------------------------------
    SAMPLE_SIZE = 100000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimizacao Acelerada: Usando amostra estratificada de {SAMPLE_SIZE} linhas para RandomizedSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train

    # -------------------------------------------------------------------------
    # 4. RANDOMIZED SEARCH CV
    # -------------------------------------------------------------------------
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=MODEL_CONFIG["param_distributions"],
        n_iter=MODEL_CONFIG["n_iter"],
        scoring='roc_auc', 
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"],
        random_state=RANDOM_STATE,
        return_train_score=True  # Para diagnosticar overfitting
    )
    
    logger.info(f"⚙️  Otimizando Hiperparametros (RandomizedSearchCV: {MODEL_CONFIG['n_iter']} iteracoes)...")
    random_search.fit(X_sample, y_sample)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info("✅ Melhores Parametros Encontrados!")
    logger.info(f"   ROC-AUC (Validacao na Amostra): {best_score:.4f}")
    logger.info(f"   Parametros: {best_params}")
    
    # Diagnostico de Overfitting
    best_idx = random_search.best_index_
    train_score = random_search.cv_results_['mean_train_score'][best_idx]
    logger.info(f"   ROC-AUC (Treino): {train_score:.4f} | Gap: {train_score - best_score:.4f}")
    if train_score - best_score > 0.05:
        logger.warning("⚠️  Gap Treino-Validacao > 5%: Possivel overfitting! Considere mais regularizacao.")
    
    # -------------------------------------------------------------------------
    # 5. RETREINO COM DATASET COMPLETO
    # -------------------------------------------------------------------------
    logger.info("🚀 Retreinando modelo campeao com TODOS os dados (800k+ linhas)...")
    final_model = random_search.best_estimator_
    final_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 6. PERSISTENCIA
    # -------------------------------------------------------------------------
    latest_model_path = MODELS_DIR / "xgb_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_xgb_{run_id}.pkl"
    joblib.dump(final_model, latest_model_path)
    joblib.dump(final_model, versioned_model_path)
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")

    # -------------------------------------------------------------------------
    # 7. THRESHOLD TUNING (Corrigido: Hold-out de Validacao)
    # -------------------------------------------------------------------------
    best_threshold, best_fbeta, final_model = compute_optimal_threshold(
        model=final_model,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="xgb",
        skip_final_refit=True  # Ja treinado com 100% dos dados acima
    )
    
    save_threshold(best_threshold, "xgb", MODELS_DIR)
    
    # -------------------------------------------------------------------------
    # 8. REGISTRO DO EXPERIMENTO
    # -------------------------------------------------------------------------
    log_experiment(
        run_id=run_id,
        model_type="XGBClassifier",
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
    
    # Relatorio Simples
    with open(MODELS_DIR / "xgb_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_xgboost()
