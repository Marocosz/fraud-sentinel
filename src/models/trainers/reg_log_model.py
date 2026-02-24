import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

# ==============================================================================
# ARQUIVO: reg_log_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo de Regressao Logistica.
#
# MELHORIAS (v2):
#   - Threshold calculado em hold-out de validacao (sem data leakage).
#   - Logging centralizado via threshold_utils.
#   - Removido fit redundante (1 fit no dataset completo ao inves de 3).
#   - Solver volta para 'liblinear' (muito mais rapido para 800k linhas).
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao (Model Training Stage).
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*l1_ratio.*")
warnings.filterwarnings("ignore", message=".*penalty.*")

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
# CONFIGURACAO DO MODELO
# ==============================================================================
MODEL_CONFIG = {
    "model_class": LogisticRegression,
    "model_params": {
        "solver": "liblinear",       # Rapido para datasets ate ~1M linhas
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "smote_strategy": None,
    "cv_folds": 3,
    "param_grid": {
        'model__C': [0.001, 0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2']
    },
    "n_jobs": 1,
    "verbose": 3
}

def train_logistic_regression():
    """
    Treina a Regressao Logistica com busca de hiperparametros.
    Fluxo otimizado: GridSearch na amostra -> Refit completo -> Threshold no hold-out.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento Otimizado (Run ID: {run_id})...")
    logger.info(f"ℹ️  Configuracao: SMOTE={MODEL_CONFIG['smote_strategy']}, Jobs={MODEL_CONFIG['n_jobs']}")
    
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino nao encontrados.")
        return

    logger.info("📂 Carregando dados de treino...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    logger.info(f"   Dimensoes: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")

    # Pipeline
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    logger.info("❌ SMOTE Desativado. Usando class_weight='balanced'.")
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    # Amostragem para busca
    SAMPLE_SIZE = 100000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimizacao Acelerada: Usando amostra estratificada de {SAMPLE_SIZE} linhas para GridSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train

    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=MODEL_CONFIG["param_grid"],
        scoring='roc_auc', 
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"]
    )
    
    logger.info("⚙️  Otimizando Hiperparametros (GridSearchCV na Amostra)...")
    logger.info(f"   Espaco de busca: {MODEL_CONFIG['param_grid']}")
    
    print(f"\n⚡ Iniciando busca de parametros (n_jobs={MODEL_CONFIG['n_jobs']})...")
    grid_search.fit(X_sample, y_sample)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info("✅ Melhores Parametros Encontrados!")
    logger.info(f"   ROC-AUC (Validacao na Amostra): {best_score:.4f}")
    logger.info(f"   Parametros: {best_params}")

    # Retreinar com dataset completo
    logger.info("🚀 Retreinando modelo campeao com TODOS os dados (800k+ linhas)...")
    final_model = grid_search.best_estimator_
    final_model.fit(X_train, y_train)
    
    # Persistencia
    latest_model_path = MODELS_DIR / "logreg_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_logreg_{run_id}.pkl"
    joblib.dump(final_model, latest_model_path)
    joblib.dump(final_model, versioned_model_path)
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")
    
    # Threshold (skip_final_refit=True pois ja fizemos fit acima)
    best_threshold, best_fbeta, final_model = compute_optimal_threshold(
        model=final_model,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="logreg",
        skip_final_refit=True  # Ja treinado com 100% dos dados acima
    )
    
    save_threshold(best_threshold, "logreg", MODELS_DIR)
    
    # Registro
    log_experiment(
        run_id=run_id,
        model_type="LogisticRegression",
        best_params=best_params,
        best_cv_score=best_score,
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None
    )
    
    with open(MODELS_DIR / "best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\nBest ROC-AUC: {best_score:.4f}\nOptimal Threshold: {best_threshold:.4f}\nParams: {best_params}\n")

if __name__ == "__main__":
    train_logistic_regression()