import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

# ==============================================================================
# ARQUIVO: decision_tree_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo de Arvore de Decisao.
#
# MELHORIAS (v2):
#   - Threshold calculado em hold-out de validacao (sem data leakage).
#   - Grid reduzido para viabilidade computacional.
#   - Amostragem para GridSearch (100k linhas).
#   - Logging centralizado via threshold_utils.
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao (Model Training Stage).
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

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

MODEL_CONFIG = {
    "model_class": DecisionTreeClassifier,
    "model_params": {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    },
    "smote_strategy": None,
    "cv_folds": 3,
    "param_grid": {
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__criterion': ['gini', 'entropy'],
        'model__max_features': ['sqrt', 'log2'],
    },
    "n_jobs": 1,
    "verbose": 2
}

def train_decision_tree():
    """
    Treina o modelo de Arvore de Decisao com otimizacao completa.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline Decision Tree (Run ID: {run_id})...")
    
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino nao encontrados.")
        return

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    logger.info(f"   Dimensoes: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")

    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    logger.info("❌ SMOTE Desativado. Usando class_weight='balanced'.")
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=MODEL_CONFIG["param_grid"],
        scoring='roc_auc',
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"]
    )
    
    # Amostragem para busca (Decision Tree e rapido, mas 800k x 648 fits e demais)
    SAMPLE_SIZE = 200000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimizacao Acelerada: Usando amostra de {SAMPLE_SIZE} linhas para GridSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train
    
    logger.info("⚙️  Otimizando Hiperparametros...")
    grid_search.fit(X_sample, y_sample)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"🏆 Melhor ROC-AUC Medio: {best_score:.4f}")
    logger.info(f"🔧 Melhores Parametros: {best_params}")
    
    # Retreinar no dataset completo
    logger.info("🚀 Retreinando modelo com TODOS os dados...")
    best_model.fit(X_train, y_train)
    
    latest_model_path = MODELS_DIR / "dt_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_dt_{run_id}.pkl"
    joblib.dump(best_model, latest_model_path)
    joblib.dump(best_model, versioned_model_path)
    
    # Threshold (skip refit, ja treinado acima)
    best_threshold, best_fbeta, best_model = compute_optimal_threshold(
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="dt",
        skip_final_refit=True
    )
    
    save_threshold(best_threshold, "dt", MODELS_DIR)
    
    log_experiment(
        run_id=run_id,
        model_type="DecisionTreeClassifier",
        best_params=best_params,
        best_cv_score=best_score,
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None
    )
    
    with open(MODELS_DIR / "dt_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\nBest ROC-AUC: {best_score:.4f}\nOptimal Threshold: {best_threshold:.4f}\nParams: {best_params}\n")

if __name__ == "__main__":
    train_decision_tree()
