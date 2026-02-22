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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

# ==============================================================================
# ARQUIVO: decision_tree_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo de Árvore de Decisão.
#
# PARTE DO SISTEMA:
#   Módulo de Treinamento e Otimização (Model Training Stage).
#
# COMUNICAÇÃO:
#   - Lê: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/dt_best_model.pkl
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

MODEL_CONFIG = {
    "model_class": DecisionTreeClassifier,
    "model_class": DecisionTreeClassifier,
    "model_params": {
        "class_weight": "balanced", # [MODIFICAÇÃO] Reativado devido remoção do SMOTE
        "random_state": RANDOM_STATE
    },
    "smote_strategy": None,
    "cv_folds": 3,
    "param_grid": {
        'model__max_depth': [5, 10, None],
        'model__min_samples_split': [2, 5],
        'model__criterion': ['gini', 'entropy']
    },
    "n_jobs": 1,
    "verbose": 2
}

def train_decision_tree():
    """
    Treina o modelo de Árvore de Decisão com otimização completa.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline Decision Tree (Run ID: {run_id})...")
    
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino não encontrados.")
        return

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Pipeline (EDA-Driven: Feature Engineering + Preprocessing + Modelo)
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("❌ SMOTE Desativado. Usando class_weight='balanced'.")
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=MODEL_CONFIG["param_grid"],
        scoring='roc_auc', cv=cv, n_jobs=MODEL_CONFIG["n_jobs"], verbose=MODEL_CONFIG["verbose"]
    )
    
    logger.info("⚙️  Otimizando Hiperparâmetros...")
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"🏆 Melhor ROC-AUC Médio: {best_score:.4f}")
    
    latest_model_path = MODELS_DIR / "dt_best_model.pkl"
    joblib.dump(best_model, latest_model_path)
    versioned_model_path = MODELS_DIR / f"model_dt_{run_id}.pkl"
    joblib.dump(best_model, versioned_model_path)
    
    year_now = datetime.datetime.now().year
    
    # 6. THRESHOLD TUNING (F1-Optimization)
    logger.info("⚖️  Calculando Best Threshold (F1-Score)...")
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    logger.info(f"🎯 Threshold Ideal: {best_threshold:.4f}")
    with open(MODELS_DIR / "dt_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    experiment_data = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": MODEL_CONFIG["model_class"].__name__,
        "smote_strategy": MODEL_CONFIG["smote_strategy"],
        "best_params": best_params,
        "best_cv_score": best_score,
        "best_threshold": float(best_threshold),
        "model_path": str(versioned_model_path.name)
    }
    
    experiments_log_path = REPORTS_DIR / "experiments_log.json"
    if experiments_log_path.exists():
        with open(experiments_log_path, "r") as f:
            try: history = json.load(f)
            except: history = []
    else:
        history = []
    history.append(experiment_data)
    with open(experiments_log_path, "w") as f: json.dump(history, f, indent=4)
        
    with open(MODELS_DIR / "dt_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_decision_tree()
