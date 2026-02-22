import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

# ==============================================================================
# ARQUIVO: mlp_model.py
#
# OBJETIVO:
#   Treinar e otimizar uma Rede Neural (Multi-Layer Perceptron).
#   Redes Neurais capturam relacionamentos não-lineares complexos, mas exigem
#   dados bem escalados (RobustScaler já cuida disso).
#
# PARTE DO SISTEMA:
#   Módulo de Treinamento e Otimização (Model Training Stage).
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURAÇÃO DO MODELO (PARÂMETROS)
# ==============================================================================
MODEL_CONFIG = {
    "model_class": MLPClassifier,
    "model_params": {
        "early_stopping": True,     # Para de treinar se não melhorar (evita overfitting)
        "validation_fraction": 0.1, # 10% do treino usado para early stopping
        "max_iter": 500,            # Máximo de épocas
        "random_state": RANDOM_STATE
    },
    
    "cv_folds": 3,
    
    # Espaço de Busca (Grid Search)
    "param_grid": {
        'model__hidden_layer_sizes': [(50,), (100,), (50, 25)], # Arquiteturas
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01], # Regularização L2
        'model__learning_rate_init': [0.001, 0.01]
    },
    
    "n_jobs": -1,
    "verbose": 2
}

def train_mlp():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento MLP (Run ID: {run_id})...")
    
    # 1. Carga de Dados
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino não encontrados.")
        return

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    # 2. Pipeline (EDA-Driven: Feature Engineering + Preprocessing + Modelo)
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    # 3. Grid Search
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=MODEL_CONFIG["param_grid"],
        scoring='roc_auc',
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"]
    )
    
    logger.info("⚡ Iniciando treinamento (Neural Network)...")
    grid_search.fit(X_train, y_train)
    
    # 4. Resultados
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info(f"🏆 Melhor ROC-AUC Médio: {best_score:.4f}")
    logger.info(f"🔧 Melhores Parâmetros: {best_params}")
    
    # Salvar Modelo
    latest_model_path = MODELS_DIR / "mlp_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_mlp_{run_id}.pkl"
    joblib.dump(best_model, latest_model_path)
    joblib.dump(best_model, versioned_model_path)
    
    # 5. Threshold Tuning
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    logger.info(f"🎯 Melhor Threshold: {best_threshold:.4f}")
    
    with open(MODELS_DIR / "mlp_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    # 6. Log
    experiment_data = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "MLPClassifier",
        "smote_strategy": None,
        "best_params": best_params,
        "best_cv_score": best_score,
        "best_threshold": float(best_threshold),
        "model_path": str(versioned_model_path.name)
    }
    
    experiments_log_path = REPORTS_DIR / "experiments_log.json"
    if experiments_log_path.exists():
        with open(experiments_log_path, "r") as f:
            try:
                history = json.load(f)
            except: history = []
    else:
        history = []
        
    history.append(experiment_data)
    with open(experiments_log_path, "w") as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    train_mlp()
