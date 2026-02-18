import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import contextlib
import json
import datetime
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ==============================================================================
# ARQUIVO: xgboost_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo XGBoost.
#
# PARTE DO SISTEMA:
#   Módulo de Treinamento e Otimização (Model Training Stage).
#
# RESPONSABILIDADES:
#   - Carregar o dataset de treino.
#   - Otimizar hiperparâmetros do XGBoost.
#   - Persistir modelo e logs.
#
# COMUNICAÇÃO:
#   - Lê: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/xgb_best_model.pkl
#   - Escreve: models/xgb_best_model_params.txt
# ==============================================================================

# Ignora avisos
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports do Projeto
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import get_preprocessor

# Configuração de Logs
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
    # Modelo Base
    "model_class": XGBClassifier,
    "model_params": {
        "eval_metric": "logloss",
        "scale_pos_weight": 90, # [MODIFICAÇÃO] Peso alto (90:1) para compensar falta de SMOTE
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    },
    
    # Estratégia de Oversampling
    "smote_strategy": None,
    
    # Validação Cruzada
    "cv_folds": 3,
    
    # Espaço de Busca
    "param_grid": {
        'model__learning_rate': [0.01, 0.1], 
        'model__n_estimators': [100, 200],
        'model__max_depth': [3, 6]
    },
    
    # Configuração de Execução
    "n_jobs": 1,
    "verbose": 2
}

def train_xgboost():
    """
    Treina o modelo XGBoost com otimização completa de hiperparâmetros.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento XGBoost (Run ID: {run_id})...")
    
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino não encontrados.")
        return

    logger.info("📂 Carregando dados de treino...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    preprocessor = get_preprocessor(X_train)
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("❌ SMOTE Desativado. Usando scale_pos_weight=90.")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    # ESTRATÉGIA DE VELOCIDADE:
    # Usamos uma amostra robusta (100k) para encontrar os melhores hiperparâmetros.
    SAMPLE_SIZE = 100000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimização Acelerada: Usando amostra estratificada de {SAMPLE_SIZE} linhas para GridSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=MODEL_CONFIG["param_grid"],
        scoring='roc_auc', 
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"]
    )
    
    logger.info("⚙️  Otimizando Hiperparâmetros (GridSearchCV na Amostra)...")
    grid_search.fit(X_sample, y_sample)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info("✅ Melhores Parâmetros Encontrados!")
    logger.info(f"   ROC-AUC (Validação na Amostra): {best_score:.4f}")
    
    logger.info("🚀 Retreinando modelo campeão com TODOS os dados (800k+ linhas)...")
    final_model = grid_search.best_estimator_
    final_model.fit(X_train, y_train)
    
    # 1. Salvar Modelo Final
    latest_model_path = MODELS_DIR / "xgb_best_model.pkl"
    joblib.dump(final_model, latest_model_path)
    
    # 2. Salvar Modelo Versionado
    versioned_model_path = MODELS_DIR / f"model_xgb_{run_id}.pkl"
    joblib.dump(final_model, versioned_model_path)
    
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")

    # -------------------------------------------------------------------------
    # 7. THRESHOLD TUNING
    # -------------------------------------------------------------------------
    logger.info("⚖️  Calculando Best Threshold (F1-Optimal)...")
    
    # Previsões de probabilidade no dataset COMPLETO de treino
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    
    # F1 Score = 2 * (P * R) / (P + R)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"🎯 Threshold Ideal: {best_threshold:.4f} (F1: {best_f1:.4f})")
    
    # Salvar threshold
    with open(MODELS_DIR / "xgb_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    # 3. Registrar Experimento
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
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
        
    history.append(experiment_data)
    
    with open(experiments_log_path, "w") as f:
        json.dump(history, f, indent=4)
    
    with open(MODELS_DIR / "xgb_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_xgboost()
