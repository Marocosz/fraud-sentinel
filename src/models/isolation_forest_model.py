import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score

# ==============================================================================
# ARQUIVO: isolation_forest_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo Isolation Forest (Detecção de Anomalias).
#   
#   DIFERENCIAL TÉCNICO:
#   O Isolation Forest é nativamente NÃO-SUPERVISIONADO (não usa labels para treinar),
#   mas estamos usando em um pipeline supervisionado para avaliação.
#   Criamos um Wrapper (IForestWrapper) para converter a saída de anomalia 
#   (decision_function) em uma "probabilidade" de 0 a 1, permitindo integração
#   com o resto do sistema (predict_model.py, visualize.py).
#
# PARTE DO SISTEMA:
#   Módulo de Treinamento e Otimização.
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import get_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class IForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper para tornar o IsolationForest compatível com pipelines de classificação
    padrão do Scikit-Learn (predict_proba).
    Turn decision_function (score de anomalia) into probability-like score.
    """
    def __init__(self, n_estimators=100, contamination='auto', n_jobs=-1, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            n_jobs=n_jobs,
            random_state=random_state
        )
        self.scaler = MinMaxScaler() # Para normalizar o score entre 0 e 1

    def fit(self, X, y=None):
        # Isolation Forest é não supervisionado, ignora Y no fit do modelo interno
        self.model.fit(X)
        
        # Mas precisamos ajustar o scaler nos scores de decisão para gerar "probabilidades"
        # Score negativo = Anomalia. Invemos para positivo = Prob. Fraude
        scores = -self.model.decision_function(X)
        self.scaler.fit(scores.reshape(-1, 1))
        
        # Salva as classes para compatibilidade com sklearn
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        # Retorna 1 se for fraude (anomalia), 0 se normal
        # IF retorna -1 para anomalia, 1 para normal
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def predict_proba(self, X):
        # Converte decision_function em probabilidade [0, 1]
        decision = -self.model.decision_function(X)
        proba_fraud = self.scaler.transform(decision.reshape(-1, 1)).ravel()
        # Clip para garantir intervalo [0,1]
        proba_fraud = np.clip(proba_fraud, 0, 1)
        
        # Retorna formato (n_samples, 2) -> [prob_0, prob_1]
        return np.vstack([1 - proba_fraud, proba_fraud]).T

def train_isolation_forest():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline Isolation Forest (Run ID: {run_id})...")
    
    # 1. Carga de Dados
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino não encontrados.")
        return

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    # 2. Pipeline
    preprocessor = get_preprocessor(X_train)
    
    # Usamos nosso Wrapper em vez do IF direto
    model = IForestWrapper(n_estimators=200, contamination=0.01, random_state=RANDOM_STATE)
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    logger.info("⚡ Iniciando treinamento (detecção de anomalias)...")
    # Não usamos GridSearch aqui para simplificar, pois IF é não-supervisionado
    # e métricas de CV padrão de classificação não se aplicam diretamente na fase de fit
    pipeline.fit(X_train, y_train)
    
    # 3. Avaliação no Treino (Para registro)
    y_probs = pipeline.predict_proba(X_train)[:, 1]
    auc_score = roc_auc_score(y_train, y_probs)
    
    logger.info(f"🏆 ROC-AUC no Treino (Estimado): {auc_score:.4f}")
    
    # Salvar Modelo
    latest_model_path = MODELS_DIR / "if_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_if_{run_id}.pkl"
    joblib.dump(pipeline, latest_model_path)
    joblib.dump(pipeline, versioned_model_path)
    
    # 4. Threshold Tuning
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    logger.info(f"🎯 Melhor Threshold Normalizado: {best_threshold:.4f}")
    
    with open(MODELS_DIR / "if_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    # 5. Log
    experiment_data = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": "IsolationForest",
        "smote_strategy": None,
        "best_params": {"n_estimators": 200, "contamination": 0.01},
        "best_cv_score": auc_score, # Usando score de treino como proxy
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
        
    # Salvar params simples
    with open(MODELS_DIR / "if_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\nROC-AUC: {auc_score:.4f}\nParams: n_estimators=200, contamination=0.01")

if __name__ == "__main__":
    train_isolation_forest()
