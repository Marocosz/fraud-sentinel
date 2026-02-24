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
from sklearn.model_selection import train_test_split

# ==============================================================================
# ARQUIVO: isolation_forest_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo Isolation Forest (Deteccao de Anomalias).
#   
# DIFERENCIAL TECNICO:
#   O Isolation Forest e nativamente NAO-SUPERVISIONADO (nao usa labels),
#   mas estamos usando em um pipeline supervisionado para avaliacao.
#   O IForestWrapper converte a saida de anomalia (decision_function) em 
#   uma "probabilidade" de 0 a 1, permitindo integracao com o sistema.
#
# MELHORIAS (v2):
#   - Threshold calculado em hold-out de validacao (sem data leakage).
#   - Logging centralizado via threshold_utils.
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao.
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline, EDAFeatureEngineer, get_preprocessor
from src.models.threshold_utils import save_threshold, log_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class IForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper para tornar o IsolationForest compativel com pipelines de classificacao
    padrao do Scikit-Learn (predict_proba).
    Converte decision_function (score de anomalia) em probabilidade.
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
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        self.model.fit(X)
        scores = -self.model.decision_function(X)
        self.scaler.fit(scores.reshape(-1, 1))
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def predict_proba(self, X):
        decision = -self.model.decision_function(X)
        proba_fraud = self.scaler.transform(decision.reshape(-1, 1)).ravel()
        proba_fraud = np.clip(proba_fraud, 0, 1)
        return np.vstack([1 - proba_fraud, proba_fraud]).T


def train_isolation_forest():
    """
    Treina o Isolation Forest com threshold otimizado em validacao.
    
    NOTA: Isolation Forest e nao-supervisionado, entao nao faz sentido usar
    GridSearchCV com ROC-AUC (pois nao usa labels no fit). O threshold, porem,
    deve ser calculado em dados nao vistos para evitar data leakage.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline Isolation Forest (Run ID: {run_id})...")
    
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino nao encontrados.")
        return

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    
    model = IForestWrapper(n_estimators=200, contamination=0.01, random_state=RANDOM_STATE)
    pipeline = build_pipeline(X_train, model)
    
    # -------------------------------------------------------------------------
    # THRESHOLD: Usar hold-out separado
    # -------------------------------------------------------------------------
    logger.info("⚖️  Separando hold-out de validacao para threshold...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    
    logger.info(f"⚡ Treinando no subset ({len(X_tr)} amostras)...")
    pipeline.fit(X_tr, y_tr)
    
    # Avaliar no hold-out
    y_val_probs = pipeline.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_probs)
    logger.info(f"🏆 ROC-AUC no Hold-out de Validacao: {val_auc:.4f}")
    
    # Threshold otimo no hold-out
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    
    logger.info(f"🎯 Melhor Threshold (Validacao): {best_threshold:.4f}")
    
    # -------------------------------------------------------------------------
    # RETREINAR COM 100% DOS DADOS
    # -------------------------------------------------------------------------
    logger.info(f"🚀 Retreinando com 100% dos dados ({len(X_train)} amostras)...")
    pipeline.fit(X_train, y_train)
    
    # Salvar
    latest_model_path = MODELS_DIR / "if_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_if_{run_id}.pkl"
    joblib.dump(pipeline, latest_model_path)
    joblib.dump(pipeline, versioned_model_path)
    
    save_threshold(best_threshold, "if", MODELS_DIR)
    
    # Log
    log_experiment(
        run_id=run_id,
        model_type="IsolationForest",
        best_params={"n_estimators": 200, "contamination": 0.01},
        best_cv_score=val_auc,
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None,
        extra_data={"evaluation_method": "hold-out validation"}
    )
    
    with open(MODELS_DIR / "if_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\nROC-AUC (Validation): {val_auc:.4f}\n")
        f.write(f"Threshold: {best_threshold:.4f}\nParams: n_estimators=200, contamination=0.01\n")

if __name__ == "__main__":
    train_isolation_forest()
