# ==============================================================================
# ARQUIVO: reg_log_model.py
#
# OBJETIVO:
#   Treinar um modelo de Regressão Logística que age simultaneamente como base 
#   nativa de probabilidade (por natureza estatística) e baseline oficial.
#
# PARTE DO SISTEMA:
#   Modelos de Predição (Machine Learning) Classificatórios Simples.
#
# RESPONSABILIDADES:
#   - Prover um solver (`liblinear`) imune a oscilações em subamostragens.
#   - Servir como termômetro sobre datasets super-balanceados lineares.
#   - Parametrizar otimizadores e reguladores severos L1/L2.
#
# COMUNICAÇÃO:
#   - Pula de fase via `base_trainer.py`.
# ==============================================================================

import warnings
from sklearn.linear_model import LogisticRegression

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RANDOM_STATE
from src.models.trainers.base_trainer import BaseTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*l1_ratio.*")
warnings.filterwarnings("ignore", message=".*penalty.*")

# Configuração e Espaço de Busca
# Intenção: Uma baseline estritamente linear compensada via cost-sensitive learning ('balanced').
MODEL_CONFIG = {
    "model_class": LogisticRegression,
    "model_params": {
        "solver": "liblinear",
        "max_iter": 1000,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
    },
    "smote_strategy": None,
    "cv_folds": 3,
    "search_type": "GridSearchCV",
    "param_grid": {
        'model__C': [0.001, 0.01, 0.1, 1, 10], # Penalidade inversamente proporcional contra overfitting
        'model__penalty': ['l1', 'l2']
    },
    "n_jobs": 1,
    "verbose": 3,
    "sample_size": 100000
}

def train_logistic_regression(undersampling_ratio=None):
    """
    Função empacotadora de execução estendida.
    
    - O que faz: Assoscia a predefinição Logistic ao fluxo unificador abstrato BaseTrainer.
    - Quando invocada: Pipeline Principal ou individual (`--models logreg`).
    """
    config = MODEL_CONFIG.copy()
    if undersampling_ratio is not None:
        config["undersampling_ratio"] = undersampling_ratio
        
    trainer = BaseTrainer("logreg", config)
    trainer.train()

if __name__ == "__main__":
    train_logistic_regression()