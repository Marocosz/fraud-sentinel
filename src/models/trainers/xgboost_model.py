# ==============================================================================
# ARQUIVO: xgboost_model.py
#
# OBJETIVO:
#   Configurar e disparar o treinamento do modelo XGBoost (Extreme Gradient Boosting).
#   Lida especificamente com o tuning e parametrização exigida pelo XGBoost para 
#   lidar com dados desbalanceados.
#
# PARTE DO SISTEMA:
#   Modelagem / Treinamento de Algoritmos Preditivos.
#
# RESPONSABILIDADES:
#   - Configurar o dicionário de parâmetros específicos do XGBClassifier.
#   - Definir a malha de hiperparâmetros (RandomizedSearchCV) para ajuste fino.
#   - Orquestrar o treinamento delegando o fluxo principal para o `BaseTrainer`.
#
# COMUNICAÇÃO:
#   - Lê configurações gerais do `src.config`.
#   - Aciona o `BaseTrainer` localizado em `src.models.trainers.base_trainer`.
# ==============================================================================

import warnings
from xgboost import XGBClassifier

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RANDOM_STATE
from src.models.trainers.base_trainer import BaseTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Configuração de Hiperparâmetros para o Treinador Base
# Intenção: Centralizar em um objeto todas as especificações que distinguem o XGBoost dos demais modelos.
# O scale_pos_weight foi ativado ativamente porque o modelo sofrerá com o desbalanceamento das classes, 
# agindo como um método de Cost-Sensitive Learning onde erros na minoritária (fraude) pesam mais.
MODEL_CONFIG = {
    "model_class": XGBClassifier,
    "model_params": {
        "eval_metric": "logloss",
        "scale_pos_weight": 90, # Compensador estatístico do desequilíbrio massivo (1% fraude)
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "tree_method": "hist",
    },
    
    "smote_strategy": None,
    "cv_folds": 3,
    
    "search_type": "RandomizedSearchCV",
    "param_distributions": {
        'model__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [3, 4, 5, 6, 7, 8],
        'model__min_child_weight': [1, 3, 5, 7],
        'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'model__gamma': [0, 0.1, 0.3, 0.5, 1.0],
        'model__reg_alpha': [0, 0.01, 0.1, 1.0],
        'model__reg_lambda': [0.1, 0.5, 1.0, 5.0],
    },
    
    "n_iter": 60,
    "n_jobs": 1,
    "verbose": 1,
    "sample_size": 100000
}

def train_xgboost(undersampling_ratio=None):
    """
    Inicializa o treinamento delegando o processo ao BaseTrainer.
    
    - Por que ela existe: Encapsular a execucao em uma funcao importavel atraves do orquestrador main.py.
    - Quando e chamada: No decorrer da execucao rotineira do torneio ou quando o argumento CLI `--models xgb` e passado.
    - Dados recebidos: undersampling_ratio opcional.
    - Dados retornados: Nenhum. Efetua a gravacao de arquivos por debaixo dos panos pelo BaseTrainer.
    """
    config = MODEL_CONFIG.copy()
    if undersampling_ratio is not None:
        config["undersampling_ratio"] = undersampling_ratio
        
    trainer = BaseTrainer("xgb", config)
    trainer.train()

if __name__ == "__main__":
    train_xgboost()
