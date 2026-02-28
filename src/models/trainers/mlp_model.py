# ==============================================================================
# ARQUIVO: mlp_model.py
#
# OBJETIVO:
#   Treinar e otimizar um Multi-Layer Perceptron (Rede Neural Feedforward).
#   Busca extrair padrões fortemente não lineares que modelos de árvore podem deixar escapar,
#   atuando como o "Veto Supervisor" de alta precisão cirúrgica no Comitê Final (Ensemble).
#
# PARTE DO SISTEMA:
#   Modelagem Preditiva Avançada / Deep Learning Básico (MLOps).
#
# RESPONSABILIDADES:
#   - Adaptar os parâmetros da classe Scikit-Learn `MLPClassifier`.
#   - Acoplar `early_stopping` para evitar catástrofes de Overfitting.
#   - Disparar métricas de penalidade na classe de Fraude (Sample Weights).
#
# INTEGRAÇÕES:
#   - Instanciado pelo core `main.py` sob a tag `--models mlp`.
#   - Despacha o dicionário de treinamento para a Fábrica Base `BaseTrainer`.
# ==============================================================================

import warnings
from sklearn.neural_network import MLPClassifier

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

# Configuração Preditiva Global
# Intenção MLOps: O MLP requer ajustes severos em hiperparâmetros de regularização para convergir
# nas instâncias minoritárias. Habilitamos early_stopping para podar iterações divergentes
# e use_sample_weight para forçar a atenção nas fraudes (pesando-as durante o Backpropagation).
MODEL_CONFIG = {
    "model_class": MLPClassifier,
    "model_params": {
        "early_stopping": True, # Desarmar gradientes divergentes poupando memória computacional.
        "validation_fraction": 0.1,
        "max_iter": 500,
        "random_state": RANDOM_STATE,
        "n_iter_no_change": 15,
        "warm_start": False,
    },
    
    "smote_strategy": None,
    "cv_folds": 3,
    
    "search_type": "RandomizedSearchCV",
    "param_distributions": {
        'model__hidden_layer_sizes': [
            (64,), (128,), (256,),
            (128, 64), (256, 128), (64, 32),
            (128, 64, 32),
        ],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01, 0.1],
        'model__learning_rate_init': [0.0005, 0.001, 0.005, 0.01],
        'model__learning_rate': ['constant', 'adaptive'],
        'model__batch_size': [256, 512],
    },
    
    "n_iter": 20,
    "n_jobs": 1,
    "verbose": 1,
    "sample_size": 100000,
    "use_sample_weight": True
}

def train_mlp(undersampling_ratio=None):
    """
    Função de delegação da inferência que embute as configurações na Orquestração da classe base.
    
    - Por que ela existe: Ser a peça única de compatibilidade com chamadas abstratas e CLI's via subrotinas.
    - Quando é chamada: Pela bateria contínua no `--models mlp` de Torneios.
    """
    config = MODEL_CONFIG.copy()
    if undersampling_ratio is not None:
        config["undersampling_ratio"] = undersampling_ratio
        
    trainer = BaseTrainer("mlp", config)
    trainer.train()

if __name__ == "__main__":
    train_mlp()
