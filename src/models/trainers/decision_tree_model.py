# ==============================================================================
# ARQUIVO: decision_tree_model.py
#
# OBJETIVO:
#   Configurar e disparar o treinamento de um modelo DecisionTreeClassifier (Árvore de Decisão Simples).
#   Serve como uma baseline interpretável e base para algoritmos de Ensemble (como Random Forest).
#
# PARTE DO SISTEMA:
#   Modelagem / Treinamento de Algoritmos Preditivos.
#
# RESPONSABILIDADES:
#   - Especificar a grade de hiperparâmetros (GridSearchCV) focada em poda de árvore (max_depth, min_samples)
#     para evitar o sobreajuste (overfitting) clássico das árvores puras.
#   - Setar `class_weight` balanceado mitigando dados raros sem utilizar Oversampling.
#
# COMUNICAÇÃO:
#   - Carrega as variáveis imutáveis de `src.config`.
#   - Instancia e utiliza ativamente o encapsulador de fluxo `BaseTrainer` em `base_trainer.py`.
# ==============================================================================

import warnings
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from .base_trainer import BaseTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Configuração de Hiperparâmetros
# Intenção: Criar a configuração de árvore única. É propositalmente mantida simples para
# contrastar a complexidade temporal do Random Forest, servindo como modelo interpretável (White-box).
MODEL_CONFIG = {
    "model_class": DecisionTreeClassifier,
    "model_params": {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE
    },
    "smote_strategy": None,
    "cv_folds": 3,
    "search_type": "GridSearchCV", # Busca exata para árvores devido ao baixo espaço amostral das possibilidades
    "param_grid": {
        'model__max_depth': [5, 10, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__criterion': ['gini', 'entropy'],
        'model__max_features': ['sqrt', 'log2'],
    },
    "n_jobs": 1,
    "verbose": 2,
    "sample_size": 200000
}

def train_decision_tree(undersampling_ratio=None):
    """
    Desencadeia a rotina de preenchimento, processamento, busca exaustiva (GridCV) e escoragem
    pela classe BaseTrainer generalizada.
    
    - Por que ela existe: Ser o módulo-gatilho importado pelo `main.py`
    - Quando é chamada: No fluxo de Torneio ou seleção manual CLI por `--models dt`.
    """
    config = MODEL_CONFIG.copy()
    if undersampling_ratio is not None:
        config["undersampling_ratio"] = undersampling_ratio
        
    trainer = BaseTrainer("dt", config)
    trainer.train()

if __name__ == "__main__":
    train_decision_tree()
