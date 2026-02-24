# ==============================================================================
# ARQUIVO: random_forest_model.py
#
# OBJETIVO:
#   Configurar e disparar o treinamento do modelo RandomForestClassifier.
#   A lógica baseia-se em um conjunto (Ensemble) de árvores de decisão que utiliza
#   Bagging para reduzir a variância e ser resistente a overfitting.
#
# PARTE DO SISTEMA:
#   Modelagem / Treinamento de Algoritmos Preditivos.
#
# RESPONSABILIDADES:
#   - Parametrizar a malha de hiperparâmetros específicos do Scikit-Learn RandomForest.
#   - Estabelecer os pesamentos de classe corretos ("balanced") lidando com o viés.
#   - Chamar o `BaseTrainer` para finalizar o ciclo de treinamento e log.
#
# COMUNICAÇÃO:
#   - Lê parâmetros aleatórios de ambiente (`src.config`).
#   - Estende a execução a partir do `src.models.trainers.base_trainer`.
# ==============================================================================

import warnings
from sklearn.ensemble import RandomForestClassifier

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

# Configuração de Hiperparâmetros para o Treinador Base
# Intenção: Utilizar 'class_weight="balanced"' impõe diretamente a regressão na folha de 
# maneira Cost-Sensitive, poupando a memória de recriar dados falsos por SMOTE.
MODEL_CONFIG = {
    "model_class": RandomForestClassifier,
    "model_params": {
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    },
    
    "smote_strategy": None,
    "cv_folds": 3,
    
    "search_type": "RandomizedSearchCV",
    "param_distributions": {
        'model__n_estimators': [100, 200, 300, 500],
        'model__max_depth': [5, 10, 15, 20, 30, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 5, 10],
        'model__max_features': ['sqrt', 'log2', 0.3, 0.5],
        'model__class_weight': ['balanced', 'balanced_subsample'],
    },
    
    "n_iter": 40,
    "n_jobs": 1,
    "verbose": 1,
    "sample_size": 200000
}

def train_random_forest():
    """
    Inicializa o treinamento delegando o processo ao BaseTrainer.
    
    - Por que ela existe: Ponto de entrada invocado de forma procedural pelo `main.py`
    - Quando é chamada: No pipeline principal quando `--models rf` é incluído.
    - Estrutura: Instancia a classe unificada de pipeline enviando a `MODEL_CONFIG`
    """
    trainer = BaseTrainer("rf", MODEL_CONFIG)
    trainer.train()

if __name__ == "__main__":
    train_random_forest()
