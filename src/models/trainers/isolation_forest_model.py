# ==============================================================================
# ARQUIVO: isolation_forest_model.py
#
# OBJETIVO:
#   Configurar e disparar o treinamento do IsolationForest. Abordar Fraudes
#   não como uma classificação binária supervisionada, mas sim como uma Tarefa
#   de Detecção de Anomalias Não-Supervisionada estendendo metadados ao pipeline.
#
# PARTE DO SISTEMA:
#   Modelagem / Treinamento de Algoritmos Analíticos (Anomaly Detection).
#
# RESPONSABILIDADES:
#   - Construir o Wrapper (Adapter Pattern) providenciando compatibilidade do 
#     IsolationForest com as curvas PR/ROC que normalmente dependem de `predict_proba`.
#   - Fornecer as distribuições específicas da malha de busca sem pesos definiveis.
#
# COMUNICAÇÃO:
#   - Adaptador customizado `IForestWrapper` extende o Scikit-Learn `BaseEstimator`.
#   - Conecta-se com `BaseTrainer` submetendo a pipeline injetada.
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import RANDOM_STATE
from src.models.trainers.base_trainer import BaseTrainer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class IForestWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper (Adapter Pattern) para adequar IsolationForest ao Pipeline de Threshold.
    
    - Por que existe: O `IsolationForest` nativo não implementa o método `.predict_proba()`
      nem a classe 0-1 (ele retorna -1 Anomalia e 1 Normal), quebrando as curvas do projeto.
    - Ação: Este encapsulador inverte as lógicas (transforma 1 na fraude) e ajusta as
      decisões da anomalia no espectro [0, 1] contínuo, espelhando probabilidade.
    - Quando é chamada: No `fit` e `predict` engatados transparentes pelo RandomCV ou ThresholdUtils.
    """
    def __init__(self, n_estimators=100, contamination='auto', n_jobs=-1, random_state=42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.random_state = random_state
        # Motor que executa real a detecção
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            n_jobs=n_jobs,
            random_state=random_state
        )
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None, **fit_params):
        """
        Treina a floresta isolada e padroniza as anomalias baseadas na distância real negativa calculada.
        """
        self.model.fit(X)
        scores = -self.model.decision_function(X) # Transforma os limites
        self.scaler.fit(scores.reshape(-1, 1))    # Encaixa o scaler pros novos range probabilísticos.
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        """Aloca 1 a -1 (Anomalia = Fraude)"""
        preds = self.model.predict(X)
        return np.where(preds == -1, 1, 0)

    def predict_proba(self, X):
        """
        Retorna simulação da probabilidade baseado no Distanciômetro isolado escalonado em Range(0, 1).
        Exemplo de Saída: array([[0.99, 0.01]])
        """
        decision = -self.model.decision_function(X)
        proba_fraud = self.scaler.transform(decision.reshape(-1, 1)).ravel()
        proba_fraud = np.clip(proba_fraud, 0, 1) # Bloqueia picos raros
        return np.vstack([1 - proba_fraud, proba_fraud]).T

MODEL_CONFIG = {
    "model_class": IForestWrapper,
    "model_params": {
        "n_estimators": 200,
        "contamination": 0.01, # Expectativa natual de fraudes no dataset inteiro é cerca de 1.1%
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    },
    "smote_strategy": None,
    "search_type": None
}

def train_isolation_forest(undersampling_ratio=None):
    """
    Constrói a rotina sem o GridSearch ativado (Anomaly Detection não possui labels reais pra Hyper-tunning 
    num fluxo puro).
    """
    config = MODEL_CONFIG.copy()
    if undersampling_ratio is not None:
        config["undersampling_ratio"] = undersampling_ratio
        
    trainer = BaseTrainer("if", config) # Assuming CustomIFTrainer is a typo and should be BaseTrainer
    trainer.train()

if __name__ == "__main__":
    train_isolation_forest()
