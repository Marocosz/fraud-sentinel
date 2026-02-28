# ==============================================================================
# ARQUIVO: base_trainer.py
#
# OBJETIVO:
#   Abstrair a lógica de treinamento para toda a suíte de algoritmos do sistema.
#   Unifica Split, Busca de Hiperparâmetros, Threshold Tuning, Escalabilidade de Pesos
#   e Persistência, acabando com a duplicação de dezenas de linhas a cada novo modelo.
#
# PARTE DO SISTEMA:
#   Módulo Core de Treinamento e Escoragem (ML Preditivo).
#
# RESPONSABILIDADES:
#   - Encapsular os dados (`_load_data`) e subamostragem (`_get_sample`).
#   - Decidir se aplica GridSearchCV ou RandomizedSearchCV transparentemente.
#   - Embutir cálculo avançado de Cost-Sensitive Learning (`compute_sample_weight`).
#   - Chamar o Tuning Curva ROC-PRC (`compute_optimal_threshold`).
#   - Registrar métricas definitivas (`log_experiment`).
#
# COMUNICAÇÃO:
#   - Ponto OBRIGATÓRIO de passagem de todos os algoritmos na pasta `/trainers/`.
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import sys
import logging
import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline
from src.models.threshold_utils import compute_optimal_threshold, save_threshold, log_experiment

logger = logging.getLogger(__name__)

class BaseTrainer:
    """
    Classe Abstrata de Treinamento e Otimização Unificada (Design Pattern: Template Method).
    
    Por que existe:
    Resolver o problema de duplicação de lógica (DRY - Don't Repeat Yourself). Antes de MLOps, cada 
    novo algoritmo (XGBoost, Random Forest, MLP) repetia a mesma calha de tratamento de Pipeline, 
    cross-validation, sampling, tuning de logs e exportações. Esta classe orquestra a cadeia 
    mestra e deixa o algoritmo se conectar apenas via 'config'.

    Responsabilidade Crítica (Integração):
    Ao instanciar, ela encapsula a busca do conjunto Treino/Teste na base local, aciona módulos externos 
    para Cost-Sensitive Learning e delega ao threshold_utils.py a extração da probabilidade ideal de 
    Ponto de Corte (Anti-Falsos Positivos) antes de selar o `.pkl`.
    """
    
    def __init__(self, model_prefix: str, config: Dict[str, Any]):
        """
        Construtor da Classe Mestra de Modelagem.
        
        Como atua: 
        Realiza Injeção de Dependências acoplando a tag temporal única (Run ID) para o rastreio
        laboratorial (Experiment Logging) de longo prazo.

        Recebe: 
          - model_prefix (str): Etiqueta curta que nomeará os `.pkl` gerados (ex: 'xgb', 'rf').
          - config (Dict): DTO de malha contendo os hiperparâmetros (param_distributions), ratio 
            de imbalanced learn, CPU cores (n_jobs) e tipo de search (Random/Grid).
        """
        self.model_prefix = model_prefix
        self.config = config
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        X_train_path = PROCESSED_DATA_DIR / "X_train.pkl"
        y_train_path = PROCESSED_DATA_DIR / "y_train.pkl"
        if not X_train_path.exists():
            raise FileNotFoundError("Arquivos de treino nao encontrados.")
        
        X_train = pd.read_pickle(X_train_path)
        y_train = pd.read_pickle(y_train_path).values.ravel()
        return X_train, y_train

    def _get_sample(self, X_train: pd.DataFrame, y_train: np.ndarray, sample_size: int) -> Tuple[pd.DataFrame, np.ndarray]:
        if len(X_train) > sample_size:
            X_sample, _, y_sample, _ = train_test_split(
                X_train, y_train, train_size=sample_size, stratify=y_train, random_state=RANDOM_STATE
            )
            return X_sample, y_sample
        return X_train, y_train

    def train(self) -> BaseEstimator:
        """
        Rotina principal de Treinamento.
        
        Por que existe:
        Centraliza todo o ciclo analítico de Machine Learning num só gatilho assíncrono.
        
        Fluxo Lógico Interno:
        1. Resgate dos Pickle de disco.
        2. Determina se o Algoritmo tentará lidar com o desbalanceamento brutal do Banco por "Corte Fisico"
           (Undersampling ImbLearn) ou "Corte Matemático" (Cost-Sensitive Learning com Sample Weights). O script
           aborta o Cost-Sensitive se perceber que um UnderSampling já atenuou as métricas (para evitar mismatch).
        3. Realiza cortes aleatórios para fôlego de processamento da nuvem (se o dataset tiver 1 milhão de linhas,
           usar todo ele no GridSearchCV levaria semanas de computação).
        4. Otimiza os hiperparâmetros pela grade Random/Grid.
        5. Reconstrói o Modelo Campeão cego ensinando TODO o dataset original à ele (Full Retrain) 
           para retenção permanente de aprendizado.
        6. Persiste as métricas matemáticas, o `.pkl` bruto e as Thresholds na pasta rastreável `reports/`.
        
        Retorna:
        BaseEstimator: Objeto pipeline sklearn vivo, engatilhado na melhor configuração de limiares prováveis.
        """
        logger.info(f"🚀 Iniciando Pipeline: {self.config.get('model_class').__name__} (Run ID: {self.run_id})")
        
        # 1. Isolamento de Leitura
        X_train, y_train = self._load_data()
        
        # 2. Criação Modular Estrutural
        clf = self.config["model_class"](**self.config.get("model_params", {}))
        undersampling_ratio = self.config.get("undersampling_ratio", None)
        pipeline = build_pipeline(X_train, clf, undersampling_ratio=undersampling_ratio)
        
        use_sample_weight = self.config.get("use_sample_weight", False)
        # BUGFIX de Integração Analítica: Evita aplicar pesos matemáticos caso o Undersampling 
        # Físico (Remoção temporal de registros de Bons Clientes) já tenha equalizado a balança.
        if undersampling_ratio is not None:
            use_sample_weight = False

        sample_weights = None
        if use_sample_weight:
            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # 3. Otimização Volumétrica (Sampling para Grid)
        sample_size = self.config.get("sample_size", 100000)
        X_sample, y_sample = self._get_sample(X_train, y_train, sample_size)
        
        sample_weights_sample = None
        if use_sample_weight:
            sample_weights_sample = compute_sample_weight(class_weight='balanced', y=y_sample)
            
        cv = StratifiedKFold(n_splits=self.config.get("cv_folds", 3), shuffle=True, random_state=RANDOM_STATE)
        
        # 4. Hyperparameter Search
        search_type = self.config.get("search_type", "RandomizedSearchCV")
        if search_type == "GridSearchCV":
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=self.config.get("param_grid", {}),
                scoring='roc_auc',
                cv=cv,
                n_jobs=self.config.get("n_jobs", 1),
                verbose=self.config.get("verbose", 1)
            )
        elif search_type == "RandomizedSearchCV":
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=self.config.get("param_distributions", {}),
                n_iter=self.config.get("n_iter", 10),
                scoring='roc_auc',
                cv=cv,
                n_jobs=self.config.get("n_jobs", 1),
                verbose=self.config.get("verbose", 1),
                random_state=RANDOM_STATE,
                return_train_score=True
            )
        else:
            search = None
            
        fit_params_search = {}
        if use_sample_weight and search is not None:
            fit_params_search['model__sample_weight'] = sample_weights_sample

        best_params = {}
        best_score = 0.0
        best_model = pipeline
        
        if search is not None:
            search.fit(X_sample, y_sample, **fit_params_search)
            best_model = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
            logger.info(f"🏆 Melhor ROC-AUC Medio (Amostra): {best_score:.4f}")
            logger.info(f"🔧 Melhores Parametros: {best_params}")
            
        # 5. Full Retrain
        logger.info("🚀 Retreinando modelo campeao com TODOS os dados...")
        fit_params_full = {}
        if use_sample_weight:
            fit_params_full['model__sample_weight'] = sample_weights
            
        best_model.fit(X_train, y_train, **fit_params_full)
        
        # 6. Persistence
        latest_model_path = MODELS_DIR / f"{self.model_prefix}_best_model.pkl"
        versioned_model_path = MODELS_DIR / f"model_{self.model_prefix}_{self.run_id}.pkl"
        joblib.dump(best_model, latest_model_path)
        joblib.dump(best_model, versioned_model_path)
        
        # 7. Optimal Threshold
        best_threshold, best_fbeta, final_model = compute_optimal_threshold(
            model=best_model,
            X_train=X_train,
            y_train=y_train,
            validation_fraction=0.2,
            random_state=RANDOM_STATE,
            beta=1.0,
            model_name=self.model_prefix,
            skip_final_refit=True,
            fit_params=fit_params_full if use_sample_weight else None
        )
        save_threshold(best_threshold, self.model_prefix, MODELS_DIR)
        
        # 8. Experiment Tracking
        extra_info = {
            "sample_size_used_for_search": len(X_sample),
            "undersampling_ratio": self.config.get("undersampling_ratio", None)
        }
        log_experiment(
            run_id=self.run_id,
            model_type=self.config["model_class"].__name__,
            best_params=best_params,
            best_cv_score=best_score if search is not None else 0.0,
            best_threshold=best_threshold,
            model_path=versioned_model_path.name,
            reports_dir=REPORTS_DIR,
            smote_strategy=self.config.get("smote_strategy", None),
            extra_data=extra_info
        )
        
        # Simple text report
        with open(MODELS_DIR / f"{self.model_prefix}_best_model_params.txt", "w") as f:
            f.write(f"Run ID: {self.run_id}\nROC-AUC: {best_score:.4f}\nThreshold: {best_threshold:.4f}\nParams: {best_params}\n")
        
        logger.info(f"✅ Treinamento concluido para {self.model_prefix}!")
        return final_model
