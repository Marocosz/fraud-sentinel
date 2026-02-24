import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

# ==============================================================================
# ARQUIVO: lightgbm_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo LightGBM (Light Gradient Boosting Machine).
#
# JUSTIFICATIVA PARA INCLUSAO:
#   O LightGBM e um dos algoritmos mais utilizados na industria para deteccao
#   de fraude por varias razoes:
#   1. VELOCIDADE: Usa histogram-based splitting (O(n*features) -> O(bins*features)),
#      tornando-o 5-20x mais rapido que XGBoost tradicional.
#   2. EFICIENCIA DE MEMORIA: Agrupa valores em bins discretos.
#   3. LEAF-WISE GROWTH: Diferente de level-wise (XGBoost), cresce a folha que
#      reduz mais o loss, convergindo mais rapido.
#   4. SUPORTE NATIVO A CATEGORICAS: Pode tratar categoricas sem One-Hot,
#      mas aqui usamos o pipeline padrao para consistencia.
#   5. RANKING DE COMPETICOES: Top performer em Kaggle para dados tabulares
#      (Fernandez-Delgado et al., 2014; Grinsztajn et al., 2022).
#
# REFERENCIA:
#   Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree."
#   NeurIPS 2017.
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao (Model Training Stage).
#
# COMUNICACAO:
#   - Le: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/lgbm_best_model.pkl, models/lgbm_threshold.txt
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline
from src.models.threshold_utils import compute_optimal_threshold, save_threshold, log_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def get_model_config():
    """
    Retorna a configuracao do modelo LightGBM baseada estruturamente no Pipeline do Projeto.
    
    - Por que ela existe: O LightGBM pode falhar na importação local C++ de SOs antigos ou não instalados. O `import lightgbm`
      fica aqui dentro para evitar que o orquestrador `main.py` dê Crash ao coletar outros módulos que não quebravam.
    - O que recebe: Dinâmico interno.
    - O que retorna: Um dicionário DTO nos mesmos padrões do `MODEL_CONFIG` dos sub-modelos.
    """
    from lightgbm import LGBMClassifier
    
    # Retorno Configuration Object
    # Intenção: Focar em parâmetros assíncronos 'leaf-wise'. Limitamos 'num_leaves' associado à 'max_depth'
    # previnindo overfit absoluto por crescimento denso unilateral da árvore típica deste pacote.
    return {
        "model_class": LGBMClassifier,
        "model_params": {
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbose": -1,      # Desativa output excessivo e logs desnecessários originais da lib.
            "importance_type": "gain",  # Mensurar feature importance não por splits, e sim pelo Information Gain (Purificação)
        },
        
        "smote_strategy": None,
        "cv_folds": 3,
        
        "param_distributions": {
            'model__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'model__n_estimators': [100, 200, 300, 500, 700],
            'model__max_depth': [-1, 3, 5, 7, 10],       
            'model__num_leaves': [15, 31, 63, 127],      
            'model__min_child_samples': [5, 10, 20, 50], 
            'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0], 
            'model__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'model__reg_alpha': [0, 0.01, 0.1, 1.0],     
            'model__reg_lambda': [0, 0.1, 1.0, 5.0],     
        },
        
        "n_iter": 60,
        "n_jobs": 1,     # Deixa 1 pra CV. O paralelismo multithreading acontece dentro da compilação OpenMP do LightGBM puro.
        "verbose": 1
    }


def train_lightgbm():
    """
    Treina o modelo LightGBM com otimizacao completa de hiperparametros.
    *Detalhe Arquitetural: Não herda de `BaseTrainer` porque trata-se de um Script Alternativo/Legado
    para validar customizações finas que o Base ainda não dispunha.*
    
    - O que ela faz: O script lê o conjunto (pickles), roda a malha (RandomSearch), computa Custom Thresholding 
      e descarrega persistência nos volumes JSONL.
    - Quando é chamada: No orquestrador primário `main.py` sob parâmetro `--models lgbm`.
    """
    try:
        MODEL_CONFIG = get_model_config()
    except ImportError:
        logger.error("❌ LightGBM nao instalado. Rode: pip install lightgbm")
        return
    
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento LightGBM (Run ID: {run_id})...")
    
    # -------------------------------------------------------------------------
    # 1. CARGA DE DADOS
    # -------------------------------------------------------------------------
    X_train_path = PROCESSED_DATA_DIR / "X_train.pkl"
    y_train_path = PROCESSED_DATA_DIR / "y_train.pkl"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino nao encontrados.")
        return

    logger.info("📂 Carregando dados de treino...")
    X_train = pd.read_pickle(X_train_path)
    y_train = pd.read_pickle(y_train_path).values.ravel()
    
    logger.info(f"   Dimensoes: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")

    # -------------------------------------------------------------------------
    # 2. PIPELINE (EDA-Driven)
    # -------------------------------------------------------------------------
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    # -------------------------------------------------------------------------
    # 3. AMOSTRAGEM PARA BUSCA
    # -------------------------------------------------------------------------
    SAMPLE_SIZE = 100000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimizacao Acelerada: Usando amostra de {SAMPLE_SIZE} linhas para RandomizedSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train

    # -------------------------------------------------------------------------
    # 4. RANDOMIZED SEARCH CV
    # -------------------------------------------------------------------------
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=MODEL_CONFIG["param_distributions"],
        n_iter=MODEL_CONFIG["n_iter"],
        scoring='roc_auc', 
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"],
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    logger.info(f"⚙️  Otimizando Hiperparametros (RandomizedSearchCV: {MODEL_CONFIG['n_iter']} iteracoes)...")
    random_search.fit(X_sample, y_sample)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info("✅ Melhores Parametros Encontrados!")
    logger.info(f"   ROC-AUC (Validacao na Amostra): {best_score:.4f}")
    logger.info(f"   Parametros: {best_params}")
    
    # Diagnostico de Overfitting
    best_idx = random_search.best_index_
    train_score = random_search.cv_results_['mean_train_score'][best_idx]
    logger.info(f"   ROC-AUC (Treino): {train_score:.4f} | Gap: {train_score - best_score:.4f}")
    
    # -------------------------------------------------------------------------
    # 5. RETREINO COM DATASET COMPLETO
    # -------------------------------------------------------------------------
    logger.info("🚀 Retreinando modelo campeao com TODOS os dados...")
    final_model = random_search.best_estimator_
    final_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 6. PERSISTENCIA
    # -------------------------------------------------------------------------
    latest_model_path = MODELS_DIR / "lgbm_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_lgbm_{run_id}.pkl"
    joblib.dump(final_model, latest_model_path)
    joblib.dump(final_model, versioned_model_path)
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")

    # -------------------------------------------------------------------------
    # 7. THRESHOLD TUNING
    # -------------------------------------------------------------------------
    best_threshold, best_fbeta, final_model = compute_optimal_threshold(
        model=final_model,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="lgbm",
        skip_final_refit=True  # Ja treinado com 100% dos dados acima
    )
    
    save_threshold(best_threshold, "lgbm", MODELS_DIR)
    
    # -------------------------------------------------------------------------
    # 8. REGISTRO DO EXPERIMENTO
    # -------------------------------------------------------------------------
    log_experiment(
        run_id=run_id,
        model_type="LGBMClassifier",
        best_params=best_params,
        best_cv_score=best_score,
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None,
        extra_data={
            "search_type": "RandomizedSearchCV",
            "n_iter": MODEL_CONFIG["n_iter"],
            "train_auc_gap": float(train_score - best_score)
        }
    )
    
    with open(MODELS_DIR / "lgbm_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_lightgbm()
