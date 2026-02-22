import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import contextlib
import json
import datetime
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ==============================================================================
# ARQUIVO: random_forest_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo de Random Forest.
#   Este script foca na otimização de hiperparâmetros especificamente para este algoritmo.
#
# PARTE DO SISTEMA:
#   Módulo de Treinamento e Otimização (Model Training Stage).
#
# RESPONSABILIDADES:
#   - Carregar o dataset de treino (X_train, y_train).
#   - Definir o espaço de busca de hiperparâmetros (Grid Search).
#   - Executar a busca com validação cruzada para garantir robustez.
#   - Persistir o melhor modelo encontrado (.pkl).
#   - Registrar logs detalhados do processo de treinamento.
#
# COMUNICAÇÃO:
#   - Lê: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/rf_best_model.pkl
#   - Escreve: models/rf_best_model_params.txt
# ==============================================================================

# Ignora avisos de depreciação do Scikit-Learn e pkg_resources
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports do Projeto
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline

# Configuração de Logs (Profissionalismo)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURAÇÃO DO MODELO (PARÂMETROS)
# ==============================================================================
MODEL_CONFIG = {
    # Modelo Base
    "model_class": RandomForestClassifier,
    "model_params": {
        "n_estimators": 100,
        "class_weight": "balanced", # [MODIFICAÇÃO] Reativado para compensar a falta do SMOTE
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    },
    
    # Estratégia de Oversampling
    "smote_strategy": None,
    
    # Validação Cruzada
    "cv_folds": 3,                  # Rápido e suficiente para grandes volumes
    
    # Espaço de Busca (Grid Search)
    "param_grid": {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None],
        'model__min_samples_split': [2, 5]
    },
    
    # Configuração de Execução
    "n_jobs": 1,                    # GridSearch jobs (o RF já usa jobs internos)
    "verbose": 2
}

def train_random_forest():
    """
    Treina o modelo de Random Forest com otimização completa de hiperparâmetros.
    """
    
    # Gerar ID único para o experimento (Timestamp)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento Random Forest (Run ID: {run_id})...")
    logger.info(f"ℹ️  Configuração carregada: SMOTE={MODEL_CONFIG['smote_strategy']}")
    
    # -------------------------------------------------------------------------
    # 1. CARGA DE DADOS
    # -------------------------------------------------------------------------
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino não encontrados. Rode 'make_dataset.py' primeiro.")
        return

    logger.info("📂 Carregando dados de treino...")
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    logger.info(f"   Dimensões: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")

    # -------------------------------------------------------------------------
    # 2. DEFINIÇÃO DO PIPELINE (EDA-DRIVEN)
    # -------------------------------------------------------------------------
    # Pipeline de 3 etapas: EDAFeatureEngineer -> ColumnTransformer -> Modelo
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("❌ SMOTE Desativado. Usando class_weight='balanced'.")
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    # -------------------------------------------------------------------------
    # 3. ESPAÇO DE HIPERPARÂMETROS (Grid Search)
    # -------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=MODEL_CONFIG["param_grid"],
        scoring='roc_auc', 
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"]
    )
    
    # -------------------------------------------------------------------------
    # 4. TREINAMENTO E OTIMIZAÇÃO
    # -------------------------------------------------------------------------
    logger.info("⚙️  Otimizando Hiperparâmetros (GridSearchCV)...")
    logger.info(f"   Espaço de busca: {MODEL_CONFIG['param_grid']}")
    
    print(f"\n⚡ Iniciando treinamento...")
    grid_search.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 5. RESULTADOS E PERSISTÊNCIA
    # -------------------------------------------------------------------------
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info("✅ Treinamento Concluído!")
    logger.info(f"🏆 Melhor ROC-AUC Médio: {best_score:.4f}")
    logger.info(f"🔧 Melhores Parâmetros: {best_params}")
    
    # 1. Salvar Modelo Final (Versão Atual/Latest para o sistema usar)
    latest_model_path = MODELS_DIR / "rf_best_model.pkl"
    joblib.dump(best_model, latest_model_path)
    
    # 2. Salvar Modelo Versionado (Histórico)
    versioned_model_path = MODELS_DIR / f"model_rf_{run_id}.pkl"
    joblib.dump(best_model, versioned_model_path)
    
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")
    logger.info(f"💾 Cópia de histórico salva em: {versioned_model_path}")
    
    # -------------------------------------------------------------------------
    # 6. THRESHOLD TUNING (F1-Score Maximization)
    # -------------------------------------------------------------------------
    logger.info("⚖️  Calculando Best Threshold...")
    
    # Previsões de probabilidade no treino
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    
    # Calcula F1 para cada threshold
    # Adicionamos epsilon para evitar divisão por zero
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"🎯 Melhor Threshold: {best_threshold:.4f} (F1 esperado: {best_f1:.4f})")
    
    # Salvar threshold
    with open(MODELS_DIR / "rf_threshold.txt", "w") as f:
        f.write(str(best_threshold))

    # 3. Registrar Experimento no Log (JSON)
    experiment_data = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": MODEL_CONFIG["model_class"].__name__,
        "smote_strategy": MODEL_CONFIG["smote_strategy"],
        "best_params": best_params,
        "best_cv_score": best_score,
        "best_threshold": float(best_threshold),
        "model_path": str(versioned_model_path.name)
    }
    
    experiments_log_path = REPORTS_DIR / "experiments_log.json"
    
    # Lê o log existente ou cria lista vazia
    if experiments_log_path.exists():
        with open(experiments_log_path, "r") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
        
    history.append(experiment_data)
    
    with open(experiments_log_path, "w") as f:
        json.dump(history, f, indent=4)
        
    logger.info(f"📝 Experimento registrado em: {experiments_log_path}")
    
    # Salvar Relatório Simples
    with open(MODELS_DIR / "rf_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_random_forest()
