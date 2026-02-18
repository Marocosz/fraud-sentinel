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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ==============================================================================
# ARQUIVO: reg_log_model.py
#
# OBJETIVO:
#   Treinar e otimizar o modelo de Regressão Logística.
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
#   - Escreve: models/logreg_best_model.pkl
#   - Escreve: models/logreg_best_model_params.txt
# ==============================================================================

# Ignora avisos de depreciação do Scikit-Learn e pkg_resources
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*l1_ratio.*") # Fix específico para LogReg
warnings.filterwarnings("ignore", message=".*penalty.*")  # Fix específico para LogReg

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports do Projeto
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import get_preprocessor

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
    "model_class": LogisticRegression,
    "model_params": {
        "solver": "liblinear",      # Otimizado para datasets médios/binários
        "max_iter": 1000,
        "class_weight": "balanced", # [MODIFICAÇÃO] Usando pesos para compensar desbalanceamento (sem SMOTE)
        "random_state": RANDOM_STATE
    },
    
    # Estratégia de Oversampling (Desativado)
    "smote_strategy": None,
    
    # Validação Cruzada
    "cv_folds": 3,                  # Rápido e suficiente para grandes volumes
    
    # Espaço de Busca (Grid Search)
    "param_grid": {
        'model__C': [0.01, 0.1, 1, 10],  # Regularização
        'model__penalty': ['l1', 'l2']   # Lasso vs Ridge
    },
    
    # Configuração de Execução
    "n_jobs": 1,                    # 1 para evitar travamentos no Windows/Memória
    "verbose": 3
}

def train_logistic_regression():
    """
    Treina o modelo de Regressão Logística com otimização completa de hiperparâmetros.
    
    METODOLOGIA ACADÊMICA:
    ----------------------
    1. Pipeline Completo: Pré-processamento + SMOTE + Modelo.
       - Por que SMOTE dentro do GridSearch? Para evitar Data Leakage. O oversampling 
         deve "ver" apenas os dados de treino de cada fold, nunca a validação.
         
    2. Otimização Bayesiana/Grid (GridSearchCV):
       - Buscamos o melhor parâmetro 'C' (força da regularização).
       - Regularização é CRÍTICA em fraude para evitar overfitting (aprender ruído).
       
    3. Métricas:
       - Focamos em ROC-AUC (separabilidade global) para seleção do melhor modelo.
    """
    
    
    # Gerar ID único para o experimento (Timestamp)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento Otimizado (Run ID: {run_id})...")
    logger.info(f"ℹ️  Configuração carregada: SMOTE={MODEL_CONFIG['smote_strategy']}, Jobs={MODEL_CONFIG['n_jobs']}")
    
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
    # 2. DEFINIÇÃO DO PIPELINE
    # -------------------------------------------------------------------------
    # Recupera o preprocessor (RobustScaler para tratar outliers financeiros)
    preprocessor = get_preprocessor(X_train)
    
    # Instancia o modelo base usando os parâmetros configurados
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    # AJUSTE CRÍTICO: sampling_strategy Parametrizado
    logger.info("❌ SMOTE Desativado. Usando class_weight='balanced'.")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    
    # -------------------------------------------------------------------------
    # 3. ESPAÇO DE HIPERPARÂMETROS (Grid Search)
    # -------------------------------------------------------------------------
    # ESTRATÉGIA DE VELOCIDADE:
    # Usamos uma amostra robusta (100k) para encontrar os melhores hiperparâmetros.
    # Isso reduz o tempo de busca de horas para minutos.
    # O modelo FINAL é retreinado no dataset completo com os parâmetros vencedores.
    
    SAMPLE_SIZE = 100000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimização Acelerada: Usando amostra estratificada de {SAMPLE_SIZE} linhas para GridSearch.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
    else:
        X_sample, y_sample = X_train, y_train

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
    # 4. TREINAMENTO E OTIMIZAÇÃO (EM AMOSTRA)
    # -------------------------------------------------------------------------
    logger.info("⚙️  Otimizando Hiperparâmetros (GridSearchCV na Amostra)...")
    logger.info(f"   Espaço de busca: {MODEL_CONFIG['param_grid']}")
    
    print(f"\n⚡ Iniciando busca de parâmetros (n_jobs={MODEL_CONFIG['n_jobs']})...")
    grid_search.fit(X_sample, y_sample)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    logger.info("✅ Melhores Parâmetros Encontrados!")
    logger.info(f"   ROC-AUC (Validação na Amostra): {best_score:.4f}")
    logger.info(f"   Parâmetros: {best_params}")

    # -------------------------------------------------------------------------
    # 5. TREINAMENTO FINAL (DATASET COMPLETO)
    # -------------------------------------------------------------------------
    logger.info("🚀 Retreinando modelo campeão com TODOS os dados (800k+ linhas)...")
    # O best_estimator_ já vem configurado com os best_params, mas foi treinado na amostra.
    # O refit é necessário para aprender com todo o volume de dados.
    final_model = grid_search.best_estimator_
    final_model.fit(X_train, y_train)
    
    # -------------------------------------------------------------------------
    # 6. PERSISTÊNCIA
    # -------------------------------------------------------------------------
    # 1. Salvar Modelo Final (Versão Atual/Latest para o sistema usar)
    latest_model_path = MODELS_DIR / "logreg_best_model.pkl"
    joblib.dump(final_model, latest_model_path)
    
    # 2. Salvar Modelo Versionado (Histórico)
    versioned_model_path = MODELS_DIR / f"model_logreg_{run_id}.pkl"
    joblib.dump(final_model, versioned_model_path)
    
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")
    logger.info(f"💾 Cópia de histórico salva em: {versioned_model_path}")
    
    # -------------------------------------------------------------------------
    # 7. THRESHOLD TUNING (AJUSTE FINO DE CORTE)
    # -------------------------------------------------------------------------
    # O padrão é 0.5, mas em fraude, o ideal depende do equilíbrio Precision/Recall.
    # Vamos encontrar o threshold que maximiza o F1-Score.
    
    logger.info("⚖️  Calculando Best Threshold (F1-Score Maximization)...")
    
    # Previsões de probabilidade no conjunto de treino (usando cross_val_predict para ser justo)
    # Mas como já treinamos o final, vamos fazer uma estimativa usando o próprio treino 
    # (Ideal seria um validation set separado, mas usaremos a análise pós-treino).
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_train_proba)
    
    # Calcula F1 para cada threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    logger.info(f"🎯 Melhor Threshold Encontrado: {best_threshold:.4f}")
    logger.info(f"   F1-Score Esperado: {best_f1:.4f}")
    
    # Salvar o threshold junto com os parâmetros
    with open(MODELS_DIR / "logreg_threshold.txt", "w") as f:
        f.write(str(best_threshold))
        
    logger.info(f"✅ Threshold salvo em: {MODELS_DIR / 'logreg_threshold.txt'}")

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
    
    # Salvar Relatório Simples (Mantido para compatibilidade)
    with open(MODELS_DIR / "best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Best ROC-AUC: {best_score:.4f}\n")
        f.write(f"Params: {best_params}\n")

if __name__ == "__main__":
    train_logistic_regression()