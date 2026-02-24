import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

# ==============================================================================
# ARQUIVO: mlp_model.py
#
# OBJETIVO:
#   Treinar e otimizar uma Rede Neural (Multi-Layer Perceptron).
#   Redes Neurais capturam relacionamentos nao-lineares complexos, mas exigem
#   dados bem escalados (RobustScaler ja cuida disso).
#
# MELHORIAS (v2):
#   - Tratamento de desbalanceamento: MLPClassifier nao suporta class_weight
#     nativamente. Solucao: compute_sample_weight para balancear durante o fit.
#   - RandomizedSearchCV no lugar de GridSearchCV (espaco de busca maior).
#   - Threshold calculado em hold-out de validacao (sem data leakage).
#   - Amostra estratificada para busca de hiperparametros (eficiencia).
#   - Arquiteturas mais variadas incluindo redes mais profundas.
#
# REFERENCIA TECNICA:
#   O MLPClassifier do scikit-learn nao possui parametro class_weight. Para
#   datasets desbalanceados, a alternativa e usar sample_weight no fit(),
#   que o Pipeline repassa ao modelo final via set_params.
#   Referencia: scikit-learn docs "sample_weight" propagation in Pipelines.
#
# PARTE DO SISTEMA:
#   Modulo de Treinamento e Otimizacao (Model Training Stage).
#
# COMUNICACAO:
#   - Le: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/mlp_best_model.pkl, models/mlp_threshold.txt
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
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

# ==============================================================================
# CONFIGURACAO DO MODELO (PARAMETROS)
# ==============================================================================
MODEL_CONFIG = {
    "model_class": MLPClassifier,
    "model_params": {
        "early_stopping": True,      # Para de treinar se nao melhorar
        "validation_fraction": 0.1,  # 10% do treino para early stopping
        "max_iter": 500,             # Maximo de epocas
        "random_state": RANDOM_STATE,
        "n_iter_no_change": 15,      # Paciencia para convergencia
        "warm_start": False,         # Nao reaproveitar pesos de treinos anteriores
    },
    
    "cv_folds": 3,
    
    # Espaco de Busca ENRIQUECIDO (RandomizedSearchCV)
    "param_distributions": {
        'model__hidden_layer_sizes': [
            (64,), (128,), (256,),               # 1 camada
            (128, 64), (256, 128), (64, 32),     # 2 camadas
            (128, 64, 32),                        # 3 camadas (deep)
        ],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01, 0.1],   # Regularizacao L2
        'model__learning_rate_init': [0.0005, 0.001, 0.005, 0.01],
        'model__learning_rate': ['constant', 'adaptive'],  # Adaptativo reduz lr automaticamente
        'model__batch_size': [256, 512],             # Mini-batch para SGD
    },
    
    "n_iter": 20,   # Combinacoes aleatorias (MLP e lento, 20 e suficiente na amostra de 100k)
    "n_jobs": 1,    # CRITICO: n_jobs=-1 com fit_params (sample_weight) trava no Windows
    "verbose": 1
}

def train_mlp():
    """
    Treina o modelo MLP (Neural Network) com tratamento de desbalanceamento.
    
    METODOLOGIA:
    ----------------------
    1. Pipeline Completo: EDAFeatureEngineer -> ColumnTransformer -> MLPClassifier.
    2. Tratamento de desbalanceamento via sample_weight (compute_sample_weight).
       Isto e CRITICO: sem isso, a MLP otimiza para a classe majoritaria (0)
       e ignora fraudes, resultando em recall ~0.01 como acontecia antes.
    3. RandomizedSearchCV com 40 iteracoes para explorar arquiteturas variadas.
    4. Threshold otimizado em hold-out de validacao (sem data leakage).
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline de Treinamento MLP (Run ID: {run_id})...")
    
    # 1. Carga de Dados
    X_train_path = PROCESSED_DATA_DIR / "X_train.csv"
    y_train_path = PROCESSED_DATA_DIR / "y_train.csv"
    
    if not X_train_path.exists():
        logger.error("❌ Arquivos de treino nao encontrados.")
        return

    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    logger.info(f"   Dimensoes: {X_train.shape[0]} amostras, {X_train.shape[1]} features.")
    
    # -------------------------------------------------------------------------
    # 2. TRATAMENTO DE DESBALANCEAMENTO VIA SAMPLE_WEIGHT
    # -------------------------------------------------------------------------
    # MLPClassifier NAO suporta class_weight. A alternativa e calcular
    # sample_weight baseado na frequencia das classes e passar via fit_params.
    # O Pipeline propaga sample_weight para o passo final (model) automaticamente
    # quando usamos o prefixo 'model__sample_weight'.
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    fraud_weight = sample_weights[y_train == 1][0]
    legit_weight = sample_weights[y_train == 0][0]
    logger.info(f"⚖️  Pesos calculados: Fraude={fraud_weight:.2f} | Legit={legit_weight:.2f} (Ratio: {fraud_weight/legit_weight:.1f}x)")
    
    # -------------------------------------------------------------------------
    # 3. DEFINICAO DO PIPELINE
    # -------------------------------------------------------------------------
    clf = MODEL_CONFIG["model_class"](**MODEL_CONFIG["model_params"])
    
    logger.info("🔬 Aplicando Feature Engineering baseado na EDA.")
    pipeline = build_pipeline(X_train, clf)
    
    # -------------------------------------------------------------------------
    # 4. AMOSTRAGEM PARA BUSCA
    # -------------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    
    SAMPLE_SIZE = 100000
    if len(X_train) > SAMPLE_SIZE:
        logger.info(f"⚡ Otimizacao Acelerada: Usando amostra estratificada de {SAMPLE_SIZE} linhas.")
        X_sample, _, y_sample, _ = train_test_split(
            X_train, y_train, train_size=SAMPLE_SIZE, stratify=y_train, random_state=RANDOM_STATE
        )
        # Recalcular sample_weights para a amostra
        sample_weights_sample = compute_sample_weight(class_weight='balanced', y=y_sample)
    else:
        X_sample, y_sample = X_train, y_train
        sample_weights_sample = sample_weights
    
    # -------------------------------------------------------------------------
    # 5. RANDOMIZED SEARCH CV
    # -------------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=MODEL_CONFIG["cv_folds"], shuffle=True, random_state=RANDOM_STATE)
    
    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=MODEL_CONFIG["param_distributions"],
        n_iter=MODEL_CONFIG["n_iter"],
        scoring='roc_auc',
        cv=cv,
        n_jobs=MODEL_CONFIG["n_jobs"],
        verbose=MODEL_CONFIG["verbose"],
        random_state=RANDOM_STATE
    )
    
    logger.info(f"⚡ Iniciando treinamento (Neural Network, {MODEL_CONFIG['n_iter']} combinacoes)...")
    
    # CRITICO: Passa sample_weight via fit_params para o Pipeline
    # O formato 'model__sample_weight' informa ao Pipeline que os pesos
    # devem ser repassados ao passo chamado 'model' (MLPClassifier)
    random_search.fit(X_sample, y_sample, model__sample_weight=sample_weights_sample)
    
    # 6. Resultados
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    logger.info(f"🏆 Melhor ROC-AUC Medio: {best_score:.4f}")
    logger.info(f"🔧 Melhores Parametros: {best_params}")
    
    # -------------------------------------------------------------------------
    # 7. RETREINO COM DATASET COMPLETO + SAMPLE_WEIGHT
    # -------------------------------------------------------------------------
    logger.info("🚀 Retreinando modelo com TODOS os dados...")
    best_model.fit(X_train, y_train, model__sample_weight=sample_weights)
    
    # Salvar Modelo
    latest_model_path = MODELS_DIR / "mlp_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_mlp_{run_id}.pkl"
    joblib.dump(best_model, latest_model_path)
    joblib.dump(best_model, versioned_model_path)
    
    # -------------------------------------------------------------------------
    # 8. THRESHOLD TUNING (Corrigido: Hold-out de Validacao)
    # -------------------------------------------------------------------------
    best_threshold, best_fbeta, best_model = compute_optimal_threshold(
        model=best_model,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="mlp",
        skip_final_refit=True,  # Ja treinado com 100% dos dados acima
        fit_params={'model__sample_weight': sample_weights}
    )
    
    save_threshold(best_threshold, "mlp", MODELS_DIR)
    
    # -------------------------------------------------------------------------
    # 9. REGISTRO DO EXPERIMENTO
    # -------------------------------------------------------------------------
    log_experiment(
        run_id=run_id,
        model_type="MLPClassifier",
        best_params=best_params,
        best_cv_score=best_score,
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None,
        extra_data={
            "search_type": "RandomizedSearchCV",
            "n_iter": MODEL_CONFIG["n_iter"],
            "sample_weight_used": True,
            "fraud_weight_ratio": float(fraud_weight / legit_weight)
        }
    )

if __name__ == "__main__":
    train_mlp()
