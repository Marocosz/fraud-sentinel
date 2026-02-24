import pandas as pd
import numpy as np
import joblib
import sys
import logging
import warnings
import json
import datetime
from pathlib import Path
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

# ==============================================================================
# ARQUIVO: stacking_model.py
#
# OBJETIVO:
#   Construir um Meta-Modelo (Stacking Ensemble) que combina as predicoes
#   dos melhores modelos individuais para obter performance superior.
#
# FUNDAMENTACAO TEORICA:
#   Stacking (Stacked Generalization) foi proposto por Wolpert (1992).
#   A ideia central e que diferentes modelos capturam padroes diferentes
#   nos dados. Um meta-learner (aqui, Logistic Regression) aprende a
#   combinar essas predicoes de forma otima.
#
#   NIVEL 0 (Base Learners):
#   - XGBoost: Boosting com regularizacao forte (captura relacoes nao-lineares)
#   - LightGBM: Boosting leaf-wise (captura padroes diferentes do XGBoost)
#   - Random Forest: Bagging (captura padroes via reducao de variancia)
#
#   NIVEL 1 (Meta-Learner):
#   - Logistic Regression: Modelo linear simples que aprende os pesos
#     otimos para combinar os base learners. Evita overfitting do ensemble.
#
#   REQUISITO: Os modelos base DEVEM ter sido treinados previamente.
#   Este script carrega os modelos ja treinados e os combina.
#
# REFERENCIA:
#   Wolpert, D. H. (1992). "Stacked Generalization." Neural Networks, 5(2).
#   Van der Laan et al. (2007). "Super Learner." Statistical Applications.
#
# PARTE DO SISTEMA:
#   Modulo de Ensemble e Meta-Aprendizado.
#
# COMUNICACAO:
#   - Le: models/*_best_model.pkl (modelos pre-treinados)
#   - Le: data/processed/X_train.csv, y_train.csv
#   - Escreve: models/stacking_best_model.pkl, models/stacking_threshold.txt
# ==============================================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR
from src.features.build_features import build_pipeline, EDAFeatureEngineer, get_preprocessor
from src.models.threshold_utils import compute_optimal_threshold, save_threshold, log_experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def train_stacking():
    """
    Constroi e treina o Stacking Ensemble a partir dos modelos base pre-treinados.
    
    - O que ela faz: Em cápsula, constrói um "Meta-Classificador". Ela instítui a extração completa 
      de pipelines (K-Fold cruzado) evitando leakages. Por fim treina uma regressão sobre as saídas.
    - Quando é chamada: Pelo main.py após todos os algoritmos primários terem finalizado com sucesso.
    - Por que ela existe: Diferente do Voting, o Stacking *aprende* dinamicamente como somar os votos dependendo das nuances.
    - Como age (Passo a passo):
      1. Extrai o classificador final de cada pipeline gravado na sessão.
      2. Constroi um novo pipeline: EDAFeatureEngineer -> Preprocessor -> Stacking.
      3. O StackingClassifier usa cross-validation interna (cv=5) para gerar
         predicoes out-of-fold dos base learners, que alimentam o meta-learner.
      4. Threshold otimizado em hold-out de validacao.
    """
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Iniciando Pipeline Stacking Ensemble (Run ID: {run_id})...")
    
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
    # 2. CARREGAR MODELOS BASE PRE-TREINADOS
    # -------------------------------------------------------------------------
    base_estimators = []
    
    # XGBoost (obrigatorio)
    xgb_path = MODELS_DIR / "xgb_best_model.pkl"
    if xgb_path.exists():
        xgb_pipeline = joblib.load(xgb_path)
        base_estimators.append(('xgb', xgb_pipeline))
        logger.info(f"✅ XGBoost carregado: {type(xgb_pipeline.named_steps['model']).__name__} (Pipeline Completo)")
    else:
        logger.error("❌ XGBoost nao encontrado. Treine primeiro com: python main.py --models xgb")
        return
    
    # Random Forest (obrigatorio)
    rf_path = MODELS_DIR / "rf_best_model.pkl"
    if rf_path.exists():
        rf_pipeline = joblib.load(rf_path)
        base_estimators.append(('rf', rf_pipeline))
        logger.info(f"✅ Random Forest carregado: {type(rf_pipeline.named_steps['model']).__name__} (Pipeline Completo)")
    else:
        logger.error("❌ Random Forest nao encontrado. Treine primeiro com: python main.py --models rf")
        return
    
    # LightGBM (opcional)
    lgbm_path = MODELS_DIR / "lgbm_best_model.pkl"
    if lgbm_path.exists():
        lgbm_pipeline = joblib.load(lgbm_path)
        base_estimators.append(('lgbm', lgbm_pipeline))
        logger.info(f"✅ LightGBM carregado: {type(lgbm_pipeline.named_steps['model']).__name__} (Pipeline Completo)")
    else:
        logger.warning("⚠️ LightGBM nao encontrado. Prosseguindo sem ele.")
    
    # Logistic Regression (opcional - adiciona diversidade linear)
    logreg_path = MODELS_DIR / "logreg_best_model.pkl"
    if logreg_path.exists():
        logreg_pipeline = joblib.load(logreg_path)
        base_estimators.append(('logreg', logreg_pipeline))
        logger.info(f"✅ Logistic Regression carregado: {type(logreg_pipeline.named_steps['model']).__name__} (Pipeline Completo)")
    
    logger.info(f"📊 Total de Base Learners: {len(base_estimators)}")
    logger.info(f"   Modelos: {[name for name, _ in base_estimators]}")
    
    if len(base_estimators) < 2:
        logger.error("❌ Minimo de 2 modelos base necessarios para Stacking.")
        return
    
    # -------------------------------------------------------------------------
    # 3. CONSTRUCAO DO STACKING ENSEMBLE
    # -------------------------------------------------------------------------
    # Meta-learner: Logistic Regression com regularizacao moderada
    # Usa 'predict_proba' para que os base learners forneçam probabilidades,
    # nao decisoes binarias — isso preserva informacao de incerteza.
    meta_learner = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=RANDOM_STATE
    )
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method='predict_proba',  # Passa probabilidades ao meta-learner
        passthrough=False,              # Nao passa features originais (evita overfitting)
        n_jobs=-1,                      # Utilizando multi-threading otimizado
        verbose=1
    )
    
    # -------------------------------------------------------------------------
    # 4. TREINAMENTO
    # -------------------------------------------------------------------------
    logger.info("⚡ Treinando Stacking Ensemble (isso pode demorar)...")
    logger.info("   O StackingClassifier usa CV interna para gerar predicoes out-of-fold em Pipelines íntegros.")
    stacking_clf.fit(X_train, y_train)
    
    stacking_pipeline = stacking_clf
    
    logger.info("✅ Treinamento do Stacking concluido!")
    
    # -------------------------------------------------------------------------
    # 6. PERSISTENCIA
    # -------------------------------------------------------------------------
    latest_model_path = MODELS_DIR / "stacking_best_model.pkl"
    versioned_model_path = MODELS_DIR / f"model_stacking_{run_id}.pkl"
    joblib.dump(stacking_pipeline, latest_model_path)
    joblib.dump(stacking_pipeline, versioned_model_path)
    logger.info(f"💾 Modelo salvo em: {latest_model_path}")
    
    # -------------------------------------------------------------------------
    # 7. THRESHOLD TUNING (Hold-out de Validacao)
    # -------------------------------------------------------------------------
    best_threshold, best_fbeta, stacking_pipeline = compute_optimal_threshold(
        model=stacking_pipeline,
        X_train=X_train,
        y_train=y_train,
        validation_fraction=0.2,
        random_state=RANDOM_STATE,
        beta=1.0,
        model_name="stacking",
        skip_final_refit=True  # Ja treinado com 100% dos dados acima
    )
    
    joblib.dump(stacking_pipeline, latest_model_path)
    save_threshold(best_threshold, "stacking", MODELS_DIR)
    
    # -------------------------------------------------------------------------
    # 8. REGISTRO DO EXPERIMENTO
    # -------------------------------------------------------------------------
    from sklearn.metrics import roc_auc_score
    y_train_proba = stacking_pipeline.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    log_experiment(
        run_id=run_id,
        model_type="StackingClassifier",
        best_params={
            "base_estimators": [name for name, _ in base_estimators],
            "meta_learner": "LogisticRegression",
            "cv_folds_internal": 5
        },
        best_cv_score=train_auc,  # Proxy (stacking nao usa GridSearch externo)
        best_threshold=best_threshold,
        model_path=versioned_model_path.name,
        reports_dir=REPORTS_DIR,
        smote_strategy=None,
        extra_data={
            "ensemble_type": "stacking",
            "n_base_learners": len(base_estimators)
        }
    )
    
    with open(MODELS_DIR / "stacking_best_model_params.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Base Learners: {[name for name, _ in base_estimators]}\n")
        f.write(f"Meta-Learner: LogisticRegression(C=1.0)\n")
        f.write(f"Optimal Threshold: {best_threshold:.4f}\n")

if __name__ == "__main__":
    train_stacking()
