import numpy as np
import json
import datetime
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, roc_auc_score

# ==============================================================================
# ARQUIVO: threshold_utils.py
#
# OBJETIVO:
#   Centralizar a lógica de Threshold Tuning e Experiment Logging para todos os
#   modelos do projeto. Corrige o problema CRITICO de Data Leakage que existia
#   quando o threshold era calculado usando predict_proba no proprio conjunto
#   de treino.
#
# METODOLOGIA ACADEMICA:
#   O threshold otimo deve ser calculado em um conjunto de VALIDACAO que o modelo
#   nao viu durante o treino. Isso evita overfitting do threshold e garante que
#   o F1-Score estimado seja proximo da realidade em dados novos.
#
#   Referencia: "Threshold-Moving Approach for Fair Classification"
#   (Menon & Williamson, 2018) e "Optimal Threshold Selection for Binary
#   Classifiers" (Freeman & Moisen, 2008).
#
# ESTRATEGIA:
#   1. Apos o modelo ser treinado (fit completo), separamos um hold-out de  
#      validacao (20% estratificado) do conjunto de TREINO.
#   2. Retreinamos o modelo final nos outros 80%.
#   3. Calculamos predict_proba no hold-out e encontramos o threshold que
#      maximiza F1-Score ou F2-Score (configuravel).
#   4. Retreinamos o modelo final com 100% dos dados para producao.
#
#   ALTERNATIVA SIMPLIFICADA (usada aqui):
#   Como o GridSearchCV ja faz cross-validation, usamos uma abordagem mais
#   eficiente: aplicamos o modelo treinado no dataset completo mas usando
#   cross_val_predict para obter predicoes out-of-fold, evitando data leakage.
#   NAO: Usamos um hold-out de validacao separado (mais conservador e robusto).
#
# PARTE DO SISTEMA:
#   Modulo de Utilidades de Treinamento (Shared Training Utilities).
# ==============================================================================

logger = logging.getLogger(__name__)


def compute_optimal_threshold(
    model, 
    X_train, 
    y_train, 
    validation_fraction=0.2, 
    random_state=42,
    beta=1.0,
    model_name="model",
    skip_final_refit=False,
    fit_params=None
):
    """
    Calcula o threshold otimo usando um hold-out de validacao separado.
    
    CORRECAO CRITICA:
    Versao anterior calculava threshold usando predict_proba(X_train), ou seja,
    no mesmo dado usado para treinar. Isso produz thresholds otimistas que nao
    generalizam. Esta versao usa um hold-out interno para estimar o threshold
    de forma honesta.
    
    ESTRATEGIA:
    1. Separa 20% do X_train como validacao (estratificado).
    2. Retreina o modelo nos 80% restantes.
    3. Calcula pred_proba no hold-out e otimiza threshold.
    4. (Opcional) Retreina no dataset COMPLETO para producao.
    
    Args:
        model: Pipeline/modelo treinado (com .fit() e .predict_proba()).
        X_train: DataFrame de treino completo.
        y_train: Array de labels completo.
        validation_fraction: Fracao do treino para validacao (default: 0.2).
        random_state: Seed para reproducibilidade.
        beta: Peso do Recall no F-beta score. 
              beta=1 -> F1 (equilibrio Precision/Recall).
              beta=2 -> F2 (Recall 2x mais importante que Precision).
        model_name: Nome do modelo (para logging).
        skip_final_refit: Se True, NAO refita o modelo no dataset completo.
              Use quando o caller ja fez fit no dataset completo antes de chamar
              esta funcao, para evitar fits redundantes e poupar tempo.
        fit_params: Dicionario com parametros extras para o fit() (ex: sample_weight
              para MLP que nao suporta class_weight).
    
    Returns:
        tuple: (best_threshold, best_fbeta, model)
    """
    logger.info(f"⚖️  [{model_name.upper()}] Calculando Threshold Otimo (hold-out de validacao)...")
    logger.info(f"   Estrategia: {int((1-validation_fraction)*100)}% treino / {int(validation_fraction*100)}% validacao | F-beta (beta={beta})")
    
    # -------------------------------------------------------------------------
    # 1. SPLIT INTERNO (Estratificado)
    # -------------------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=validation_fraction,
        stratify=y_train,
        random_state=random_state
    )
    
    logger.info(f"   Hold-out: {len(X_tr)} treino / {len(X_val)} validacao (Fraude: {y_val.mean()*100:.2f}%)")
    
    # -------------------------------------------------------------------------
    # 2. RETREINAR NO SUBCONJUNTO (para threshold honesto)
    # -------------------------------------------------------------------------
    from sklearn.base import clone
    model_for_threshold = clone(model)
    
    if fit_params:
        # Recalcular sample_weight para o subset
        from sklearn.utils.class_weight import compute_sample_weight
        subset_fit_params = {}
        for key, val in fit_params.items():
            if 'sample_weight' in key:
                subset_fit_params[key] = compute_sample_weight(class_weight='balanced', y=y_tr)
            else:
                subset_fit_params[key] = val
        model_for_threshold.fit(X_tr, y_tr, **subset_fit_params)
    else:
        model_for_threshold.fit(X_tr, y_tr)
    
    # -------------------------------------------------------------------------
    # 3. PREVER NO HOLD-OUT
    # -------------------------------------------------------------------------
    y_val_proba = model_for_threshold.predict_proba(X_val)[:, 1]
    
    # -------------------------------------------------------------------------
    # 4. OTIMIZAR THRESHOLD (F-beta Score)
    # -------------------------------------------------------------------------
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)
    
    # F-beta Score: F_beta = (1 + beta^2) * (P * R) / (beta^2 * P + R)
    beta_sq = beta ** 2
    fbeta_scores = (1 + beta_sq) * (precisions * recalls) / (beta_sq * precisions + recalls + 1e-10)
    
    best_idx = np.argmax(fbeta_scores)
    best_threshold = float(thresholds[best_idx])
    best_fbeta = float(fbeta_scores[best_idx])
    
    # Metricas adicionais no hold-out para contexto
    val_auc = roc_auc_score(y_val, y_val_proba)
    y_val_pred = (y_val_proba >= best_threshold).astype(int)
    val_f1 = f1_score(y_val, y_val_pred)
    
    from sklearn.metrics import precision_score, recall_score
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    
    logger.info(f"🎯 Threshold Otimo: {best_threshold:.4f}")
    logger.info(f"   Metricas no Hold-out de Validacao:")
    logger.info(f"   ROC-AUC: {val_auc:.4f} | F1: {val_f1:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
    
    # -------------------------------------------------------------------------
    # 5. RETREINAR NO DATASET COMPLETO (para producao) — APENAS SE NECESSARIO
    # -------------------------------------------------------------------------
    if not skip_final_refit:
        logger.info(f"🚀 Retreinando modelo final com 100% dos dados ({len(X_train)} amostras)...")
        if fit_params:
            model.fit(X_train, y_train, **fit_params)
        else:
            model.fit(X_train, y_train)
    else:
        logger.info(f"⏩ Refit no dataset completo pulado (modelo ja treinado com 100% dos dados).")
    
    return best_threshold, best_fbeta, model


def save_threshold(threshold, model_prefix, models_dir):
    """
    Salva o threshold otimizado em arquivo texto.
    
    Args:
        threshold: Valor do threshold.
        model_prefix: Prefixo do modelo (e.g., 'logreg', 'xgb').
        models_dir: Diretorio de modelos.
    """
    threshold_path = models_dir / f"{model_prefix}_threshold.txt"
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    logger.info(f"✅ Threshold salvo em: {threshold_path}")
    return threshold_path


def load_threshold(model_prefix, models_dir, default=0.5):
    """
    Carrega o threshold otimizado de um arquivo.
    Retorna o default (0.5) se o arquivo nao existir.
    
    Args:
        model_prefix: Prefixo do modelo (e.g., 'logreg', 'xgb').
        models_dir: Diretorio de modelos.
        default: Threshold padrao caso arquivo nao exista.
    
    Returns:
        float: Threshold carregado.
    """
    threshold_path = models_dir / f"{model_prefix}_threshold.txt"
    if threshold_path.exists():
        try:
            threshold = float(threshold_path.read_text().strip())
            return threshold
        except (ValueError, IOError):
            logger.warning(f"⚠️ Erro ao ler threshold de {threshold_path}, usando default {default}")
            return default
    else:
        logger.warning(f"⚠️ Threshold file nao encontrado: {threshold_path}, usando default {default}")
        return default


def log_experiment(
    run_id,
    model_type,
    best_params,
    best_cv_score,
    best_threshold,
    model_path,
    reports_dir,
    smote_strategy=None,
    extra_data=None
):
    """
    Registra um experimento no log centralizado (experiments_log.json).
    
    Elimina a duplicacao de codigo de logging que existia em cada modelo.
    
    Args:
        run_id: ID unico do experimento (timestamp).
        model_type: Nome da classe do modelo.
        best_params: Dicionario de melhores parametros.
        best_cv_score: Melhor score de validacao cruzada.
        best_threshold: Threshold otimizado.
        model_path: Caminho do modelo salvo.
        reports_dir: Diretorio de relatorios.
        smote_strategy: Estrategia SMOTE usada (None se desativado).
        extra_data: Dicionario com dados adicionais a registrar.
    """
    experiment_data = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "model_type": model_type,
        "smote_strategy": smote_strategy,
        "best_params": best_params,
        "best_cv_score": float(best_cv_score),
        "best_threshold": float(best_threshold),
        "model_path": str(model_path) if not isinstance(model_path, str) else model_path
    }
    
    # Adiciona dados extras se fornecidos
    if extra_data:
        experiment_data.update(extra_data)
    
    experiments_log_path = reports_dir / "experiments_log.json"
    
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
    return experiments_log_path
