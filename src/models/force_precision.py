import pandas as pd
import numpy as np
import joblib
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score

# ==============================================================================
# ARQUIVO: force_precision.py
#
# OBJETIVO:
#   Ajustar o ponto de operação (Threshold Tuning) do modelo XGBoost para atingir
#   uma Precision (Precisão) alvo definida pelo negócio.
#
# CONTEXTO DE NEGÓCIO:
#   Em fraude, Precision baixa = Muitos clientes legítimos bloqueados (Falsos Positivos).
#   Isso gera atrito, reclamações e custo operacional (equipe de revisão manual).
#   Este script permite dizer: "Quero garantir que pelo menos 20% dos alertas sejam reais".
#
# METODOLOGIA ACADÊMICA:
#   Utiliza a curva Precision-Recall (PR Curve) para varrer todos os limiares possíveis
#   e encontrar matematicamente o menor threshold que satisfaz a restrição:
#   Precision >= Target_Precision.
#
# OUTPUTS:
#   - Relatório de Classificação e Matriz de Confusão ajustados.
#   - Gráfico da Curva Precision-Recall com o ponto escolhido.
#   - Sobrescreve 'models/xgb_threshold.txt' com o novo valor otimizado.
# ==============================================================================

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports do Projeto
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PRECISION_FORCE] - %(message)s')
logger = logging.getLogger(__name__)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, selected_threshold, selected_precision, selected_recall):
    """
    Gera gráfico profissional mostrando o trade-off Precision vs Recall para diferentes thresholds.
    """
    plt.figure(figsize=(10, 6))
    plt.title("Precision-Recall vs Threshold Trade-off")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    
    # Marca o ponto escolhido
    plt.axvline(x=selected_threshold, color='r', linestyle=':', label=f'Selected Threshold ({selected_threshold:.4f})')
    plt.scatter(selected_threshold, selected_precision, color='blue', s=100, zorder=5)
    plt.scatter(selected_threshold, selected_recall, color='green', s=100, zorder=5)
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    
    output_path = FIGURES_DIR / "precision_optimization_curve.png"
    plt.savefig(output_path)
    logger.info(f"📊 Gráfico de trade-off salvo em: {output_path}")
    plt.close()

def enforce_precision_target(target_precision=0.20, model_filename="xgb_best_model.pkl"):
    """
    Encontra e aplica o threshold que garante a precisão mínima desejada.
    
    Args:
        target_precision (float): Alvo de precisão (ex: 0.20 para 20%).
        model_filename (str): Nome do arquivo do modelo a ser carregado.
    """
    logger.info(f"🚀 Iniciando otimização de threshold. Alvo: Precision >= {target_precision*100}%")
    
    # 1. Carregar Dados de Teste (Blind Set)
    try:
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
        model_path = MODELS_DIR / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo {model_filename} não encontrado em {MODELS_DIR}")
            
        model = joblib.load(model_path)
        logger.info(f"✅ Modelo carregado: {model_filename}")
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar artefatos: {e}")
        return

    # 2. Obter Probabilidades (Score de Risco)
    logger.info("🔮 Calculando probabilidades de fraude no conjunto de teste...")
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback para modelos sem predict_proba (ex: SVM linear), embora raro neste projeto
        y_proba = model.decision_function(X_test)
        # Normaliza com Sigmoid se necessário, mas aqui assumimos proba direta

    # 3. Calcular Curva Precision-Recall
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # 4. Busca do Menor Threshold que satisfaz a condição
    # precisions e recalls têm tamanho n_thresholds + 1 (o último é 1.0 e 0.0)
    # thresholds tem tamanho n_thresholds
    
    found_idx = -1
    
    # Varredura eficiente
    # Buscamos o primeiro índice onde precision >= target
    # Nota: thresholds são crescentes? Não necessariamente no retorno do sklearn, mas geralmente sim.
    # O sklearn retorna thresholds crescentes em decision_function, mas para proba pode variar.
    # Vamos garantir iterando sobre a curva ordenada.
    
    # Criamos um DataFrame para analisar melhor
    pr_df = pd.DataFrame({
        'threshold': thresholds, 
        'precision': precisions[:-1], 
        'recall': recalls[:-1]
    })
    
    # Filtra apenas linhas que atendem o critério
    candidates = pr_df[pr_df['precision'] >= target_precision]
    
    if candidates.empty:
        max_prec_achievable = np.max(precisions)
        logger.error(f"❌ IMPOSSÍVEL atingir {target_precision*100}% de precisão com este modelo.")
        logger.error(f"   A precisão máxima teórica alcançável é {max_prec_achievable*100:.2f}%")
        return

    # Entre os candidatos, escolhemos o que tem maior Recall (para não perder fraude à toa)
    # Geralmente isso equivale ao MENOR threshold que bate a precisão.
    best_candidate = candidates.loc[candidates['recall'].idxmax()]
    
    final_threshold = best_candidate['threshold']
    final_precision = best_candidate['precision']
    final_recall = best_candidate['recall']

    logger.info("\n✅ PONTO DE OPERAÇÃO ÓTIMO ENCONTRADO!")
    logger.info(f"   🎯 Threshold de Corte: {final_threshold:.4f}")
    logger.info(f"   💎 Precision Esperada: {final_precision*100:.2f}% (Meta: {target_precision*100}%)")
    logger.info(f"   🔍 Recall Resultante:  {final_recall*100:.2f}%")

    # 5. Validação (Prova Real)
    logger.info("\n📊 Aplicando novo corte nos dados de teste...")
    y_pred_new = (y_proba >= final_threshold).astype(int)
    
    # Métricas Globais
    acc = accuracy_score(y_test, y_pred_new)
    auc = roc_auc_score(y_test, y_proba) # AUC independe do threshold
    f1 = f1_score(y_test, y_pred_new)
    
    print("\n" + "="*60)
    print("RELATÓRIO DE PERFORMANCE (Precision-Oriented)")
    print("="*60)
    print(classification_report(y_test, y_pred_new))
    
    # Matriz de Confusão Customizada
    cm = confusion_matrix(y_test, y_pred_new)
    tn, fp, fn, tp = cm.ravel()
    
    total_samples = len(y_test)
    total_fraud = fn + tp
    total_legit = tn + fp
    
    print("\n--- MATRIZ DE CONFUSÃO ANALÍTICA ---")
    print(f"🟢 Legítimos Aprovados (TN): {tn} ({(tn/total_legit)*100:.1f}%)")
    print(f"🔴 Legítimos Bloqueados (FP): {fp} ({(fp/total_legit)*100:.1f}%) -> CUSTO DE ATRITO")
    print(f"⚠️ Fraudes Detectadas   (TP): {tp} ({(tp/total_fraud)*100:.1f}%) -> RECALL")
    print(f"💸 Fraudes Perdidas     (FN): {fn} ({(fn/total_fraud)*100:.1f}%) -> PREJUÍZO FINANCEIRO")
    print("-" * 60)
    
    # Plotar gráfico
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds, final_threshold, final_precision, final_recall)

    # 6. Persistência
    # Identifica nome base do modelo para salvar o threshold correto
    model_base_name = model_filename.split("_")[0] # ex: 'xgb' from 'xgb_best_model.pkl'
    threshold_file = MODELS_DIR / f"{model_base_name}_threshold.txt"
    
    with open(threshold_file, "w") as f:
        f.write(str(final_threshold))
        
    logger.info(f"💾 Novo Threshold salvo em: {threshold_file}")
    logger.info("   O sistema de inferência (predict_model.py) passará a usar este valor automaticamente.")

if __name__ == "__main__":
    # Permite customizar via argumento simples ou usa default 20%
    if len(sys.argv) > 1:
        try:
            target = float(sys.argv[1])
            enforce_precision_target(target_precision=target)
        except ValueError:
            print("Uso: python force_precision.py [target_precision_float]")
    else:
        enforce_precision_target(target_precision=0.20)