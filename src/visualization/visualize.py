# ==============================================================================
# ARQUIVO: visualize.py
#
# OBJETIVO:
#   Realizar a avaliacao final e detalhada da performance do modelo escolhido.
#   Gera visualizacoes criticas (Curva ROC, Curva PR, Matriz de Confusao,
#   Importancia de Features) para validar se o modelo esta pronto para producao.
#
# MELHORIAS (v2):
#   CORRECAO CRITICA: Agora aplica o threshold otimizado salvo durante o treino.
#   A versao anterior chamava model.predict(X_test) que usa threshold fixo de 0.5,
#   ignorando completamente o threshold tuning feito por cada modelo. Isso resultava
#   em metricas enganosas (precision ~4%).
#
#   ADICOES:
#   - Comparacao lado-a-lado: Default (0.5) vs Threshold Otimizado.
#   - PR-AUC (Precision-Recall AUC) como metrica adicional.
#     Em dados desbalanceados (1.1% fraude), PR-AUC e mais informativa que ROC-AUC
#     porque foca na performance da classe positiva (minoritaria).
#     Referencia: Saito & Rehmsmeier (2015) - "The Precision-Recall Plot is More 
#     Informative than the ROC Plot When Evaluating Binary Classifiers."
#   - Feature Importance para modelos de arvore (XGBoost, LightGBM, RF).
#
# PARTE DO SISTEMA:
#   Modulo de Avaliacao e Monitoramento (Model Evaluation Stage).
#
# COMUNICACAO:
#   - Le: models/*_best_model.pkl, models/*_threshold.txt
#   - Le: data/processed/X_test.csv, y_test.csv
#   - Escreve: reports/figures/*.png
# ==============================================================================

import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # Backend nao-interativo para evitar problemas em servidores
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import json
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score
)

# Adiciona raiz ao path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR
from src.models.threshold_utils import load_threshold


def plot_coefficients(model, feature_names, model_name="model"):
    """
    Plota os coeficientes da Regressao Logistica OU feature importances
    para modelos de arvore (XGBoost, LightGBM, RF, DT).
    """
    try:
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['model']
        else:
            classifier = model

        # Caso 1: Modelo linear (LogReg)
        if hasattr(classifier, 'coef_'):
            coefs = classifier.coef_[0]
            coef_df = pd.DataFrame({'Feature': feature_names, 'Importance': coefs})
            coef_df['Abs_Importance'] = coef_df['Importance'].abs()
            coef_df = coef_df.sort_values(by='Abs_Importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=coef_df, x='Importance', y='Feature', palette='vlag')
            plt.title(f"Top 20 Features (Coeficientes) - {model_name.upper()}")
            plt.axvline(0, color='black', linewidth=0.8)
            plt.tight_layout()
            
            save_path = FIGURES_DIR / f"feature_importance_{model_name}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"📊 Grafico de Coeficientes salvo em: {save_path}")
            
        # Caso 2: Modelo de arvore (XGBoost, LightGBM, RF, DT)
        elif hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Garante que feature_names e importances tenham o mesmo tamanho
            if len(feature_names) != len(importances):
                print(f"⚠️ Tamanho de features ({len(feature_names)}) != importances ({len(importances)}). Pulando grafico.")
                return
            
            imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            imp_df = imp_df.sort_values(by='Importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis')
            plt.title(f"Top 20 Features (Importancia) - {model_name.upper()}")
            plt.tight_layout()
            
            save_path = FIGURES_DIR / f"feature_importance_{model_name}.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"📊 Grafico de Feature Importance salvo em: {save_path}")
            
        # Caso 3: Stacking ou MLP (sem feature importances diretas)
        else:
            print(f"ℹ️  Modelo {model_name} nao possui coeficientes/importances diretas.")
            
    except Exception as e:
        print(f"⚠️ Nao foi possivel plotar importancias: {e}")


def evaluate(model_name="logreg"):
    """
    Funcao de Exame e Extracao Grafica e Métrica do Modelo Candidato (Report de Aceitação).
    
    - O que ela faz: Roda os Pipelines num conjunto Cego. Tabula e avalia predições do Ponto Ótimo de Corte ('Threshold' Cost-Sensitive). 
      Emite Relatórios de Classification (F1, Sensibilidade). Salva o cruzamento em Matriz e Imagens ROC/PRC em formato PNG pra Dashboarding.
    - Quando é ativada: Sempre subsequente à Compilação/Treino de um modelo em Orquestradores Principais.
    - Benefício/Comunicação: Empilha os JSONLs garantindo métricas para rastreabilidade, alimentando Frontends.
    """
    print(f"\n📊 AVALIANDO MODELO FINAL: {model_name.upper()}")
    
    # --------------------------------------------------------------------------
    # 1. CARGA DE ARTEFATOS
    # --------------------------------------------------------------------------
    try:
        X_test = pd.read_pickle(PROCESSED_DATA_DIR / "X_test.pkl")
        y_test = pd.read_pickle(PROCESSED_DATA_DIR / "y_test.pkl").values.ravel()
        
        model_path = MODELS_DIR / f"{model_name}_best_model.pkl"
        model = joblib.load(model_path)
        print(f"🔹 Modelo carregado de: {model_path}")
    except FileNotFoundError as e:
        print(f"❌ Erro: {e}. Treine o modelo primeiro.")
        return

    # --------------------------------------------------------------------------
    # 2. CARREGAR THRESHOLD OTIMIZADO
    # --------------------------------------------------------------------------
    optimal_threshold = load_threshold(model_name, MODELS_DIR, default=0.5)
    print(f"🎯 Threshold Otimizado Carregado: {optimal_threshold:.4f}")
    
    # --------------------------------------------------------------------------
    # 3. INFERENCIA EM LOTE
    # --------------------------------------------------------------------------
    print("🔮 Realizando inferencia no conjunto de Teste...")
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Predicoes com AMBOS os thresholds
    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    # --------------------------------------------------------------------------
    # 4. METRICAS E RELATORIOS
    # --------------------------------------------------------------------------
    # ROC-AUC (independente do threshold)
    auc_roc = roc_auc_score(y_test, y_proba)
    
    # PR-AUC (Precision-Recall AUC — mais informativa para dados desbalanceados)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print(f"\n{'='*60}")
    print(f"  METRICAS INDEPENDENTES DO THRESHOLD")
    print(f"{'='*60}")
    print(f"  🌟 ROC-AUC Score:  {auc_roc:.4f}")
    print(f"  🌟 PR-AUC Score:   {pr_auc:.4f}")
    print(f"{'='*60}")
    
    # Relatorio com Threshold DEFAULT (0.5)
    print(f"\n📋 Relatorio com Threshold DEFAULT (0.5):")
    print(classification_report(y_test, y_pred_default))
    
    f1_default = f1_score(y_test, y_pred_default)
    prec_default = precision_score(y_test, y_pred_default, zero_division=0)
    rec_default = recall_score(y_test, y_pred_default, zero_division=0)
    
    # Relatorio com Threshold OTIMIZADO
    print(f"\n📋 Relatorio com Threshold OTIMIZADO ({optimal_threshold:.4f}):")
    print(classification_report(y_test, y_pred_optimal))
    
    f1_optimal = f1_score(y_test, y_pred_optimal)
    prec_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    rec_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
    
    # Comparacao lado a lado
    print(f"\n{'='*60}")
    print(f"  COMPARACAO: DEFAULT vs OTIMIZADO")
    print(f"{'='*60}")
    print(f"  {'Metrica':<15} {'Default (0.5)':<18} {'Otimizado':<18} {'Delta':<10}")
    print(f"  {'-'*55}")
    print(f"  {'F1-Score':<15} {f1_default:<18.4f} {f1_optimal:<18.4f} {f1_optimal-f1_default:+.4f}")
    print(f"  {'Precision':<15} {prec_default:<18.4f} {prec_optimal:<18.4f} {prec_optimal-prec_default:+.4f}")
    print(f"  {'Recall':<15} {rec_default:<18.4f} {rec_optimal:<18.4f} {rec_optimal-rec_default:+.4f}")
    print(f"{'='*60}")

    # -------------------------------------------------------------------------
    # 5. PERSISTENCIA DE METRICAS NO HISTORICO
    # -------------------------------------------------------------------------
    metrics_data = {
        "roc_auc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "optimal_threshold": float(optimal_threshold),
        "metrics_default_threshold": {
            "f1": float(f1_default),
            "precision": float(prec_default),
            "recall": float(rec_default)
        },
        "metrics_optimal_threshold": {
            "f1": float(f1_optimal),
            "precision": float(prec_optimal),
            "recall": float(rec_optimal)
        },
        "classification_report": classification_report(y_test, y_pred_optimal, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred_optimal).tolist()
    }
    
    experiments_log_path = REPORTS_DIR / "experiments_log.jsonl"
    
    if experiments_log_path.exists():
        try:
            history = []
            with open(experiments_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
                
            if history:
                # Procure pelo ultimo experimento pertencente a este modelo
                target_exp = None
                for exp in reversed(history):
                    if exp.get("model_type") == model_name:
                        target_exp = exp
                        break
                
                if target_exp:
                    target_exp.update(metrics_data)
                else:
                    history[-1].update(metrics_data) # Fallback
                
                with open(experiments_log_path, "w", encoding="utf-8") as f:
                    for exp in history:
                        f.write(json.dumps(exp) + "\n")
                print(f"📝 Metricas adicionadas ao log: {experiments_log_path}")
            else:
                print("⚠️ Log de experimentos vazio.")
                 
        except Exception as e:
            print(f"⚠️ Erro ao atualizar log: {e}")

    # --------------------------------------------------------------------------
    # 6. VISUALIZACOES
    # --------------------------------------------------------------------------
    
    # FIGURA 1: Matriz de Confusao (com Threshold Otimizado)
    cm = confusion_matrix(y_test, y_pred_optimal)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Legitimo', 'Fraude'],
                yticklabels=['Legitimo', 'Fraude'])
    plt.title(f'Matriz de Confusao - {model_name.upper()}\n(Threshold: {optimal_threshold:.4f})')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png", dpi=150)
    plt.close()
    
    # FIGURA 2: Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc_roc:.4f}", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f"Curva ROC - {model_name.upper()}")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos (Recall)")
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"roc_curve_{model_name}.png", dpi=150)
    plt.close()
    
    # FIGURA 3: Curva Precision-Recall (NOVA — essencial para dados desbalanceados)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.4f}", color='darkcyan', lw=2)
    
    # Marcar o ponto do threshold otimizado
    plt.scatter([rec_optimal], [prec_optimal], color='red', s=100, zorder=5,
                label=f"Threshold = {optimal_threshold:.4f}")
    
    # Linha de baseline (prevalencia da classe positiva)
    baseline = y_test.mean()
    plt.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f"Baseline = {baseline:.4f}")
    
    plt.title(f"Curva Precision-Recall - {model_name.upper()}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"precision_recall_curve_{model_name}.png", dpi=150)
    plt.close()
    
    # FIGURA 4: Feature Importance / Coeficientes
    try:
        preprocessor = model.named_steps['preprocessor']
        feature_names = preprocessor.get_feature_names_out()
        plot_coefficients(model, feature_names, model_name)
    except Exception:
        print(f"ℹ️  Nao foi possivel gerar grafico de features para {model_name}.")

    print(f"\n✅ Avaliacao completa! Graficos salvos em: {FIGURES_DIR}")


if __name__ == "__main__":
    evaluate(model_name="logreg")