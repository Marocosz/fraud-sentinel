# ==============================================================================
# ARQUIVO: visualize.py
#
# OBJETIVO:
#   Realizar a avaliação final e detalhada da performance do modelo escolhido.
#   Gera visualizações críticas (Curva ROC, Matriz de Confusão, Importância de Features)
#   para validar se o modelo está pronto para produção.
#
# PARTE DO SISTEMA:
#   Módulo de Avaliação e Monitoramento (Model Evaluation Stage).
#
# RESPONSABILIDADES:
#   - Carregar os dados de teste (X_test, y_test) que o modelo NUNCA viu.
#   - Gerar métricas de negócio (Recall, Precision, F1) e técnicas (AUC).
#   - Plotar gráficos interpretáveis para stakeholders não-técnicos.
#   - Salvar todos os artefatos visuais em 'reports/figures'.
#
# COMUNICAÇÃO:
#   - Lê: models/*_best_model.pkl
#   - Lê: data/processed/X_test.csv, y_test.csv
#   - Escreve: reports/figures/*.png
# ==============================================================================

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Adiciona raiz ao path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports de Configurações
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR

def plot_coefficients(model, feature_names):
    """
    Plota os coeficientes da Regressão Logística para interpretação do modelo.
    
    O que isso mostra?
    - Quais variáveis aumentam a chance de fraude (coeficiente positivo, barra à direita).
    - Quais variáveis diminuem a chance de fraude (coeficiente negativo, barra à esquerda).
    
    Args:
        model: Objeto do modelo treinado (Pipeline).
        feature_names (list): Lista com nomes das colunas correspondentes aos coeficientes.
    """
    try:
        # Acessa o classificador dentro do Pipeline (passo final 'model')
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['model']
        else:
            classifier = model

        # Verifica se o modelo tem coeficientes (LogReg, SVM linear)
        if hasattr(classifier, 'coef_'):
            coefs = classifier.coef_[0]
            
            # Organiza os dados para plotagem
            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
            coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
            
            # Filtra apenas as Top 20 features mais influentes para o gráfico ficar legível
            coef_df = coef_df.sort_values(by='Abs_Coef', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            # Cores divergentes: Azul (negativo) vs Vermelho (positivo)
            sns.barplot(data=coef_df, x='Coefficient', y='Feature', palette='vlag')
            plt.title("Top 20 Features que definem Fraude (Pesos do Modelo)")
            plt.axvline(0, color='black', linewidth=0.8) # Linha central do zero
            plt.tight_layout()
            
            save_path = FIGURES_DIR / "feature_importance_coefficients.png"
            plt.savefig(save_path)
            print(f"📊 Gráfico de Coeficientes salvo em: {save_path}")
        else:
            print("⚠️ Modelo não é linear (sem coeficientes), pulando gráfico de importância.")
            
    except Exception as e:
        print(f"⚠️ Não foi possível plotar coeficientes: {e}")

def evaluate(model_name="logreg"):
    """
    Função principal de avaliação.
    
    Fluxo:
    1. Carrega dados de Teste (Blind Set).
    2. Carrega modelo treinado.
    3. Gera predições (Classe e Probabilidade).
    4. Imprime relatório textual (Classification Report).
    5. Gera e salva 3 gráficos essenciais: Confusão, ROC e Features.
    """
    print(f"\n📊 AVALIANDO MODELO FINAL: {model_name.upper()}")
    
    # --------------------------------------------------------------------------
    # 1. CARGA DE ARTEFATOS
    # --------------------------------------------------------------------------
    try:
        X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
        
        # Carrega o modelo treinado (LogReg)
        model_path = MODELS_DIR / f"{model_name}_best_model.pkl"
        model = joblib.load(model_path)
        print(f"🔹 Modelo carregado de: {model_path}")
    except FileNotFoundError:
        print("❌ Erro: Arquivos não encontrados. Treine o modelo primeiro (main.py --step train).")
        return

    # --------------------------------------------------------------------------
    # 2. INFERÊNCIA EM LOTE
    # --------------------------------------------------------------------------
    print("🔮 Realizando inferência no conjunto de Teste...")
    y_pred = model.predict(X_test)         # Decisão Binária (0 ou 1)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilidade de Fraude (0.0 a 1.0)

    # --------------------------------------------------------------------------
    # 3. MÉTRICAS E RELATÓRIOS
    # --------------------------------------------------------------------------
    print("\n--- Relatório de Classificação ---")
    # Mostra Precision, Recall e F1 para ambas as classes
    print(classification_report(y_test, y_pred))
    
    # AUC: Área sob a curva. Quanto mais próximo de 1.0, melhor o modelo separa as classes.
    auc = roc_auc_score(y_test, y_proba)
    print(f"🌟 ROC-AUC Score: {auc:.4f}")

    # -------------------------------------------------------------------------
    # PERSISTÊNCIA DE MÉTRICAS NO HISTÓRICO (experiments_log.json)
    # -------------------------------------------------------------------------
    metrics_data = {
        "roc_auc": float(auc),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    
    experiments_log_path = REPORTS_DIR / "experiments_log.json"
    
    if experiments_log_path.exists():
        try:
            with open(experiments_log_path, "r") as f:
                history = json.load(f)
                
            # Assume que a última entrada do log é a do treino atual
            if history:
                last_experiment = history[-1]
                # Atualiza com as métricas de avaliação
                last_experiment.update(metrics_data)
                
                with open(experiments_log_path, "w") as f:
                    json.dump(history, f, indent=4)
                print(f"📝 Métricas adicionadas ao log de experimentos: {experiments_log_path}")
            else:
                 print("⚠️ Log de experimentos vazio. Não foi possível vincular métricas.")
                 
        except Exception as e:
            print(f"⚠️ Erro ao atualizar log de experimentos: {e}")
            


    # --------------------------------------------------------------------------
    # 4. VISUALIZAÇÕES (Figuras)
    # --------------------------------------------------------------------------
    
    # FIGURA 1: Matriz de Confusão
    # Essencial para ver Falsos Positivos vs Falsos Negativos
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusão - {model_name.upper()}')
    plt.ylabel('Real (0=Legal, 1=Fraude)')
    plt.xlabel('Predito pelo Modelo')
    plt.savefig(FIGURES_DIR / f"confusion_matrix_{model_name}.png")
    
    # FIGURA 2: Curva ROC
    # Mostra o trade-off entre Sensibilidade e Especificidade em vários thresholds
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--') # Linha pontilhada (sorte/aleatório)
    plt.title(f"Curva ROC - {model_name.upper()}")
    plt.xlabel("Taxa de Falsos Positivos (False Alarm Rate)")
    plt.ylabel("Taxa de Verdadeiros Positivos (Recall)")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"roc_curve_{model_name}.png")
    
    # FIGURA 3: Importância das Features (Coeficientes)
    # Tenta recuperar os nomes das colunas após o One-Hot Encoding do pipeline
    try:
        preprocessor = model.named_steps['preprocessor']
        # Pega nomes das colunas numéricas + categóricas transformadas
        feature_names = preprocessor.get_feature_names_out()
        plot_coefficients(model, feature_names)
    except:
        print("⚠️ Não foi possível recuperar nomes das features para plotagem.")

    print(f"\n✅ Avaliação completa! Gráficos salvos em: {FIGURES_DIR}")

if __name__ == "__main__":
    evaluate(model_name="logreg")