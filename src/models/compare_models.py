import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ==============================================================================
# ARQUIVO: compare_models.py
#
# OBJETIVO:
#   Executar um torneio de comparação (benchmark) entre múltiplos algoritmos de Machine Learning.
#   O script treina cada modelo usando validação cruzada (Cross-Validation) e seleciona o melhor
#   com base em métricas chave como ROC-AUC, Recall e F1-Score.
#
# PARTE DO SISTEMA:
#   Módulo de Seleção de Modelos (Model Selection Stage).
#
# RESPONSABILIDADES:
#   - Carregar o dataset de treino processado (X_train.csv, y_train.csv).
#   - Aplicar amostragem estratificada para acelerar a comparação inicial (evitar horas de treino em 1M linhas).
#   - Definir uma lista de competidores (LogReg, Random Forest, XGBoost, etc).
#   - Garantir que o pré-processamento (SMOTE, Scaler) ocorra DENTRO de cada fold da validação cruzada (prevenção de Data Leakage).
#   - Exportar resultados em CSV (persistência) e TXT (relatório executivo).
#   - Gerar gráficos comparativos para facilitar a decisão visual.
#
# COMUNICAÇÃO:
#   - Lê: data/processed/X_train.csv, y_train.csv
#   - Escreve: reports/data/models_comparison_results.csv (Tabela de métricas)
#   - Escreve: reports/model_comparison_report.txt (Relatório textual)
#   - Escreve: reports/figures/model_comparison_metrics.png (Gráfico de barras)
#
# DEPENDÊNCIAS EXTERNAS:
#   - Scikit-Learn (Pipelines, Models)
#   - Imbalanced-Learn (SMOTE, ImbPipeline)
#   - XGBoost / LightGBM (Gradient Boosting otimizado)
# ==============================================================================

# Adiciona raiz ao path para garantir que imports do pacote 'src' funcionem
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Tenta importar configurações centralizadas; define fallback para execução isolada
try:
    from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, FIGURES_DIR, REPORTS_DIR
except ImportError:
    # Caminhos padrão caso o script seja executado fora do contexto do pacote principal
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
    REPORTS_DIR = PROJECT_ROOT / "reports"
    RANDOM_STATE = 42

from src.features.build_features import get_preprocessor, EDAFeatureEngineer

# Configurações Globais e Constantes
# Filtra warning específico do LightGBM quando usado em Pipeline do Scikit-Learn
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names, but LGBMClassifier was fitted with feature names.*")

SAMPLE_SIZE = 50000  # Tamanho da amostra: 50k é estatisticamente suficiente para rankear algoritmos
CV_FOLDS = 5         # Número de folds: 5 garante robustez estatística sem demorar demais
COMPARISON_DATA_DIR = REPORTS_DIR / "data"
COMPARISON_REPORT_FILE = REPORTS_DIR / "model_comparison_report.txt"

# Garante que os diretórios de saída existam
COMPARISON_DATA_DIR.mkdir(parents=True, exist_ok=True)

def compare_algorithms():
    """
    Função principal que orquestra todo o benchmark de modelos.
    """
    print(f"🥊 INICIANDO TORNEIO DE MODELOS (Amostra: {SAMPLE_SIZE} linhas)")
    
    # --------------------------------------------------------------------------
    # 1. CARGA DE DADOS
    # Carrega os dados processados que foram gerados na etapa de Feature Engineering.
    # --------------------------------------------------------------------------
    try:
        X = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
        # Garante que y seja um array 1D (vetor), necessário para o scikit-learn
        y = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()
    except FileNotFoundError:
        print("❌ Erro: Arquivos de treino não encontrados. Rode 'python main.py --step split'.")
        return

    # --------------------------------------------------------------------------
    # 2. AMOSTRAGEM ESTRATIFICADA
    # Reduz o tamanho do dataset para agilizar o benchmark inicial.
    # "Estratificada" significa que mantemos a mesma % de fraudes original na amostra.
    # --------------------------------------------------------------------------
    if len(X) > SAMPLE_SIZE:
        print(f"✂️ Reduzindo dataset para {SAMPLE_SIZE} instâncias (mantendo estratificação)...")
        from sklearn.model_selection import train_test_split
        X_sample, _, y_sample, _ = train_test_split(
            X, y, train_size=SAMPLE_SIZE, stratify=y, random_state=RANDOM_STATE
        )
    else:
        # Se o dataset for pequeno, usamos ele inteiro
        X_sample, y_sample = X, y

    # --------------------------------------------------------------------------
    # 3. DEFINIÇÃO DOS COMPETIDORES
    # Lista de tuplas (Nome, Instância do Modelo).
    # Usamos parâmetros básicos + balanceamento de classes onde possível.
    # --------------------------------------------------------------------------
    models = [
        # Regressão Logística: Baseline linear (simples e interpretável)
        ('LogReg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)),
        
        # Árvore de Decisão: Baseline não-linear (captura regras if-else simples)
        ('DecisionTree', DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_STATE)),
        
        # Random Forest: Ensemble robusto (reduz variância, bom baseline forte)
        ('RandomForest', RandomForestClassifier(n_estimators=50, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)),
        
        # Gradient Boosting (Sklearn): Boosting padrão (reduz viés)
        ('GradientBoosting', GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE)),

        # Histogram-based Gradient Boosting (Sklearn): Inspirado no LightGBM, muito mais rápido que o padrão
        ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=RANDOM_STATE)),

        # Extra Trees: Similar ao Random Forest, mas com splits mais aleatórios (reduz ainda mais a variância)
        ('ExtraTrees', ExtraTreesClassifier(n_estimators=50, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)),

        # AdaBoost: Boosting clássico, foca nos erros anteriores (bom para combinar com árvores simples)
        ('AdaBoost', AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)),
        
        # XGBoost: Estado da arte em boosting (rápido e performático). scale_pos_weight ajusta o desbalanceamento.
        ('XGBoost', XGBClassifier(eval_metric='logloss', scale_pos_weight=90, n_jobs=-1, random_state=RANDOM_STATE))
    ]

    # Tentativa de importar LightGBM (ótimo para grandes volumes, mas requer instalação extra)
    try:
        from lightgbm import LGBMClassifier
        models.append(('LightGBM', LGBMClassifier(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE, verbose=-1)))
        print("✅ LightGBM incluído no torneio.")
    except ImportError:
        print("⚠️ LightGBM não encontrado. Pulando...")

    # Tentativa de importar CatBoost (Excelente com features categóricas e robusto a overfitting)
    try:
        from catboost import CatBoostClassifier
        # verbose=0 remove o output de treinamento
        models.append(('CatBoost', CatBoostClassifier(verbose=0, auto_class_weights='Balanced', random_state=RANDOM_STATE)))
        print("✅ CatBoost incluído no torneio.")
    except ImportError:
        print("⚠️ CatBoost não encontrado. Pulando (instale 'pip install catboost' para testar)...")

    # --------------------------------------------------------------------------
    # 4. CONFIGURAÇÃO DE MÉTRICAS E PIPELINE
    # Definimos quais métricas queremos acompanhar. ROC_AUC é a principal para classificação desbalanceada.
    # --------------------------------------------------------------------------
    scoring_metrics = {
        'recall': 'recall',       # Capacidade de encontrar TODAS as fraudes
        'precision': 'precision', # Capacidade de não alertar alarmes falsos
        'f1': 'f1',               # Equilíbrio entre Recall e Precision
        'roc_auc': 'roc_auc'      # Capacidade geral de separar classes (independente do threshold)
    }
    
    results_list = []
    
    # Buffer para o relatório textual
    report_buffer = [
        f"RELATÓRIO DE COMPARAÇÃO DE MODELOS",
        f"===================================",
        f"Amostra: {len(X_sample)} linhas",
        f"Folds: {CV_FOLDS}",
        f"Estratégia: Preprocessamento -> SMOTE -> Modelo",
        f"-----------------------------------"
    ]

    print(f"\n🏃 Rodando Cross-Validation ({CV_FOLDS} folds)...")
    
    # Recupera o pipeline de transformacao com Feature Engineering EDA-driven
    eda_engineer = EDAFeatureEngineer()
    X_engineered = eda_engineer.fit_transform(X_sample)
    preprocessor = get_preprocessor(X_engineered)

    for name, model in models:
        print(f"   >> Avaliando: {name}...", end=" ")
        
        # CRITICO: Pipeline com Imbalanced-Learn + Feature Engineering
        # O EDAFeatureEngineer aplica as melhorias do EDA (sentinelas, outliers, flags).
        # O SMOTE (criacao de dados sinteticos) deve ocorrer DENTRO do pipeline.
        # Isso garante que ele so veja os dados de TREINO do fold atual.
        pipeline = ImbPipeline(steps=[
            ('eda_features', eda_engineer),                # 0. Feature Engineering EDA-driven
            ('preprocessor', preprocessor),                # 1. Trata categoricas/numericas
            ('smote', SMOTE(random_state=RANDOM_STATE)),   # 2. Balanceia as classes artificialmente
            ('model', model)                               # 3. Treina o modelo
        ])
        
        # Executa a Validação Cruzada
        cv_results = cross_validate(
            pipeline, X_sample, y_sample, 
            cv=CV_FOLDS, scoring=scoring_metrics, 
            n_jobs=-1, return_train_score=False
        )
        
        # Processamento dos Resultados do Fold
        row = {'Model': name}
        txt_row = f"Model: {name: <15} | "
        
        for metric in scoring_metrics:
            mean_score = cv_results[f'test_{metric}'].mean()
            std_score = cv_results[f'test_{metric}'].std()
            
            row[f'{metric}_mean'] = mean_score
            row[f'{metric}_std'] = std_score
            txt_row += f"{metric.upper()}: {mean_score:.4f} (+/-{std_score:.4f})  "
        
        results_list.append(row)
        report_buffer.append(txt_row)
        print("Feito.")

    # --------------------------------------------------------------------------
    # 5. PERSISTÊNCIA DOS RESULTADOS
    # Salva os artefatos para análise posterior.
    # --------------------------------------------------------------------------
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by='roc_auc_mean', ascending=False) # Ordenação pelo ranking principal
    
    # Salva Tabela CSV (Dados Brutos)
    csv_path = COMPARISON_DATA_DIR / "models_comparison_results.csv"
    results_df.to_csv(csv_path, index=False)
    
    # Salva Relatório TXT (Formatado)
    report_buffer.append("\nRANKING FINAL (Ordenado por ROC_AUC):")
    report_buffer.append(results_df.to_string(index=False))
    
    with open(COMPARISON_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report_buffer))
        
    print(f"\n💾 Resultados salvos em:")
    print(f"   - CSV: {csv_path}")
    print(f"   - TXT: {COMPARISON_REPORT_FILE}")

    # --------------------------------------------------------------------------
    # 6. VISUALIZAÇÃO (GRÁFICOS)
    # Cria gráfico de barras comparativo.
    # --------------------------------------------------------------------------
    # Transformação "melt" para formato longo, necessário para o Seaborn plotar barras agrupadas
    metrics_to_plot = ['roc_auc_mean', 'recall_mean', 'precision_mean', 'f1_mean']
    
    plt.figure(figsize=(14, 8))
    melted = results_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    sns.barplot(data=melted, x="Model", y="Score", hue="Metric", palette="viridis")
    plt.title("Comparação de Métricas por Modelo")
    plt.ylim(0, 1.05) # Eixo Y fixo entre 0 e 1 (já que são porcentagens)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    img_path = FIGURES_DIR / "model_comparison_metrics.png"
    plt.savefig(img_path)
    print(f"   - Imagem: {img_path}")

    # --------------------------------------------------------------------------
    # 7.RECOMENDAÇÃO AUTOMÁTICA
    # Identifica o vencedor baseado puramente na métrica alvo (ROC-AUC).
    # --------------------------------------------------------------------------
    winner = results_df.iloc[0]
    print(f"\n🏆 VENCEDOR GERAL: {winner['Model']} (ROC-AUC: {winner['roc_auc_mean']:.4f})")
    print(f"   Recomendação: Utilize o {winner['Model']} para a etapa de otimização de hiperparâmetros.")

if __name__ == "__main__":
    compare_algorithms()