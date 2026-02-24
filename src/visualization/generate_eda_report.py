import sys
import io
import pandas as pd
import numpy as np
# Fix de compatibilidade: Sweetviz ainda usa VisibleDeprecationWarning removido no Numpy 2.0+
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = UserWarning

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import mutual_info_classif

# ==============================================================================
# ARQUIVO: generate_eda_report.py
#
# OBJETIVO:
#   Automatizar a geração de relatórios de Análise Exploratória de Dados (EDA).
#   O script carrega os dados brutos, calcula estatísticas descritivas,
#   gera visualizações (distribuições, correlações, risco por categoria) e
#   exporta um resumo textual, tabelas CSV e report HTML para análise.
#
# PARTE DO SISTEMA:
#   Módulo de Visualização e Análise de Dados (Preprocessing Stage).
#
# RESPONSABILIDADES:
#   - Carregar e validar o dataset inicial.
#   - Identificar automaticamente colunas numéricas, categóricas e o target.
#   - Gerar métricas de qualidade de dados (nulos, tipos, cardinalidade, duplicatas).
#   - Realizar testes estatísticos (Mann-Whitney) e quantificação de outliers (IQR).
#   - Produzir artefatos visuais (gráficos) salvos em 'reports/figures/eda'.
#   - Produzir artefatos de dados (CSVs estatísticos) salvos em 'reports/data'.
#   - Produzir artefato textual (relatório) salvo em 'reports/eda_summary.txt'.
#   - Gerar dashboard interativo HTML (Sweetviz) salvo em 'reports/sweetviz_report.html'.
#
# COMUNICAÇÃO:
#   - Lê: data/raw/Base.csv (padrão ou configurado no config.py)
#   - Escreve: reports/figures/eda/* (PNGs das análises)
#   - Escreve: reports/data/*.csv (Tabelas de métricas para persistência)
#   - Escreve: reports/eda_summary.txt (Relatório consolidado)
#   - Escreve: reports/sweetviz_report.html (Dashboard interativo)
# ==============================================================================

# Adiciona raiz ao path para garantir que imports do pacote 'src' funcionem
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Tenta importar configurações centralizadas; define fallback para execução isolada
try:
    from src.config import RAW_DATA_PATH, FIGURES_DIR, REPORTS_DIR
except ImportError:
    # Caminhos padrão caso o script seja executado fora do contexto do pacote principal
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Base.csv"
    FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
    REPORTS_DIR = PROJECT_ROOT / "reports"

# Configurações Globais de Saída
EDA_OUTPUT_DIR = FIGURES_DIR / "eda"
EDA_DATA_DIR = REPORTS_DIR / "data"  # Diretório para persistência de CSVs
EDA_REPORT_FILE = REPORTS_DIR / "eda_summary.txt"

# Configurações Estéticas de Plotagem (Seaborn/Matplotlib)
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams["figure.figsize"] = (12, 6)

class EDAReporter:
    """
    Classe responsável por orquestrar toda a análise exploratória.
    Encapsula o estado dos dados, configurações de diretório e lógica de geração de relatórios.
    """
    
    def __init__(self, data_path):
        """
        Inicializa o reporter com o caminho dos dados.
        
        - O que recebe:
          data_path (str/Path): Caminho para o arquivo CSV de dados brutos.
          Ex: 'data/raw/Base.csv'
        - O que retorna: Instância de Si Mesmo.
        - Quando é chamada: Imediatamente antes de invocar o `.run()` no pipeline principal.
        """
        self.data_path = Path(data_path)
        self.df = None
        self.target_col = None
        self.num_cols = []
        self.cat_cols = []
        self.report_buffer = io.StringIO() # Buffer em memória para construir o relatório texto incrementalmente

    def _log(self, title, content):
        """
        Método auxiliar para registrar uma seção no buffer do relatório textual e imprimir feedback no console.
        
        Args:
            title (str): Título da seção (ex: "Estatísticas Descritivas").
            content (str): Corpo do texto ou representação string de um DataFrame.
        """
        self.report_buffer.write(f"\n{'='*80}\n")
        self.report_buffer.write(f" {title.upper()}\n")
        self.report_buffer.write(f"{'='*80}\n")
        self.report_buffer.write(f"{content}\n")
        print(f"✅ [Processado]: {title}")

    def setup_directories(self):
        """
        Garante que os diretórios de saída (imagens, dados, relatórios) existam antes de salvar arquivos.
        Utiliza 'mkdir(parents=True)' para criar caminhos aninhados se necessário.
        """
        EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        EDA_DATA_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """
        Carrega o dataset e realiza a introspecção inicial das colunas.
        
        Lógica:
            1. Carrega o CSV apontado pelo Path.
            2. Identifica automaticamente a coluna alvo (target) buscando por nomes comuns em fraude.
            3. Segrega colunas em listas de Numéricas e Categóricas para processamento diferenciado.
            4. Remove o target da lista de features numéricas para evitar redundância/vazamento nos gráficos.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {self.data_path}")
        
        self.df = pd.read_csv(self.data_path)
        
        # Identificação automática do target (Regra de Negócio: suporta nomenclaturas padrão de datasets de fraude)
        if 'fraud_bool' in self.df.columns:
            self.target_col = 'fraud_bool'
        elif 'is_fraud' in self.df.columns:
            self.target_col = 'is_fraud'
        
        # Separação de Colunas por Tipo
        self.cat_cols = self.df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        self.num_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        
        # Ajuste estratégico: O target não deve ser tratado como feature numérica comum nas análises de input
        if self.target_col and self.target_col in self.num_cols:
            self.num_cols.remove(self.target_col)

        self._log("Carga de Dados", f"Dataset carregado: {self.df.shape}\nTarget: {self.target_col}")

    def generate_structure_report(self):
        """
        Gera métricas de qualidade e estrutura dos dados para identificar problemas iniciais.
        
        Análises realizadas:
            - Tipos de dados e uso de memória (df.info).
            - Contagem e percentual de nulos por coluna.
            - Cardinalidade (valores únicos) para detectar constantes ou IDs.
            - Detecção de linhas duplicadas (integridade do dado).
            
        Persistência:
            - Salva a tabela de qualidade em 'reports/data/data_quality.csv'.
        """
        # Captura o output do df.info() que normalmente vai para o console
        buffer = io.StringIO()
        self.df.info(buf=buffer, verbose=True, show_counts=True)
        info_str = buffer.getvalue()
        
        # Cria DataFrame resumo de qualidade
        quality = pd.DataFrame({
            'Dtype': self.df.dtypes,
            'Nulos': self.df.isnull().sum(),
            '% Nulos': (self.df.isnull().sum() / len(self.df)) * 100,
            'Cardinalidade': self.df.nunique()
        }).sort_values(by='% Nulos', ascending=False)

        # Verificação de Duplicatas
        n_duplicates = self.df.duplicated().sum()
        dup_pct = (n_duplicates / len(self.df)) * 100
        dup_msg = f"Duplicatas: {n_duplicates} ({dup_pct:.2f}%)"

        # PERSISTÊNCIA: Salva em CSV para consumo posterior
        quality.to_csv(EDA_DATA_DIR / "data_quality.csv")

        report_content = f"{info_str}\n\n--- RELATÓRIO DE QUALIDADE ---\n{dup_msg}\n\n{quality.to_string()}"
        self._log("Estrutura e Qualidade", report_content)

    def analyze_categorical_domain(self):
        """
        Analisa o domínio das variáveis categóricas.
        
        Lógica:
            - Para baixa cardinalidade (<= 30): Lista todos os valores únicos (útil para entender categorias como 'status', 'type').
            - Para alta cardinalidade: Lista apenas o Top 5 mais frequentes para evitar poluição visual.
        """
        buffer_str = ""
        for col in self.cat_cols:
            unique_vals = self.df[col].unique()
            if len(unique_vals) <= 30:
                buffer_str += f"\nFeature '{col}' ({len(unique_vals)} categorias):\n   {sorted(unique_vals, key=lambda x: str(x))}\n"
            else:
                top_5 = self.df[col].value_counts().head(5).index.tolist()
                buffer_str += f"\nFeature '{col}': Alta cardinalidade ({len(unique_vals)} únicos). Top 5 mais frequentes: {top_5}...\n"
        
        self._log("Domínio das Variáveis Categóricas", buffer_str)

    def analyze_outliers(self):
        """
        Quantifica outliers usando o método estatístico IQR (Interquartile Range).
        
        Regra:
            - Outlier Inferior < Q1 - 1.5 * IQR
            - Outlier Superior > Q3 + 1.5 * IQR
            
        Persistência:
            - Salva a tabela de contagem de outliers em 'reports/data/outliers_iqr.csv'.
            
        Objetivo:
            - Alertar sobre a necessidade de tratamento (remoção ou capping) antes da modelagem.
        """
        outlier_report = []
        
        for col in self.num_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Limites teóricos comuns
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            n_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            pct_outliers = (n_outliers / len(self.df)) * 100
            
            outlier_report.append({
                'Feature': col,
                'Outliers': n_outliers,
                '% Outliers': pct_outliers,
                'Lower Bound': lower_bound,
                'Upper Bound': upper_bound
            })
            
        if outlier_report:
            outlier_df = pd.DataFrame(outlier_report).sort_values(by='% Outliers', ascending=False)
            
            # PERSISTÊNCIA
            outlier_df.to_csv(EDA_DATA_DIR / "outliers_iqr.csv", index=False)
            
            self._log("Quantificação de Outliers (IQR Method)", outlier_df.to_string())

    def generate_statistics_report(self):
        """
        Calcula estatísticas descritivas básicas (média, desvio padrão, quartis).
        Essencial para entender a escala ("ordem de grandeza") e dispersão das variáveis numéricas.
        
        Persistência:
            - Salva em 'reports/data/descriptive_statistics.csv'.
        """
        desc = self.df.describe().T
        # PERSISTÊNCIA
        desc.to_csv(EDA_DATA_DIR / "descriptive_statistics.csv")
        self._log("Estatísticas Descritivas (Numéricas)", desc.to_string())

    def perform_statistical_tests(self):
        """
        Realiza testes de hipótese (Mann-Whitney U) para verificar significância estatística.
        
        Objetivo:
            - Validar se a distribuição de uma feature é estatisticamente diferente entre 'Fraude' e 'Genuíno'.
            - Se p-value < 0.05, rejeita-se a nulidade: a feature PROVAVELMENTE ajuda a separar fraude.
        
        Por que Mann-Whitney?
            - É um teste não-paramétrico (não assume distribuição Normal), ideal para dados financeiros 
              que costumam ter caudas longas e outliers.

        Persistência:
            - Salva em 'reports/data/statistical_tests_mann_whitney.csv'.
        """
        if not self.target_col: return

        results = []
        # Separa os grupos
        fraud_data = self.df[self.df[self.target_col] == 1]
        legit_data = self.df[self.df[self.target_col] == 0]

        # Amostragem para performance se o dataset for massivo (>100k linhas)
        # O teste é O(n*m), então amostras de 10k já dão significância com performance
        if len(self.df) > 100000:
             fraud_sample = fraud_data.sample(min(len(fraud_data), 10000), random_state=42)
             legit_sample = legit_data.sample(min(len(legit_data), 10000), random_state=42)
        else:
             fraud_sample = fraud_data
             legit_sample = legit_data

        for col in self.num_cols:
            # Mann-Whitney U test (two-sided)
            stat, p_value = mannwhitneyu(fraud_sample[col], legit_sample[col], alternative='two-sided')
            
            significancia = "Significativo (p<0.05)" if p_value < 0.05 else "Não Significativo"
            results.append({
                'Feature': col,
                'Mann-Whitney Stat': stat,
                'P-Value': p_value,
                'Conclusão': significancia
            })
            
        stats_df = pd.DataFrame(results).sort_values(by='P-Value')
        
        # PERSISTÊNCIA
        stats_df.to_csv(EDA_DATA_DIR / "statistical_tests_mann_whitney.csv", index=False)
        
        self._log("Testes Estatísticos (Fraude vs Legit)", stats_df.to_string())

    def calculate_mutual_information(self):
        """
        Calcula o Score de Informação Mútua (Mutual Information) entre features e o target.
        
        Diferença vs Correlação:
            - Correlação mede relação Linear/Monotônica.
            - Mutual Info mede QUALQUER dependência (ex: relação quadrática, senoidal, complexa).
            
        Importância:
            - Features com alta MI são candidatas fortes para o modelo, mesmo se a correlação for baixa.
            
        Persistência:
            - Salva rankings em 'reports/data/mutual_information_scores.csv'.
            - Salva gráfico em 'reports/figures/eda/05_mutual_information.png'.
        """
        if not self.target_col: return

        # Amostragem para performance (MI é muito custoso computacionalmente com KNN interno)
        sample_size = min(50000, len(self.df))
        df_sample = self.df.sample(sample_size, random_state=42)
        
        X = df_sample[self.num_cols].fillna(0) # MI do sklearn não aceita NaNs
        y = df_sample[self.target_col]

        mi_scores = mutual_info_classif(X, y, discrete_features=False, random_state=42)
        mi_df = pd.DataFrame({'Feature': self.num_cols, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Plot
        plt.figure(figsize=(10, 8))
        # Fix FutureWarning: Assign y to hue
        sns.barplot(x=mi_df['MI Score'], y=mi_df['Feature'], hue=mi_df['Feature'], palette='viridis', legend=False)
        plt.title("Mutual Information Score (Top Features)")
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / "05_mutual_information.png")
        plt.close()

        self._log("Mutual Information (Importância de Features)", mi_df.to_string())
        
        # PERSISTÊNCIA
        mi_df.to_csv(EDA_DATA_DIR / "mutual_information_scores.csv", index=False)

    def plot_comparative_boxplots(self):
        """
        Gera Boxplots comparativos (Fraude vs Não Fraude) para variáveis numéricas.
        
        Visualização:
            - Eixo X: Classes (0 e 1).
            - Eixo Y: Valor da Feature (Log Scale).
            
        Objetivo:
            - Visualizar se existe separação visual clara entre as classes.
            - Verificar se fraudes tendem a ter valores maiores/menores ou mais variância.
            - Usa escala simétrica logarítmica (symlog) para lidar com dados financeiros distorcidos.
        """
        if not self.target_col: return

        # Seleciona features limitadas para o grid não ficar gigante (Top 12 da lista original)
        cols_to_plot = self.num_cols[:12] 
        n_cols = 3
        n_rows = (len(cols_to_plot) // n_cols) + 1

        # Preparação para plotagem: Map target to string labels to avoid Matplotlib warnings
        df_plot = self.df.copy()
        df_plot[self.target_col] = df_plot[self.target_col].replace({0: 'Genuíno', 1: 'Fraude'})

        plt.figure(figsize=(18, 5 * n_rows))
        
        for i, col in enumerate(cols_to_plot):
            plt.subplot(n_rows, n_cols, i+1)
            # Fix FutureWarning: Assign x to hue
            sns.boxplot(x=self.target_col, y=col, data=df_plot, hue=self.target_col, palette='Set2', legend=False)
            plt.title(f"{col} por Classe")
            plt.yscale('symlog') 
        
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / "06_comparative_boxplots.png")
        plt.close()

    def plot_temporal_analysis(self):
        """
        Analisa a taxa de fraude ao longo do tempo (apenas se houver coluna 'month').
        
        Objetivo:
            - Detectar sazonalidade (ex: fraude aumenta no natal?) ou tendências (ataque crescendo?).
        """
        if 'month' in self.df.columns and self.target_col:
            fraud_by_month = self.df.groupby('month')[self.target_col].mean()
            
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=fraud_by_month.index, y=fraud_by_month.values, marker='o', color='crimson')
            plt.title("Taxa de Fraude por Mês (Sazonalidade)")
            plt.ylabel("Taxa de Fraude (Média)")
            plt.xlabel("Mês")
            plt.grid(True)
            plt.savefig(EDA_OUTPUT_DIR / "07_temporal_fraud_rate.png")
            plt.close()
            
            self._log("Análise Temporal", f"Taxa de Fraude por Mês:\n{fraud_by_month.to_string()}")

    def plot_target_distribution(self):
        """
        Visualiza o balanceamento das classes de fraude.
        
        Objetivo:
            - Mostrar graficamente e textualmente o quão desbalanceado é o dataset.
            - Essencial para definir métricas de avaliação (evitar Acurácia em datasets 99% vs 1%).
        """
        if not self.target_col: return

        # Preparação para plotagem: Map target to string labels
        df_plot = self.df.copy()
        df_plot[self.target_col] = df_plot[self.target_col].replace({0: 'Genuíno', 1: 'Fraude'})

        plt.figure(figsize=(8, 5))
        ax = sns.countplot(x=self.target_col, data=df_plot, palette='viridis', hue=self.target_col, legend=False)
        plt.title(f"Distribuição do Target: {self.target_col}")
        
        # Calcula proporções
        count = self.df[self.target_col].value_counts()
        pct = self.df[self.target_col].value_counts(normalize=True) * 100
        self._log("Distribuição do Target", pd.DataFrame({'Total': count, '%': pct}).to_string())

        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / "01_target_distribution.png")
        plt.close()

    def plot_correlations(self):
        """
        Gera e salva a matriz de correlação (Spearman) entre variáveis numéricas.
        
        Decisão Técnica:
            - Utiliza 'Spearman' (rank-order) em vez de Pearson.
            - Motivo: Dados de fraude raramente são lineares/normais. Spearman captura relações monotônicas.
            
        Persistência:
            - Salva matriz completa em 'reports/data/correlation_matrix.csv'.
            - Filtra e exibe no log textual as features mais correlacionadas com o target.
        """
        corr = self.df[self.num_cols + [self.target_col]].corr(method='spearman')
        
        # Plot do Heatmap
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool)) # Máscara para limpar a diagonal superior (é espelhada)
        sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
        plt.title("Matriz de Correlação (Spearman)")
        plt.savefig(EDA_OUTPUT_DIR / "02_correlation_matrix.png")
        plt.close()

        # PERSISTÊNCIA
        corr.to_csv(EDA_DATA_DIR / "correlation_matrix.csv")

        # Texto: Identifica e loga features com maior correlação absoluta com fraude
        if self.target_col:
            target_corr = corr[self.target_col].sort_values(ascending=False)
            self._log("Correlações com o Target", target_corr.to_string())

    def plot_all_histograms(self):
        """
        Gera histogramas para TODAS as colunas numéricas em formato de grid.
        
        Objetivo:
            - Visão panorâmica (Big Picture) das distribuições.
            - Rápida identificação visual de caudas longas, bimodais ou dados concentrados.
        """
        n_cols = len(self.num_cols)
        n_rows = (n_cols // 4) + 1
        
        plt.figure(figsize=(20, 4 * n_rows))
        
        for i, col in enumerate(self.num_cols):
            plt.subplot(n_rows, 4, i+1)
            data_to_plot = self.df[col]
            sns.histplot(x=data_to_plot, bins=30, kde=False, color='steelblue', edgecolor='black', linewidth=0.5)
            plt.title(col, fontsize=10)
            plt.xlabel("")
        
        plt.tight_layout()
        plt.savefig(EDA_OUTPUT_DIR / "03_all_numerical_distributions.png")
        plt.close()

    def plot_categorical_risks(self):
        """
        Analisa o Risco Relativo (Taxa de Fraude) por categoria.
        
        Lógica:
            - Agrupa por categoria e calcula a média do target (0 a 1).
            - Média 0.05 significa 5% de fraude naquela categoria.
            - Filtra categorias com cardinalidade > 50 para o gráfico não quebrar/ficar ilegível.
        """
        buffer_cats = ""
        
        for col in self.cat_cols:
            # Regra de Visualização: Ignora alta cardinalidade
            if self.df[col].nunique() > 50: continue

            # Cálculo do Risco
            risk = self.df.groupby(col)[self.target_col].mean().sort_values(ascending=False)
            buffer_cats += f"\n--- Risco por {col} ---\n{risk.to_string()}\n"

            # Plotagem
            plt.figure(figsize=(12, 6))
            sns.barplot(x=risk.index, y=risk.values, palette='magma', hue=risk.index, legend=False)
            plt.title(f"Risco de Fraude por {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(EDA_OUTPUT_DIR / f"04_risk_{col}.png")
            plt.close()
            
        self._log("Análise de Risco Categórico", buffer_cats)

    def generate_interactive_report(self):
        """
        Gera um relatório HTML interativo avançado utilizando a biblioteca Sweetviz.
        
        O que ele faz:
            - Cria um arquivo .html standalone (offline) com dashboard interativo.
            - Compara distribuições do Target (Fraude vs Não Fraude) lado a lado em todas as features.
            - Mostra nulos, valores distintos e estatísticas in-place.
            
        Observação:
            - Requer a biblioteca 'sweetviz' instalada. 
            - O try/except garante que a execução não quebre se o usuário não tiver a lib.
        """
        print("📊 Tentando importar sweetviz...")
        try:
            import sweetviz as sv
            print("✅ Import realizado com sucesso!")
            
            print("📊 Gerando relatório interativo com Sweetviz (pode demorar um pouco)...")
            
            # Se tivermos target, configuramos para ele comparar as distribuições "Target=0 vs Target=1"
            if self.target_col:
                report = sv.analyze([self.df, "Training Data"], target_feat=self.target_col)
            else:
                report = sv.analyze([self.df, "Training Data"])
                
            html_path = REPORTS_DIR / "sweetviz_report.html"
            report.show_html(filepath=str(html_path), open_browser=False)
            print(f"✅ Relatório Interativo HTML salvo em: {html_path}")
            
        except ImportError:
             print("\n⚠️ Sweetviz não encontrado. Instale com 'pip install sweetviz' para habilitar o dashboard HTML.")
        except Exception as e:
             print(f"\n❌ Erro na geração do relatório Sweetviz: {e}")

    def save_report(self):
        """
        Persiste o conteúdo textual acumulado (self.report_buffer) no disco.
        """
        with open(EDA_REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(self.report_buffer.getvalue())
        print(f"\n📄 Relatório de Texto salvo em: {EDA_REPORT_FILE}")
        print(f"🖼️ Gráficos salvos em: {EDA_OUTPUT_DIR}")

    def run(self):
        """
        Orquestrador principal (Pipeline de Execução).
        Define a ordem lógica das análises: Setup -> Carga -> Estrutura -> Estatística -> Visualização.
        """
        print("🚀 Iniciando Análise Exploratória Automatizada (Modo Avançado)...")
        self.setup_directories()
        self.load_data()
        
        # Etapa 1: Entendimento dos Dados (Data Understanding)
        self.generate_structure_report()
        self.analyze_categorical_domain()
        self.generate_statistics_report()
        self.analyze_outliers()
        self.plot_target_distribution()
        
        # Etapa 2: Análises de Correlação e Causalidade
        if self.target_col:
            # Análises Avançadas (Acadêmico/Profissional)
            self.perform_statistical_tests()
            self.calculate_mutual_information()
            self.plot_comparative_boxplots()
            self.plot_temporal_analysis()

        # Etapa 3: Visualizações Gerais
        self.plot_correlations()
        self.plot_all_histograms()
        
        if self.target_col:
            self.plot_categorical_risks()
            
        # Etapa 4: Dashboard Interativo (Output Rico)
        self.generate_interactive_report()
            
        # Finalização
        self.save_report()
        print("🏁 Análise Concluída com Sucesso!")

if __name__ == "__main__":
    # Configura pandas para não truncar colunas/linhas na impressão do console/relatório
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    reporter = EDAReporter(RAW_DATA_PATH)
    reporter.run()