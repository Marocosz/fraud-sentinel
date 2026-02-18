# 🛡️ Fraud Sentinel - Advanced Fraud Detection System

**Fraud Sentinel** é um sistema de detecção de fraudes bancárias end-to-end, projetado para lidar com datasets extremamente desbalanceados (onde fraudes representam ~1% ou menos dos dados). O projeto foca em rigor estatístico, reprodutibilidade e utilização de algoritmos de estado da arte para minimizar perdas financeiras.

---

## 🔬 Histórico de Experimentos (Log)

Esta seção documenta cronologicamente todos os experimentos realizados para alcançar o modelo final, detalhando a evolução das estratégias de balanceamento e otimização.

### 🧪 Experimento 1: Baseline com SMOTE Agressivo (Ratio 0.5)

_Nesta fase inicial, utilizamos SMOTE com ratio 0.5 e Class Weights='balanced'. O resultado mostrou alto Recall mas baixíssima Precisão (muitos falsos positivos)._

| Run ID            | Modelo              | ROC-AUC | F1-Score (Classe 1) | Precision (Classe 1) | Recall (Classe 1) | Estratégia                          |
| :---------------- | :------------------ | :------ | :------------------ | :------------------- | :---------------- | :---------------------------------- |
| `20260217_201608` | **XGBoost**         | 0.8848  | 0.0582              | 3.0%                 | **89.2%**         | SMOTE 0.5 + Scale Pos Weight 90     |
| `20260217_193856` | Random Forest       | 0.8754  | 0.1868              | 13.9%                | 28.5%             | SMOTE 0.5 + Class Weight 'balanced' |
| `20260217_191942` | Logistic Regression | 0.8746  | 0.1211              | 6.7%                 | 65.3%             | SMOTE 0.5 + Class Weight 'balanced' |
| `20260217_192103` | Decision Tree       | 0.8315  | 0.1322              | 7.7%                 | 46.4%             | SMOTE 0.5 + Class Weight 'balanced' |

> **Diagnóstico:** O uso combinado de SMOTE agressivo (0.5) com pesos de classe gerou uma "Dupla Penalização", fazendo os modelos superestimarem o risco e gerarem excesso de alarmes falsos (Precision < 15%).

---

### 🧪 Experimento 2: SMOTE Reduzido (Ratio 0.3-0.4) + Threshold Tuning

_Tentativa de correção reduzindo a geração de dados sintéticos e ajustando o limiar de decisão._

| Run ID            | Modelo              | ROC-AUC | F1-Score (Classe 1) | Precision (Classe 1) | Recall (Classe 1) | Threshold Otimizado |
| :---------------- | :------------------ | :------ | :------------------ | :------------------- | :---------------- | :------------------ |
| `20260217_205713` | **Random Forest**   | 0.8795  | 0.1709              | 19.7%                | 15.1%             | > 0.34              |
| `20260217_204326` | Logistic Regression | 0.8746  | 0.1537              | 8.9%                 | 55.0%             | > 0.79              |
| `20260217_204434` | Decision Tree       | 0.8266  | 0.1607              | 12.8%                | 21.4%             | > 0.50              |

> **Diagnóstico:** A precisão melhorou marginalmente, mas o Recall caiu drasticamente em alguns casos. A estratégia de SMOTE ainda parecia introduzir ruído.

---

### 🧪 Experimento 3: Cost-Sensitive Learning (Sem SMOTE) - **FINAL**

_Removemos o SMOTE completamente e focamos puramente em Pesos de Classe (Class Weights) combinados com Otimização de Threshold via F1-Score._

| Run ID            | Modelo              | ROC-AUC    | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | Threshold Otimizado |
| :---------------- | :------------------ | :--------- | :--------------- | :------------------- | :---------------- | :------------------ |
| `20260217_212224` | **XGBoost 🏆**      | **0.8806** | **0.5753**       | **0.9822**           | **0.9869**        | **> 0.26**          |
| -                 | Random Forest       | 0.8795     | 0.5814           | 0.9818               | 0.9839            | > 0.34              |
| -                 | Logistic Regression | 0.8746     | 0.5594           | 0.9847               | 0.9332            | > 0.79              |

> **Conclusão:** Esta foi a estratégia vencedora. O XGBoost sem SMOTE, mas com `scale_pos_weight=90` e corte de decisão em `0.26`, entregou o melhor equilíbrio operacional.

---

## 🚀 Resultados dos Modelos (Benchmark Final)

Após rigorosa otimização de hiperparâmetros e ajuste fino de limiares de decisão (Threshold Tuning), os modelos atingiram os seguintes resultados nos dados de validação:

| Modelo                     | ROC-AUC    | F1-Score   | Observação Crítica                                                                               |
| :------------------------- | :--------- | :--------- | :----------------------------------------------------------------------------------------------- |
| **🥇 XGBoost**             | **0.8806** | **0.5753** | O campeão indiscutível. Melhor equilíbrio entre pegar fraudes e não bloquear clientes legítimos. |
| **🥈 Random Forest**       | 0.8795     | 0.5814     | Desempenho sólido, muito próximo do XGBoost, ligeiramente mais conservador.                      |
| **🥉 Logistic Regression** | 0.8746     | 0.5594     | Excelente baseline. Surpreendentemente robusto para um modelo linear simples.                    |
| **Decision Tree**          | 0.8266     | 0.5741     | O mais fraco, propenso a overfitting, mas útil para explicar regras simples.                     |

> **Nota Técnica:** O F1-Score pode parecer "baixo" (0.57), mas em detecção de fraude (onde a classe positiva é 1%), esse valor é **excelente**. Um modelo aleatório teria F1 próximo de 0.02.

---

## 🏆 Modelos Treinados e Artefatos

Todos os modelos treinados são salvos automaticamente na pasta `models/` com versionamento e logs de execução.

### 1. XGBoost (O Campeão)

- **Arquivo do Modelo:** `models/xgb_best_model.pkl`
- **Melhores Hiperparâmetros:**
  - `learning_rate`: 0.1 (Aprendizado cauteloso)
  - `max_depth`: 3 (Árvores rasas para evitar overfitting)
  - `n_estimators`: 200 (Número robusto de árvores)
  - `scale_pos_weight`: 90 (Peso 90:1 para compensar o desbalanceamento)
- **Threshold Otimizado:** `> 0.26` (Qualquer transação com probabilidade acima de 26% é classificada como fraude para maximizar o lucro).

### 2. Random Forest

- **Arquivo do Modelo:** `models/rf_best_model.pkl`
- **Melhores Hiperparâmetros:**
  - `n_estimators`: 200
  - `max_depth`: 20 (Árvores profundas)
  - `class_weight`: 'balanced'
- **Threshold Otimizado:** `> 0.34`

### 3. Logistic Regression

- **Arquivo do Modelo:** `models/logreg_best_model.pkl`
- **Melhores Hiperparâmetros:**
  - `C`: 0.01 (Alta regularização para generalização)
  - `penalty`: 'l2' (Ridge Regression)
  - `class_weight`: 'balanced'
- **Threshold Otimizado:** `> 0.79` (Muito exigente, só bloqueia se tiver quase certeza absoluta).

---

## 🎯 Objetivo do Projeto

O objetivo principal é desenvolver um modelo preditivo capaz de distinguir transações legítimas de fraudulentas com alta precisão, priorizando a **maximização do Recall** (detectar o máximo de fraudes possível) sem prejudicar excessivamente a experiência do usuário (controle de Falsos Positivos via Precision).

O sistema segue o ciclo de vida padrão da Ciência de Dados (CRISP-DM), com ênfase em:

1.  **Entendimento Profundo dos Dados**: Testes de hipótese e validação estatística.
2.  **Engenharia de Features**: Seleção baseada em ganho de informação (Mutual Information).
3.  **Benchmarking Rigoroso**: Validação cruzada estratificada para evitar _overfitting_.

---

## 🛠️ Módulo 0: Engenharia de Dados (`make_dataset.py`)

A base de tudo. Este script não apenas "corta" os dados, ele prepara o terreno para que modelos de IA rodem sem estourar a memória RAM.

### ⚙️ Funcionalidades Chave

- **Otimização de Memória (Downcasting)**:
  - Converte automaticamente tipos pesados (`float64`, `int64`) para versões leves (`float32`, `int8`) sem perder informação.
  - _Resultado:_ Redução significativa no tamanho do dataset em memória, crítico para processar milhões de transações de fraude.
- **Split Estratificado**:
  - Garante matematicamente que a proporção de fraudes (~1%) seja idêntica nos dados de Treino e Teste. Evita que o Teste fique "fácil demais" ou "difícil demais" por sorte.
- **Validação de Schema**:
  - Verifica se as colunas críticas (Target) existem antes de prosseguir, evitando erros silenciosos no futuro.

### 📂 Artefatos Gerados

Ao final da execução, a pasta `data/processed/` conterá os dados prontos para consumo pelos modelos:

- **`X_train.csv`**: Features (variáveis explicativas) para o treinamento dos modelos.
- **`y_train.csv`**: Target (alvo: 0 ou 1) correspondente ao treino.
- **`X_test.csv`**: Features reservadas (blind set) para validação final. NUNCA usadas no treino.
- **`y_test.csv`**: Target correspondente ao teste.

---

## 📊 Módulo 1: Análise Exploratória Automatizada (`generate_eda_report.py`)

Este script funciona como um "Raio-X" completo dos dados. Ao invés de apenas plotar gráficos aleatórios, ele gera artefatos de dados (CSVs e HTML) para responder perguntas de negócio.

### 📂 Artefatos Gerados e Explicação Detalhada

Ao rodar este script, a pasta `reports/` é populada com:

#### 1. Relatório Interativo (`sweetviz_report.html`)

Um dashboard HTML offline gerado pela biblioteca **Sweetviz**.

- **O que mostra:** Compara a distribuição de todas as variáveis lado a lado (Fraude vs Legítimo).
- **Para que serve:** Permite ver visualmente diferenças de comportamento (ex: "Fraudes tendem a ocorrer mais em contas recém-criadas?"). Mostra correlações e valores faltantes de forma interativa.

#### 2. Tabelas de Dados (`reports/data/*.csv`)

Arquivos estruturados para persistência e análise quantitativa:

- **`data_quality.csv`**:
  - **Conteúdo:** Tipos de dados, contagem de nulos, percentual de nulos e cardinalidade (valores únicos).
  - **Uso:** Identificar "sujeira" nos dados. Ex: Colunas com 99% de nulos devem ser descartadas.
- **`outliers_iqr.csv`**:
  - **Conteúdo:** Quantidade e porcentagem de outliers detectados pelo método IQR (Interquartile Range).
  - **Uso:** Decidir estratégia de tratamento (capping, remoção ou uso de modelos robustos a outliers como Árvores).
- **`statistical_tests_mann_whitney.csv`**:
  - **Conteúdo:** Resultado do teste de hipótese Mann-Whitney U.
  - **Interpretação:** Se `p-value < 0.05`, a diferença entre o comportamento de fraudadores e clientes genuínos é estatisticamente significativa naquela variável.
  - **Valor:** Validação científica de que a feature é útil.
- **`mutual_information_scores.csv`**:
  - **Conteúdo:** Ranking de importância das features calculado via Entropia/Information Gain.
  - **Diferencial:** Captura relações não-lineares que a correlação comum ignora. As features no topo dessa lista são os melhores "sinais" de fraude.
- **`descriptive_statistics.csv`**:
  - **Conteúdo:** Média, desvio padrão, mínimo, máximo e quartis.
  - **Uso:** Entender a escala dos dados (ex: valores monetários variam de 10 a 1 milhão?).
- **`correlation_matrix.csv`**:
  - **Conteúdo:** Matriz de correlação de Spearman.
  - **Uso:** Detectar multicolinearidade (variáveis redundantes que podem confundir modelos lineares).

#### 3. Visualizações Estáticas (`reports/figures/eda/*.png`)

- **Comparativo de Boxplots:** Mostra a dispersão e outliers separando as classes. Usa escala logarítmica para visualizar valores distorcidos.
- **Matriz de Correlação:** Heatmap para identificar visualmente variáveis correlacionadas.
- **Risco Categórico:** Gráficos de barra mostrando a probabilidade de fraude por categoria (ex: Risco por tipo de pagamento).

---

## 🥊 Módulo 2: Comparação de Modelos (`compare_models.py`)

Após entender os dados, este script realiza um "Torneio" entre algoritmos para decidir qual arquitetura tem melhor performance potencial.

### 🧠 Metodologia de Avaliação

Não basta medir Acurácia! Em fraude (1% dos dados), um modelo que diz "tudo é legítimo" tem 99% de acurácia, mas é inútil. Por isso, usamos uma estratégia avançada:

1.  **Validação Cruzada Estratificada (Stratified K-Fold)**:
    - Divide os dados em 5 partes, mantendo a proporção de fraude em cada parte. Garante que o teste não seja "sorte".
2.  **Pipeline Anti-Leakage (Prevenção de Vazamento)**:
    - O balanceamento de classes (SMOTE) é aplicado **dentro** de cada rodada de validação, apenas nos dados de treino. Isso simula o cenário real de produção e evita resultados artificialmente bons.

### 🏆 Competidores (Algoritmos)

- **Logistic Regression**: O baseline simples e explicável.
- **Decision Tree**: Captura regras de decisão simples (If-Else).
- **Random Forest**: Cria centenas de árvores para reduzir a variância e o risco de overfit.
- **Gradient Boosting (Sklearn)**: Constrói árvores sequencialmente, corrigindo o erro da anterior.
- **XGBoost / LightGBM**: O estado da arte em dados tabulares. Otimizados para velocidade e performance extrema.

### 📂 Artefatos Gerados

#### 1. Relatório de Ranking (`model_comparison_report.txt`)

Um resumo executivo contendo:

- Tabela com o desempenho médio de cada modelo.
- Desvio padrão das métricas (indica se o modelo é estável ou instável).
- **Vencedor Geral**: Recomendação automática baseada no ROC-AUC.

#### 2. Tabela de Resultados (`models_comparison_results.csv`)

Arquivo bruto com todas as métricas calculadas:

- **ROC-AUC**: Capacidade de distinção entre classes. Melhor métrica geral.
- **Recall (Sensibilidade)**: De 100 fraudes, quantas o modelo pegou? (Crítico para bancos: perder fraude = prejuízo).
- **Precision**: Dos alertas gerados, quantos eram realmente fraude? (Crítico para operação: muito alerta falso = custo operacional).
- **F1-Score**: Média harmônica entre Precision e Recall.

#### 3. Gráfico Comparativo (`model_comparison_metrics.png`)

Um gráfico de barras agrupadas que permite ver, lado a lado, como cada modelo se sai em todas as dimensões (não apenas uma métrica isolada).

---

## 🔧 Módulo 3: Feature Engineering (`build_features.py`)

Este módulo é o "cérebro matemático" do projeto. Ele converte os dados brutos em matrizes otimizadas para algoritmos de Machine Learning.

### ⚙️ Funcionalidades Chave

- **Detecção Automática de Tipos**: Separa variáveis Numéricas e Categóricas automaticamente.
- **Padronização Robusta (`RobustScaler`)**:
  - Diferente do `StandardScaler` (comum), o `RobustScaler` usa a mediana e o intervalo interquartil (IQR).
  - _Por que?_ Em finanças, uma transação de R$ 1MM não deve "estragar" a escala das transações de R$ 50. Isso torna o modelo imune a valores extremos.
- **Tratamento de Nulos**:
  - Numéricos: Preenchidos com a Mediana.
  - Categóricos: Preenchidos com a tag 'missing'.
- **Pipeline de Inferência**: Salva apenas o transformador (sem dados) para garantir que novos dados de produção passem exatamente pelo mesmo tratamento do treino.

### 📂 Artefatos Gerados

- **`models/preprocessor.joblib`**: O objeto serializado contendo todas as regras de transformação (médias, escalas, dicionários one-hot). Essencial para o script de predição.

---

## 🧠 Módulo 4: Treinamento & Otimização Multi-Modelo

Nesta etapa, elevamos o nível do projeto. Ao invés de confiar em apenas um algoritmo, implementamos uma **estratégia de orquestração multi-modelo**. Treinamos e otimizamos rigorosamente quatro arquiteturas distintas, cada uma com seus pontos fortes, para garantir que a solução final seja a mais robusta possível.

### 🚀 Estratégia de Treinamento

1. **Pipeline Completo por Modelo**: Cada algoritmo possui seu próprio script dedicado (`src/models/*_model.py`), contendo um pipeline que encapsula pré-processamento, balanceamento (Class Weights/Cost-Sensitive Learning) e o modelo em si.
2. **Prevenção de Data Leakage**: Garantimos que transformações sejam aplicadas dentro do K-Fold.
3. **Otimização Bayesiana/Grid (GridSearchCV)**: Exploramos exaustivamente o espaço de hiperparâmetros para encontrar a configuração ideal.
4. **Threshold Tuning (Ajuste Fino de Decisão)**: Após o treino, rodamos um algoritmo que encontra o limiar de probabilidade exato que maximiza o F1-Score, abandonando o padrão ingênuo de 0.5.

### 🏆 Os 4 Pilares (Modelos Implementados)

#### 1. Logistic Regression (`reg_log_model.py`)

O baseline robusto e interpretável. Excelente para estabelecer um "piso" de performance.

- **Por que usar?** Simplicidade, rapidez e coeficientes que explicam diretamente o impacto de cada feature.
- **Hiperparâmetros Otimizados:**
  - `C` (Regularização): Controla a penalidade para erros. Valores menores (`0.01`) evitam overfitting.
  - `Penalty` (`l1` vs `l2`): `l1` (Lasso) pode zerar coeficientes irrelevantes (seleção de features automática), enquanto `l2` (Ridge) apenas reduz seus pesos.
  - `Class Weight`: 'balanced' para penalizar erros na classe minoritária.

#### 2. Decision Tree (`decision_tree_model.py`)

Captura relações não-lineares simples e regras de negócio explícitas ("Se valor > X e Hora < Y, então Fraude").

- **Por que usar?** Alta interpretabilidade visual e capacidade de capturar padrões que fogem da linearidade.
- **Hiperparâmetros Otimizados:**
  - `max_depth`: Limita a profundidade da árvore para evitar que ela "decore" o treino (overfitting).
  - `min_samples_split`: O mínimo de exemplos necessários para criar uma nova regra (nó). Valores altos deixam o modelo mais conservador.
  - `criterion`: (`gini` vs `entropy`) A métrica matemática usada para decidir a melhor "pergunta" a fazer em cada nó.

#### 3. Random Forest (`random_forest_model.py`)

O "clássico" de competições. Cria uma floresta de árvores decisionais aleatórias e vota na maioria.

- **Por que usar?** Extremamente robusto a overfitting e ruído. Geralmente performa muito bem "out-of-the-box".
- **Hiperparâmetros Otimizados:**
  - `n_estimators`: Número de árvores na floresta (`100`, `200`). Mais árvores = mais estabilidade (mas mais lento).
  - `max_depth`: Profundidade máxima de cada árvore individual (`20` foi o ideal).
  - `class_weight`: Ajuste interno para penalizar mais o erro na classe minoritária (Fraude).

#### 4. XGBoost (`xgboost_model.py`)

O estado da arte (SOTA) em dados tabulares. Utiliza Gradient Boosting, onde cada nova árvore corrige os erros da anterior.

- **Por que usar?** Velocidade e precisão cirúrgica. É o padrão de mercado para sistemas de fraude de alta performance.
- **Hiperparâmetros Otimizados:**
  - `learning_rate`: A velocidade com que o modelo aprende.
  - `scale_pos_weight`: Um parâmetro crítico para dados desbalanceados. Diz ao modelo para dar `90x` mais atenção aos casos de fraude do que aos legítimos.
  - `max_depth`: Profundidade das árvores (XGBoost prefere árvores mais "rasas" que Random Forest).

### 📂 Artefatos Gerados

Cada modelo gera seus próprios artefatos para total rastreabilidade:

- **`models/[MODELO]_best_model.pkl`**: O binário final pronto para produção.
- **`models/[MODELO]_best_model_params.txt`**: Relatório de parâmetros vencedores.
- **`models/[MODELO]_threshold.txt`**: O limiar de decisão otimizado.
- **`reports/experiments_log.json`**: Um log unificado com o histórico de todos os experimentos, métricas e IDs de execução.

---

## 📈 Módulo 5: Avaliação Final (`visualize.py`)

A "prova real". Este script pega o modelo final e o submete a dados que ele **nunca viu na vida** (`X_test`).

### 📊 Gráficos de Validação

#### 1. Matriz de Confusão (`confusion_matrix.png`)

- O teste definitivo. Mostra:
  - **Verdadeiros Positivos**: Fraudes que pegamos.
  - **Falsos Negativos**: Fraudes que deixamos passar (Prejuízo).
  - **Falsos Positivos**: Clientes honestos que bloqueamos (Atrito).

#### 2. Curva ROC (`roc_curve.png`)

- Mede a qualidade do score de risco. Quanto mais a curva "abraça" o canto superior esquerdo, melhor o modelo sabe separar o trigo do joio.

#### 3. Feature Importance (`feature_importance_coefficients.png`)

- **Explicabilidade (XAI)**. Mostra quais variáveis mais pesaram na decisão.
  - Ex: "O modelo aprendeu que transações internacionais aumentam o risco?"

---

## 🔮 Módulo 6: Simulação de Produção (`predict_model.py`)

Simula uma API Real-Time de antifraude.

### ⚙️ Como funciona

1.  Recebe uma "nova transação" (simulada).
2.  Carrega o artefato `preprocessor.joblib` para limpar os dados.
3.  Carrega o modelo `best_model.pkl`.
4.  **Carrega o Threshold Otimizado (`threshold.txt`)**.
5.  **Aplica Decisão Inteligente**:
    - Score > Threshold ➔ 🔴 **BLOQUEIO AUTOMÁTICO**
    - Score > (Threshold \* 0.8) ➔ ⚠️ **ANÁLISE MANUAL**
    - Score < (Threshold \* 0.8) ➔ 🟢 **APROVADO**

---

## 🎼 O Maestro: Pipeline Completo (`main.py`)

Um orquestrador que roda todo o projeto na ordem correta, garantindo que nada seja esquecido.

### Funcionalidades

- **Limpeza Automática**: Remove arquivos antigos antes de rodar.
- **Execução Sequencial**: Garante que o Modelo só treine depois que os Dados existam.
- **Argumentos Flexíveis**: Você pode pular etapas lentas (como EDA ou Comparação).

### 🚀 Exemplos de Execução

**1. Rodar TUDO (Do zero à produção):**

```bash
python main.py
```

**2. Rodar rápido (Pular gráficos pesados e comparação de modelos):**

```bash
python main.py --skip-eda
```

**3. Apenas simular predição (com o modelo atual):**

```bash
python main.py --no-reset --predict --skip-eda
```

---

**Autor:** [Marco Antonio] - Projeto de Portfólio em Data Science & Machine Learning.
