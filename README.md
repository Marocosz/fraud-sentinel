# Fraud Sentinel - Sistema Avancado de Deteccao de Fraudes Bancarias

> [!NOTE]
> Link para download da base de dados usada: https://drive.google.com/file/d/1KWKHddAwpZ2HAwsWmL0HWlUXFw8HWf9N/view?usp=sharing

# 1. Visao Geral do Projeto

| Item                     | Descricao                                                                                                                                                                                                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nome**                 | Fraud Sentinel                                                                                                                                                                                                                                                                                        |
| **Objetivo Principal**   | Desenvolver um pipeline completo de Machine Learning para detectar fraudes em aberturas de contas bancarias, priorizando a maximizacao do Recall (capturar o maximo de fraudes) com controle de Precision (minimizar falsos alarmes).                                                                 |
| **Problema que Resolve** | Fraudes bancarias na abertura de contas causam prejuizos financeiros massivos. O sistema automatiza a triagem de solicitacoes, classificando-as como legitimas ou fraudulentas com base em padroes historicos de comportamento e atributos sociodemograficos.                                         |
| **Publico-Alvo**         | Cientistas de dados, engenheiros de ML, analistas de fraude e instituicoes financeiras que necessitam de um motor de decisao anti-fraude baseado em dados. Tambem serve como projeto de portfolio academico em Data Science.                                                                          |
| **Contexto de Uso**      | O sistema opera sobre o dataset Bank Account Fraud (BAF) Suite, publicado no NeurIPS 2022, que simula dados reais de aberturas de conta com rotulagem binaria (0 = legitima, 1 = fraude). A taxa de fraude e extremamente baixa (~1%), exigindo tecnicas especializadas de balanceamento e avaliacao. |
| **Tarefa de Mineracao**  | Classificacao Binaria Supervisionada                                                                                                                                                                                                                                                                  |
| **Metodologia**          | CRISP-DM (Cross-Industry Standard Process for Data Mining)                                                                                                                                                                                                                                            |

---

# 2. Arquitetura Geral

## 2.1 Tipo de Arquitetura

O Fraud Sentinel adota uma **arquitetura modular orientada a pipeline**, organizada em camadas funcionais independentes. Cada camada possui responsabilidade unica e se comunica com as demais exclusivamente atraves de artefatos persistidos em disco (arquivos CSV, PKL, JSON e TXT). Essa abordagem garante reprodutibilidade, rastreabilidade e desacoplamento entre etapas.

## 2.2 Diagrama da Arquitetura

```
+------------------------------------------------------------------+
|                        main.py (MAESTRO)                         |
|            Orquestrador Central / CLI com argparse               |
+------------------------------------------------------------------+
         |              |              |              |
         v              v              v              v
+----------------+ +----------+ +-----------+ +-------------+
| make_dataset   | | EDA      | | compare   | | train_*     |
| (Data Eng.)    | | Reporter | | _models   | | _model.py   |
+----------------+ +----------+ +-----------+ +-------------+
    |                  |              |              |
    v                  v              v              v
+--------+      +-----------+  +-----------+  +------------+
| data/  |      | reports/  |  | reports/  |  | models/    |
|processed|     | figures/  |  | data/     |  | *.pkl      |
| *.csv  |      | *.png     |  | *.csv     |  | *.txt      |
+--------+      +-----------+  +-----------+  +------------+
                                                    |
                                    +---------------+--------+
                                    v                        v
                             +------------+          +-------------+
                             | visualize  |          | predict     |
                             | .py        |          | _model.py   |
                             | (Avaliacao)|          | (Inferencia)|
                             +------------+          +-------------+
```

## 2.3 Fluxo Macro (Requisicao ate Resposta)

| Etapa                               | Entrada                                                                                                                                                                                                                                                                                                                                                                               | Arquivo                                                                                                                                 | Descricao                                                                                                                                                                                                                                                                                                                                                                                         | Saida                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Ingestao de Dados                | `data/raw/Base.csv` -- O dataset bruto do BAF Suite (NeurIPS 2022) e o ponto de partida de todo o sistema. Contem todas as features sociodemograficas e comportamentais das aberturas de conta, com rotulagem binaria de fraude. E necessario como fonte primaria porque todo o pipeline depende de dados historicos rotulados para aprender padroes.                                 | `make_dataset.py`                                                                                                                       | Carrega o CSV bruto, aplica downcasting de tipos numericos (float64 para float32, int64 para int8) para reduzir consumo de RAM, valida a existencia da coluna target (`fraud_bool`), e executa a divisao estratificada 80/20 que garante matematicamente que a proporcao de fraudes (~1%) seja identica nos conjuntos de treino e teste.                                                          | `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` -- Quatro arquivos CSV limpos e otimizados. Sao separados em features (X) e target (y) porque o scikit-learn exige essa separacao. O split estratificado e salvo em disco para que todas as etapas subsequentes trabalhem sobre exatamente os mesmos dados, garantindo reprodutibilidade.                                                                                                                                                 |
| 2. Analise Exploratoria             | `data/raw/Base.csv` -- O dataset bruto original e carregado novamente (nao os processados) porque a EDA precisa analisar os dados no estado natural, sem transformacoes de escala ou encoding, para identificar problemas reais como nulos, outliers e distribuicoes originais.                                                                                                       | `generate_eda_report.py`                                                                                                                | Executa um raio-X completo dos dados: calcula estatisticas descritivas, quantifica outliers (IQR), roda testes de hipotese (Mann-Whitney U) para validar significancia estatistica de cada feature, calcula Mutual Information para ranquear importancia, gera boxplots comparativos, heatmaps de correlacao (Spearman), analises de risco categorico, e um dashboard HTML interativo (Sweetviz). | `reports/data/*.csv` (7 tabelas de metricas), `reports/figures/eda/*.png` (7+ graficos), `reports/eda_summary.txt` (relatorio textual consolidado), `reports/sweetviz_report.html` (dashboard interativo) -- Esses artefatos servem para o cientista de dados tomar decisoes informadas sobre quais features usar, quais tratamentos aplicar, e validar cientificamente que os dados possuem sinal discriminativo para fraude.                                                                                    |
| 3. Benchmark de Modelos (Opcional)  | `data/processed/X_train.csv`, `y_train.csv` -- Os dados de treino processados sao necessarios porque o benchmark precisa avaliar algoritmos sobre dados comparaveis. Uma amostra estratificada de 50k linhas e extraida para viabilizar a execucao em tempo razoavel sem perder representatividade estatistica.                                                                       | `compare_models.py`                                                                                                                     | Executa um torneio entre 8 a 10 algoritmos (LogReg, DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting, ExtraTrees, AdaBoost, XGBoost, e opcionalmente LightGBM e CatBoost) usando validacao cruzada estratificada de 5 folds. O SMOTE e aplicado dentro de cada fold via ImbPipeline para prevenir data leakage. Mede ROC-AUC, Recall, Precision e F1.                           | `reports/data/models_comparison_results.csv` (tabela com medias e desvios de todas as metricas), `reports/model_comparison_report.txt` (relatorio executivo com ranking), `reports/figures/model_comparison_metrics.png` (grafico de barras comparativo) -- Esses artefatos permitem escolher objetivamente qual algoritmo tem melhor potencial antes de investir tempo na otimizacao de hiperparametros.                                                                                                         |
| 4. Treinamento e Otimizacao         | `data/processed/X_train.csv`, `y_train.csv` -- Os dados de treino sao necessarios para o modelo aprender os padroes de fraude. Cada script de modelo os carrega para construir o pipeline completo (preprocessamento + classificador) e otimizar hiperparametros via busca exaustiva.                                                                                                 | `reg_log_model.py`, `decision_tree_model.py`, `random_forest_model.py`, `xgboost_model.py`, `mlp_model.py`, `isolation_forest_model.py` | Cada script cria um pipeline (preprocessor + modelo), executa GridSearchCV com Stratified K-Fold (3 folds) para encontrar os melhores hiperparametros, retreina o modelo vencedor no dataset completo (quando aplicavel), e executa Threshold Tuning que varre a curva Precision-Recall para encontrar o limiar de decisao que maximiza o F1-Score.                                               | `models/{nome}_best_model.pkl` (modelo serializado pronto para producao), `models/model_{nome}_{timestamp}.pkl` (copia versionada para historico), `models/{nome}_threshold.txt` (threshold otimizado), `models/{nome}_best_model_params.txt` (hiperparametros vencedores), `reports/experiments_log.json` (registro do experimento) -- Cada artefato cumpre um papel: o PKL e o modelo reutilizavel, o threshold define o ponto de operacao, e o JSON garante rastreabilidade completa de todos os experimentos. |
| 5. Avaliacao Final                  | `data/processed/X_test.csv`, `y_test.csv` (dados que o modelo nunca viu) e `models/{nome}_best_model.pkl` (modelo treinado) -- O blind test set e essencial porque simula dados reais de producao. Usar dados de treino para avaliar geraria metricas artificialmente infladas (overfitting). O modelo e carregado serializado para simular exatamente o que aconteceria em producao. | `visualize.py`                                                                                                                          | Carrega o modelo treinado e os dados de teste, gera predicoes de classe e probabilidade, calcula metricas finais (ROC-AUC, Precision, Recall, F1), plota a Matriz de Confusao (visualiza falsos positivos e negativos), a Curva ROC (capacidade de discriminacao) e o grafico de importancia de features (explicabilidade). Atualiza o log de experimentos com as metricas reais.                 | `reports/figures/confusion_matrix_{nome}.png`, `reports/figures/roc_curve_{nome}.png`, `reports/figures/feature_importance_coefficients.png` (se modelo linear) -- Graficos essenciais para validar se o modelo esta pronto para producao e comunicar resultados para stakeholders. O `experiments_log.json` e atualizado com metricas reais de teste, fechando o ciclo de rastreabilidade.                                                                                                                       |
| 6. Simulacao de Producao (Opcional) | `models/{nome}_best_model.pkl`, `models/{nome}_threshold.txt`, `data/processed/X_test.csv`, `y_test.csv` -- O modelo, o threshold e os dados de teste sao combinados para simular o comportamento real de um motor anti-fraude. O gabarito (y_test) e usado apenas para mostrar se o modelo acertou, mas nao influencia a decisao.                                                    | `predict_model.py`                                                                                                                      | Simula o recebimento de transacoes uma a uma. Para cada transacao: calcula o score de risco (probabilidade de fraude), aplica o motor de decisao trinivel (BLOQUEIO se score > threshold, REVISAO MANUAL se score > threshold\*0.8, APROVADO caso contrario), e exibe o resultado detalhado no console com comparacao ao gabarito. Amostra balanceada (50% fraude) para fins de demonstracao.     | Saida no console com decisao por transacao (BLOQUEIO/REVISAO/APROVADO), score de risco, gabarito real e indicacao de acerto/erro -- Nao gera artefatos em disco. A saida serve como demonstracao funcional do sistema operando em regime de inferencia, util para validacao qualitativa e apresentacoes.                                                                                                                                                                                                          |

## 2.4 Separacao de Camadas

| Camada                 | Diretorio            | Responsabilidade                                                                  |
| ---------------------- | -------------------- | --------------------------------------------------------------------------------- |
| Configuracao           | `src/config.py`      | Caminhos, constantes globais, seed aleatoria                                      |
| Engenharia de Dados    | `src/data/`          | Carga, otimizacao de memoria, split estratificado                                 |
| Engenharia de Features | `src/features/`      | Preprocessamento (Scaler, Imputer, OneHot), SMOTE                                 |
| Modelos                | `src/models/`        | Treinamento, otimizacao de hiperparametros, threshold tuning, predicao, benchmark |
| Visualizacao           | `src/visualization/` | EDA automatizada, avaliacao final, graficos                                       |
| Orquestracao           | `main.py`            | Pipeline end-to-end via CLI                                                       |
| Testes                 | `tests/`             | Stubs de testes unitarios (nao implementados)                                     |

---

# 3. Estrutura de Diretorios

```
fraud-sentinel/
|-- main.py                    # Orquestrador principal (CLI)
|-- requirements.txt           # Dependencias do projeto
|-- ideia_inicial.md           # Documento de concepcao e historico de experimentos
|-- .gitignore                 # Regras de exclusao do Git
|-- .env                       # Variaveis de ambiente (vazio)
|
|-- src/                       # Codigo-fonte principal
|   |-- __init__.py            # Torna src um pacote Python
|   |-- config.py              # Configuracoes globais centralizadas
|   |
|   |-- data/
|   |   |-- __init__.py
|   |   |-- make_dataset.py    # Carga, otimizacao e split de dados
|   |
|   |-- features/
|   |   |-- __init__.py
|   |   |-- build_features.py  # Pipeline de preprocessing (Scaler, Imputer, OneHot)
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- reg_log_model.py       # Treinamento Logistic Regression
|   |   |-- decision_tree_model.py # Treinamento Decision Tree
|   |   |-- random_forest_model.py # Treinamento Random Forest
|   |   |-- xgboost_model.py       # Treinamento XGBoost
|   |   |-- mlp_model.py           # Treinamento MLP (Rede Neural)
|   |   |-- isolation_forest_model.py # Treinamento Isolation Forest
|   |   |-- compare_models.py      # Benchmark comparativo de algoritmos
|   |   |-- predict_model.py       # Simulacao de inferencia em producao
|   |   |-- force_precision.py     # Ajuste fino de threshold por Precision-alvo
|   |
|   |-- visualization/
|       |-- __init__.py
|       |-- generate_eda_report.py # EDA automatizada completa
|       |-- visualize.py          # Avaliacao final com graficos
|
|-- data/
|   |-- raw/                   # Dataset bruto (Base.csv) -- nao versionado
|   |-- processed/             # Artefatos processados (X_train, X_test, etc.)
|   |-- external/              # Dados externos (reservado)
|
|-- models/                    # Modelos serializados (.pkl) e parametros (.txt)
|
|-- reports/
|   |-- data/                  # CSVs de metricas (qualidade, correlacao, MI, etc.)
|   |-- figures/               # Graficos PNG (EDA, avaliacao, comparacao)
|   |-- eda_summary.txt        # Relatorio textual consolidado da EDA
|   |-- model_comparison_report.txt # Relatorio do benchmark
|   |-- experiments_log.json   # Historico unificado de todos os experimentos
|   |-- sweetviz_report.html   # Dashboard interativo HTML
|
|-- tests/                     # Stubs de testes (nao implementados)
|
|-- venvmine/                  # Ambiente virtual Python (nao versionado)
```

## 3.1 Descricao Detalhada de Cada Arquivo

### main.py -- Orquestrador Principal

| Atributo         | Descricao                                                                               |
| ---------------- | --------------------------------------------------------------------------------------- |
| **Funcao**       | Orquestra todo o pipeline na ordem correta via CLI (argparse)                           |
| **Funcoes**      | `reset_project_artifacts()`, `main()`                                                   |
| **Entradas**     | Argumentos CLI: `--no-reset`, `--skip-eda`, `--compare-models`, `--predict`, `--models` |
| **Saidas**       | Execucao sequencial de todos os modulos                                                 |
| **Dependencias** | Todos os modulos em `src/`                                                              |

Flags disponiveis:

| Flag               | Efeito                                                |
| ------------------ | ----------------------------------------------------- |
| `--no-reset`       | Pula a limpeza de artefatos antigos                   |
| `--skip-eda`       | Pula a analise exploratoria                           |
| `--compare-models` | Executa o benchmark de algoritmos                     |
| `--predict`        | Roda simulacao de inferencia ao final                 |
| `--models`         | Seleciona modelos especificos (ex: `--models xgb,rf`) |

Identificadores de modelos: `logreg`, `dt`, `rf`, `xgb`, `mlp`, `if`.

### src/config.py -- Configuracoes Globais

Define todas as constantes compartilhadas pelo sistema:

| Constante            | Valor               | Finalidade                     |
| -------------------- | ------------------- | ------------------------------ |
| `PROJECT_ROOT`       | Raiz do projeto     | Base para todos os caminhos    |
| `RAW_DATA_PATH`      | `data/raw/Base.csv` | Caminho do dataset bruto       |
| `PROCESSED_DATA_DIR` | `data/processed/`   | Saida do make_dataset          |
| `MODELS_DIR`         | `models/`           | Armazenamento de modelos       |
| `FIGURES_DIR`        | `reports/figures/`  | Saida de graficos              |
| `RANDOM_STATE`       | 42                  | Semente para reprodutibilidade |
| `TEST_SIZE`          | 0.2                 | Proporcao do test split (20%)  |
| `TARGET_COL`         | `fraud_bool`        | Nome da coluna alvo            |

Cria automaticamente diretorios inexistentes na importacao.

### src/data/make_dataset.py -- Engenharia de Dados

| Atributo     | Descricao                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------- |
| **Funcoes**  | `optimize_memory_usage(df)`, `load_and_split_data()`                                        |
| **Entrada**  | `data/raw/Base.csv`                                                                         |
| **Saida**    | `data/processed/{X_train, X_test, y_train, y_test}.csv`                                     |
| **Excecoes** | `FileNotFoundError` se Base.csv nao existir; `ValueError` se coluna alvo nao for encontrada |

Fluxo interno:

1. Carrega CSV bruto com `pd.read_csv`
2. Valida existencia da coluna target (fallback para `is_fraud`)
3. Aplica downcasting de tipos (`float64`->`float32`, `int64`->`int8`) para otimizar RAM
4. Separa features (X) e target (y)
5. Executa `train_test_split` com `stratify=y` para manter proporcao de fraude
6. Salva 4 CSVs processados

### src/features/build_features.py -- Pipeline de Features

| Atributo    | Descricao                                                                   |
| ----------- | --------------------------------------------------------------------------- |
| **Funcoes** | `get_preprocessor(X)`, `process_features()`                                 |
| **Entrada** | DataFrame X com features brutas                                             |
| **Saida**   | `ColumnTransformer` configurado; opcionalmente `models/preprocessor.joblib` |

Pipeline numerico: `SimpleImputer(median)` -> `RobustScaler()`
Pipeline categorico: `SimpleImputer(constant='missing')` -> `OneHotEncoder(handle_unknown='ignore')`

Decisao tecnica: uso de `RobustScaler` em vez de `StandardScaler` porque dados financeiros possuem outliers extremos que distorceriam a media e desvio padrao.

### src/models/reg_log_model.py -- Logistic Regression

| Atributo                           | Descricao                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Funcao principal**               | `train_logistic_regression()`                                                                    |
| **Estrategia de desbalanceamento** | `class_weight='balanced'` (sem SMOTE)                                                            |
| **Grid Search**                    | `C`: [0.01, 0.1, 1, 10]; `penalty`: ['l1', 'l2']                                                 |
| **Otimizacao**                     | Amostra estratificada de 100k linhas para GridSearch; retreino final com dataset completo        |
| **Saidas**                         | `logreg_best_model.pkl`, `logreg_threshold.txt`, `best_model_params.txt`, `experiments_log.json` |

### src/models/decision_tree_model.py -- Decision Tree

| Atributo             | Descricao                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_decision_tree()`                                                                   |
| **Grid Search**      | `max_depth`: [5, 10, None]; `min_samples_split`: [2, 5]; `criterion`: ['gini', 'entropy'] |
| **Saidas**           | `dt_best_model.pkl`, `dt_threshold.txt`, `dt_best_model_params.txt`                       |

### src/models/random_forest_model.py -- Random Forest

| Atributo             | Descricao                                                                            |
| -------------------- | ------------------------------------------------------------------------------------ |
| **Funcao principal** | `train_random_forest()`                                                              |
| **Grid Search**      | `n_estimators`: [100, 200]; `max_depth`: [10, 20, None]; `min_samples_split`: [2, 5] |
| **Nota**             | RF usa `n_jobs=-1` internamente; GridSearch usa `n_jobs=1` para evitar conflito      |
| **Saidas**           | `rf_best_model.pkl`, `rf_threshold.txt`, `rf_best_model_params.txt`                  |

### src/models/xgboost_model.py -- XGBoost

| Atributo             | Descricao                                                                     |
| -------------------- | ----------------------------------------------------------------------------- |
| **Funcao principal** | `train_xgboost()`                                                             |
| **Estrategia**       | `scale_pos_weight=90` para compensar desbalanceamento                         |
| **Grid Search**      | `learning_rate`: [0.01, 0.1]; `n_estimators`: [100, 200]; `max_depth`: [3, 6] |
| **Otimizacao**       | Amostra estratificada de 100k para GridSearch; retreino no dataset completo   |
| **Saidas**           | `xgb_best_model.pkl`, `xgb_threshold.txt`, `xgb_best_model_params.txt`        |

### src/models/mlp_model.py -- MLP Neural Network

| Atributo             | Descricao                                                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_mlp()`                                                                                                                                      |
| **Grid Search**      | `hidden_layer_sizes`: [(50,), (100,), (50,25)]; `activation`: ['relu','tanh']; `alpha`: [0.0001, 0.001, 0.01]; `learning_rate_init`: [0.001, 0.01] |
| **Nota**             | Usa `early_stopping=True` com 10% de validacao interna                                                                                             |
| **Saidas**           | `mlp_best_model.pkl`, `mlp_threshold.txt`                                                                                                          |

### src/models/isolation_forest_model.py -- Isolation Forest

| Atributo             | Descricao                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_isolation_forest()`                                                                                      |
| **Classe auxiliar**  | `IForestWrapper(BaseEstimator, ClassifierMixin)` -- Wrapper que converte `decision_function` em `predict_proba` |
| **Nota**             | Algoritmo nao-supervisionado adaptado para pipeline supervisionado. Sem GridSearch.                             |
| **Parametros fixos** | `n_estimators=200`, `contamination=0.01`                                                                        |
| **Saidas**           | `if_best_model.pkl`, `if_threshold.txt`                                                                         |

A classe `IForestWrapper` inverte o score de anomalia (`-decision_function`), normaliza com `MinMaxScaler` para [0,1] e empacota no formato `predict_proba` padrao `(n_samples, 2)`.

### src/models/compare_models.py -- Benchmark de Algoritmos

| Atributo             | Descricao                                                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Funcao principal** | `compare_algorithms()`                                                                                                                              |
| **Competidores**     | LogReg, DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting, ExtraTrees, AdaBoost, XGBoost, LightGBM (opcional), CatBoost (opcional) |
| **Metodologia**      | Stratified 5-Fold CV com pipeline SMOTE _dentro_ de cada fold                                                                                       |
| **Metricas**         | ROC-AUC, Recall, Precision, F1-Score                                                                                                                |
| **Saidas**           | `models_comparison_results.csv`, `model_comparison_report.txt`, `model_comparison_metrics.png`                                                      |

### src/models/predict_model.py -- Simulacao de Producao

| Atributo                     | Descricao                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| **Funcoes**                  | `load_inference_artifacts()`, `load_threshold()`, `explain_prediction()`, `predict_sample()` |
| **Motor de decisao**         | Score > Threshold -> BLOQUEIO; Score > Threshold\*0.8 -> REVISAO MANUAL; Abaixo -> APROVADO  |
| **Estrategia de amostragem** | Forca 50% de fraude na demonstracao para visualizacao                                        |

### src/models/force_precision.py -- Ajuste de Precision-Alvo

| Atributo             | Descricao                                                               |
| -------------------- | ----------------------------------------------------------------------- |
| **Funcao principal** | `enforce_precision_target(target_precision, model_filename)`            |
| **Objetivo**         | Encontrar o menor threshold que garanta Precision >= alvo (padrao: 20%) |
| **Metodologia**      | Varredura da curva Precision-Recall no conjunto de teste                |
| **Saidas**           | Sobrescreve `*_threshold.txt`, gera `precision_optimization_curve.png`  |

### src/visualization/generate_eda_report.py -- EDA Automatizada

| Atributo              | Descricao                                                                                                                                                                                                                                                                                                                                                                                    |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Classe**            | `EDAReporter`                                                                                                                                                                                                                                                                                                                                                                                |
| **Metodos**           | `load_data`, `generate_structure_report`, `analyze_categorical_domain`, `analyze_outliers`, `generate_statistics_report`, `perform_statistical_tests`, `calculate_mutual_information`, `plot_comparative_boxplots`, `plot_temporal_analysis`, `plot_target_distribution`, `plot_correlations`, `plot_all_histograms`, `plot_categorical_risks`, `generate_interactive_report`, `save_report` |
| **Artefatos gerados** | `data_quality.csv`, `outliers_iqr.csv`, `statistical_tests_mann_whitney.csv`, `mutual_information_scores.csv`, `descriptive_statistics.csv`, `correlation_matrix.csv`, `sweetviz_report.html`, `eda_summary.txt`, 7+ graficos PNG                                                                                                                                                            |

### src/visualization/visualize.py -- Avaliacao Final

| Atributo         | Descricao                                                               |
| ---------------- | ----------------------------------------------------------------------- |
| **Funcoes**      | `plot_coefficients(model, feature_names)`, `evaluate(model_name)`       |
| **Graficos**     | Matriz de Confusao, Curva ROC, Feature Importance (coeficientes)        |
| **Persistencia** | Atualiza `experiments_log.json` com metricas de avaliacao no blind test |

---

# 4. Fluxos Detalhados

## 4.1 Fluxo Principal do Sistema

```
[Usuario] --> python main.py
    |
    v
[reset_project_artifacts]
    Limpa data/processed/*.csv (exceto .gitkeep)
    |
    v
[load_and_split_data]
    Le data/raw/Base.csv
    Otimiza tipos (float64->float32, int64->int8)
    Valida coluna target (fraud_bool ou is_fraud)
    Split 80/20 estratificado
    Salva X_train.csv, X_test.csv, y_train.csv, y_test.csv
    |
    v
[EDAReporter.run] (se --skip-eda NAO ativo)
    Le data/raw/Base.csv
    Gera metricas de qualidade, testes estatisticos, graficos
    Salva em reports/
    |
    v
[compare_algorithms] (se --compare-models ativo)
    Le data/processed/X_train.csv, y_train.csv
    Amostra 50k linhas estratificadas
    Roda 5-Fold CV com SMOTE para 8-10 algoritmos
    Gera ranking e graficos
    |
    v
[Para cada modelo selecionado (--models)]:
    [train_*()]
        Le X_train.csv, y_train.csv
        Cria preprocessor via get_preprocessor()
        Monta Pipeline: preprocessor -> modelo
        GridSearchCV com StratifiedKFold (3 folds)
        Retreina no dataset completo com melhores params
        Threshold Tuning (maximiza F1 na curva PR)
        Salva modelo.pkl, threshold.txt, params.txt
        Registra em experiments_log.json
    [evaluate()]
        Le X_test.csv, y_test.csv
        Le modelo.pkl
        Gera predicoes e probabilidades
        Calcula metricas (AUC, F1, Precision, Recall)
        Gera Matriz de Confusao, Curva ROC
        Atualiza experiments_log.json
    |
    v
[predict_sample] (se --predict ativo)
    Le modelo.pkl e threshold.txt
    Sorteia transacoes do X_test
    Aplica motor de decisao (BLOQUEIO/REVISAO/APROVADO)
    Imprime resultado detalhado
```

## 4.2 Fluxo de Treinamento de Modelo (Generico)

```
train_*():
    1. run_id = timestamp atual
    2. X_train = pd.read_csv("data/processed/X_train.csv")
    3. y_train = pd.read_csv("data/processed/y_train.csv").ravel()
    4. preprocessor = get_preprocessor(X_train)  # Detecta tipos, monta pipelines
    5. clf = ModelClass(**params)
    6. pipeline = Pipeline([preprocessor, clf])
    7. [Opcional: Amostra 100k para GridSearch]
    8. GridSearchCV(pipeline, param_grid, scoring='roc_auc', cv=3)
    9. grid_search.fit(X_sample, y_sample)
   10. final_model = grid_search.best_estimator_
   11. final_model.fit(X_train, y_train)  # Retreino no completo
   12. joblib.dump(final_model, "*_best_model.pkl")
   13. joblib.dump(final_model, "model_*_{run_id}.pkl")  # Versionado
   14. y_proba = final_model.predict_proba(X_train)[:,1]
   15. precision_recall_curve --> argmax(F1) --> best_threshold
   16. Salva threshold.txt, params.txt
   17. Appenda experiment em experiments_log.json
```

## 4.3 Fluxo de Inferencia (Predicao)

```
predict_sample(model_name, n_samples):
    1. model = joblib.load("models/{model_name}_best_model.pkl")
    2. threshold = float(open("{model_name}_threshold.txt").read())
    3. X_test, y_test = carregar dados de teste
    4. Seleciona amostra balanceada (50% fraude, 50% legit)
    5. Para cada transacao:
       a. proba = model.predict_proba(transacao)[0,1]
       b. Se proba > threshold:       BLOQUEIO
          Se proba > threshold * 0.8: REVISAO MANUAL
          Senao:                       APROVADO
       c. Compara com gabarito (y_test)
```

## 4.4 Fluxo de Tratamento de Erros

Os erros sao tratados em 3 niveis:

1. **Importacao**: Bloco try/except global em `main.py` que aborta com `sys.exit(1)` se modulos estiverem faltando
2. **Dados**: `FileNotFoundError` se CSVs nao existirem; `ValueError` se coluna target nao for encontrada
3. **Modelos**: Fallback de threshold para 0.5 se `*_threshold.txt` nao existir

## 4.5 Fluxo de Logs

O sistema utiliza `logging.basicConfig` com saida para `stdout`. Cada modelo gera logs com formato padrao `%(asctime)s - %(levelname)s - %(message)s`. Alem disso, o `experiments_log.json` funciona como log persistido de todos os experimentos.

---

# 5. Banco de Dados

O Fraud Sentinel **nao utiliza banco de dados relacional ou NoSQL**. A persistencia e inteiramente baseada em arquivos:

| Tipo              | Formato             | Localizacao                    | Finalidade                      |
| ----------------- | ------------------- | ------------------------------ | ------------------------------- |
| Dados brutos      | CSV                 | `data/raw/Base.csv`            | Dataset BAF Suite original      |
| Dados processados | CSV                 | `data/processed/`              | Splits de treino/teste          |
| Modelos           | PKL (Pickle/Joblib) | `models/`                      | Modelos serializados            |
| Thresholds        | TXT                 | `models/`                      | Limiares de decisao otimizados  |
| Parametros        | TXT                 | `models/`                      | Hiperparametros vencedores      |
| Historico         | JSON                | `reports/experiments_log.json` | Log unificado de experimentos   |
| Metricas EDA      | CSV                 | `reports/data/`                | Tabelas de analise exploratoria |
| Visualizacoes     | PNG                 | `reports/figures/`             | Graficos de avaliacao           |
| Dashboard         | HTML                | `reports/sweetviz_report.html` | EDA interativa                  |

A estrategia de "banco de dados" e baseada em **flat files**, onde cada execucao appenda ao `experiments_log.json` e versiona modelos com timestamps no nome (`model_xgb_20260218_105559.pkl`).

---

# 6. Regras de Negocio

## 6.1 Regras de Classificacao

- O target e binario: `fraud_bool` = 0 (legitima) ou 1 (fraude)
- O threshold de decisao NAO e 0.5 fixo. E otimizado por modelo via maximizacao do F1-Score na curva Precision-Recall
- A decisao final e trinivelada: BLOQUEIO automatico, REVISAO manual, ou APROVACAO

## 6.2 Regras de Balanceamento

- SMOTE foi testado nos experimentos 1 e 2, mas **removido** no experimento final por gerar "Dupla Penalizacao" (excesso de falsos positivos)
- A estrategia vencedora usa **Cost-Sensitive Learning**: `class_weight='balanced'` (LogReg, RF, DT) ou `scale_pos_weight=90` (XGBoost)

## 6.3 Regras de Validacao

- Split estratificado obrigatorio (`stratify=y`) para manter proporcao de fraude
- Preprocessamento ocorre **dentro** do pipeline para evitar Data Leakage
- SMOTE (quando usado) e aplicado **dentro** de cada fold de validacao cruzada
- GridSearch avalia por `roc_auc` (metrica independente de threshold)

## 6.4 Regras de Versionamento de Modelos

- Modelo "latest" em `{nome}_best_model.pkl` (sempre sobrescrito)
- Modelo historico em `model_{nome}_{timestamp}.pkl` (nunca sobrescrito)
- Pasta de modelos nao e limpa pelo reset para manter historico

## 6.5 Regras de Motor de Decisao

```
Se score > threshold:             --> BLOQUEIO AUTOMATICO (Alto Risco)
Se score > threshold * 0.8:       --> REVISAO MANUAL (Medio Risco)
Se score <= threshold * 0.8:      --> APROVADO (Baixo Risco)
```

---

# 7. Integracoes Externas

O projeto **nao possui integracoes com APIs externas** em tempo de execucao. Todas as dependencias sao bibliotecas Python instaladas localmente.

## 7.1 Bibliotecas Criticas

| Biblioteca           | Versao          | Finalidade                                  |
| -------------------- | --------------- | ------------------------------------------- |
| scikit-learn         | 1.8.0           | Pipelines, modelos, metricas, preprocessing |
| xgboost              | 3.2.0           | Gradient Boosting otimizado                 |
| imbalanced-learn     | 0.14.1          | SMOTE e ImbPipeline                         |
| lightgbm             | 4.6.0           | Gradient Boosting (benchmark)               |
| pandas               | 2.3.3           | Manipulacao de dados tabulares              |
| numpy                | 2.3.5           | Operacoes numericas                         |
| matplotlib / seaborn | 3.10.0 / 0.13.2 | Visualizacoes                               |
| sweetviz             | 2.3.1           | Dashboard EDA interativo                    |
| scipy                | 1.16.3          | Testes estatisticos (Mann-Whitney)          |
| joblib               | 1.5.3           | Serializacao de modelos                     |
| tqdm                 | 4.67.3          | Barras de progresso (opcional)              |

---

# 8. Logica e Algoritmos

## 8.1 Otimizacao de Memoria (Downcasting)

O algoritmo em `optimize_memory_usage()` itera por todas as colunas e verifica se o range de valores cabe em tipos menores (`int8`, `int16`, `float32`), reduzindo significativamente o footprint de memoria para datasets com milhoes de linhas.

## 8.2 Threshold Tuning

Todos os modelos executam apos o treinamento:

1. Calculam `predict_proba` no conjunto de treino
2. Geram a curva Precision-Recall com todos os thresholds possiveis
3. Calculam F1 = 2*(P*R)/(P+R) para cada threshold
4. Selecionam o threshold que maximiza F1

Isso substitui o corte ingenuo de 0.5 por um ponto de operacao otimizado para o problema.

## 8.3 IForestWrapper (Adapter Pattern)

O Isolation Forest nao implementa `predict_proba` nativamente. O `IForestWrapper` aplica o padrao Adapter:

1. `fit()`: Treina o IF e ajusta um `MinMaxScaler` nos scores de decisao invertidos
2. `predict()`: Converte -1 (anomalia) para 1 (fraude) e 1 (normal) para 0
3. `predict_proba()`: Normaliza scores para [0,1] e retorna formato `(n, 2)`

## 8.4 Amostragem Estratificada para GridSearch

LogReg e XGBoost usam amostra de 100k linhas para GridSearch (economia de horas de processamento), seguida de retreino no dataset completo com os parametros vencedores.

## 8.5 Informacao Mutua (MI)

A EDA calcula Mutual Information com `mutual_info_classif` para ranquear features por capacidade preditiva, capturando relacoes nao-lineares que correlacao de Pearson/Spearman ignora.

---

# 9. Configuracoes e Variaveis de Ambiente

| Variavel       | Arquivo             | Finalidade                     | Valor        |
| -------------- | ------------------- | ------------------------------ | ------------ |
| `RANDOM_STATE` | `config.py`         | Semente para reprodutibilidade | 42           |
| `TEST_SIZE`    | `config.py`         | Proporcao do split de teste    | 0.2          |
| `TARGET_COL`   | `config.py`         | Coluna alvo                    | `fraud_bool` |
| `SAMPLE_SIZE`  | Modelos individuais | Amostra para GridSearch        | 100000       |
| `CV_FOLDS`     | Modelos individuais | Folds de validacao cruzada     | 3            |
| `.env`         | Raiz                | Reservado (vazio)              | --           |

---

# 10. Como Executar o Projeto

## 10.1 Requisitos

- Python 3.12+
- ~8GB RAM (recomendado para dataset BAF completo)
- Windows 10/11 (testado)

## 10.2 Instalacao

```bash
git clone https://github.com/Marocosz/fraud-sentinel.git
cd fraud-sentinel
python -m venv venvmine
venvmine\Scripts\activate    # Windows
pip install -r requirements.txt
```

## 10.3 Preparacao dos Dados

Baixe o dataset BAF Suite do Kaggle e coloque em `data/raw/Base.csv`:

- URL: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

## 10.4 Execucao

```bash
# Pipeline completo (primeira vez)
python main.py

# Sem EDA (mais rapido)
python main.py --skip-eda

# Apenas modelos especificos
python main.py --skip-eda --models xgb,rf

# Com benchmark de algoritmos (demorado)
python main.py --compare-models

# Com simulacao de producao
python main.py --predict

# Sem limpeza (reuso de dados processados)
python main.py --no-reset --skip-eda --models xgb

# Ajuste fino de Precision
python src/models/force_precision.py 0.20
```

---

# 11. Estrategia de Logs e Monitoramento

## 11.1 Logs em Console

Todos os modulos utilizam `logging.basicConfig` com nivel `INFO` e saida para `stdout`. O formato padrao e `%(asctime)s - %(levelname)s - %(message)s`.

## 11.2 Log Persistido (experiments_log.json)

Cada treinamento appenda um registro contendo:

- `run_id` (timestamp)
- `model_type`, `smote_strategy`, `best_params`
- `best_cv_score`, `best_threshold`
- `model_path` (nome do arquivo versionado)
- Metricas de avaliacao (AUC, classification_report, confusion_matrix) -- adicionadas pelo `visualize.py`

## 11.3 Diagnostico de Problemas

1. **Modelo nao encontrado**: Verificar se `main.py` foi executado antes de `predict_model.py`
2. **Memoria insuficiente**: Reduzir `SAMPLE_SIZE` nos modelos ou usar `--models` para treinar menos modelos
3. **Threshold nao encontrado**: O sistema faz fallback para 0.5 com warning

---

# 12. Pontos Criticos do Sistema

## 12.1 Gargalos de Performance

- **Random Forest GridSearch**: Com `n_estimators=200` e `max_depth=None`, pode gerar modelos de 300MB+ e demorar horas
- **MLP GridSearch**: 36 combinacoes de hiperparametros com 3 folds = 108 treinamentos de rede neural
- **EDA com Sweetviz**: O dashboard HTML pode demorar minutos para datasets grandes

## 12.2 Riscos Arquiteturais

- **Threshold calculado no conjunto de treino**: Idealmente deveria usar um validation set separado para evitar overfitting do threshold
- **Pipeline refitado a cada chamada**: `get_preprocessor()` cria um novo ColumnTransformer a cada treinamento. Nao ha garantia de consistencia entre modelos se o schema mudar
- **Sem pipeline de CI/CD**: Nao ha testes automatizados funcionais

## 12.3 Partes Sensiveis

- `experiments_log.json`: Append concorrente pode corromper JSON se dois treinamentos rodarem simultaneamente
- `data/raw/Base.csv`: Sem validacao de schema (colunas podem mudar entre versoes do dataset)

---

# 13. Teoria Tecnica Envolvida

## 13.1 Padroes de Projeto

| Padrao              | Aplicacao                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| **Pipeline**        | Encadeamento de transformacoes via `sklearn.pipeline.Pipeline`                                    |
| **Strategy**        | Cada `*_model.py` implementa a mesma interface de treinamento com algoritmos diferentes           |
| **Adapter**         | `IForestWrapper` adapta a interface do Isolation Forest para compatibilidade com `predict_proba`  |
| **Template Method** | Todos os modelos seguem o mesmo esqueleto: carga -> pipeline -> grid -> threshold -> persistencia |
| **Registry**        | `experiments_log.json` funciona como registro central de experimentos                             |

## 13.2 Conceitos de ML Aplicados

- **Stratified K-Fold**: Mantem proporcao de classes em cada fold, essencial para dados desbalanceados
- **Cost-Sensitive Learning**: Penaliza erros na classe minoritaria sem gerar dados sinteticos
- **Threshold Tuning**: Otimiza o ponto de operacao do modelo para o trade-off Precision/Recall desejado
- **RobustScaler**: Normalizacao baseada em mediana e IQR, imune a outliers financeiros
- **One-Hot Encoding com handle_unknown='ignore'**: Tolera categorias novas em producao sem quebrar

## 13.3 Conceitos Estatisticos

- **Mann-Whitney U Test**: Teste nao-parametrico para verificar se distribuicoes de fraude e legitimidade diferem significativamente
- **Mutual Information**: Mede dependencia estatistica geral (linear e nao-linear) entre features e target
- **IQR (Interquartile Range)**: Metodo robusto para deteccao de outliers
- **Correlacao de Spearman**: Captura relacoes monotonicas sem assumir linearidade

---

# 14. Melhorias Futuras

## 14.1 Sugestoes Estruturais

1. **Criar classe base abstrata `BaseModelTrainer`** para eliminar duplicacao entre os 6 arquivos de modelo
2. **Implementar testes unitarios** (arquivos em `tests/` estao vazios)
3. **Separar validation set** do train set para threshold tuning mais robusto
4. **Adicionar type hints** e docstrings padrao (Google style ou NumPy style)

## 14.2 Melhorias de Performance

1. **Substituir GridSearchCV por Optuna/BayesSearchCV** para busca mais eficiente de hiperparametros
2. **Implementar cache de preprocessamento** para evitar recomputacao em cada modelo
3. **Usar Parquet em vez de CSV** para leitura 5-10x mais rapida

## 14.3 Refatoracoes Recomendadas

1. **Extrair logica de persistencia de experimentos** para um modulo `experiment_tracker.py`
2. **Centralizar configuracao de modelos** em um unico YAML/JSON em vez de dicionarios por arquivo
3. **Implementar SHAP/LIME** para explicabilidade (placeholder existe em `predict_model.py`)

---

# 15. Analise Critica da Arquitetura

## 15.1 Codigo Duplicado (Alto Impacto)

Os 6 arquivos de modelo (`reg_log_model.py`, `random_forest_model.py`, `xgboost_model.py`, `decision_tree_model.py`, `mlp_model.py`, `isolation_forest_model.py`) compartilham **~80% do codigo identico**:

- Carga de dados (identica em todos)
- Chamada a `get_preprocessor()` (identica)
- Montagem do pipeline (identica, exceto modelo)
- Criacao do GridSearchCV (identica, exceto param_grid)
- Persistencia de modelo (identica, exceto nome)
- Threshold Tuning (identico)
- Registro em `experiments_log.json` (identico)

**Recomendacao**: Criar uma classe base `BaseTrainer` com metodo `train()` generico, onde subclasses definem apenas `MODEL_CONFIG`.

## 15.2 Chave Duplicada no Dicionario

Em `decision_tree_model.py`, linha 42-43, a chave `"model_class"` e definida duas vezes no `MODEL_CONFIG`:

```python
MODEL_CONFIG = {
    "model_class": DecisionTreeClassifier,
    "model_class": DecisionTreeClassifier,  # DUPLICADA
    ...
}
```

Nao causa erro funcional (Python usa a ultima), mas indica copy-paste descuidado.

## 15.3 Variavel Nao Utilizada

Em `decision_tree_model.py`, linha 106, a variavel `year_now` e atribuida mas nunca utilizada:

```python
year_now = datetime.datetime.now().year  # Nao usado em nenhum lugar
```

## 15.4 Import Duplicado

Em `reg_log_model.py`, `precision_recall_curve` e importado duas vezes: no topo (linha 14) e dentro da funcao (linha 234).

## 15.5 Testes Nao Implementados

Os arquivos `tests/test_data.py` e `tests/test_features.py` contem apenas comentarios, sem nenhum teste real. Isso representa um risco significativo para manutencao.

## 15.6 Inconsistencia na Estrategia de Amostragem para GridSearch

- LogReg e XGBoost usam amostra de 100k linhas para GridSearch + retreino no completo
- Random Forest, Decision Tree e MLP usam o dataset completo direto no GridSearch
- Essa inconsistencia pode resultar em tempos de treinamento drasticamente diferentes

## 15.7 Acoplamento com Sistema de Arquivos

Todos os modulos dependem diretamente de caminhos de arquivo para comunicacao. Nao ha abstracoes de I/O, tornando dificil adaptar para outros meios de armazenamento (S3, banco de dados, etc.).

---
