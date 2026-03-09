# Fraud Sentinel - Sistema Avancado de Deteccao de Fraudes Bancarias

# Visao Geral do Projeto

| Item                     | Descricao                                                                                                                                                                                                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Nome**                 | Fraud Sentinel                                                                                                                                                                                                                                                                                        |
| **Objetivo Principal**   | Desenvolver um pipeline completo de Machine Learning para detectar fraudes em aberturas de contas bancarias, priorizando a maximizacao do Recall (capturar o maximo de fraudes) com controle de Precision (minimizar falsos alarmes).                                                                 |
| **Problema que Resolve** | Fraudes bancarias na abertura de contas causam prejuizos financeiros massivos. O sistema automatiza a triagem de solicitacoes, classificando-as como legitimas ou fraudulentas com base em padroes historicos de comportamento e atributos sociodemograficos.                                         |
| **Contexto de Uso**      | O sistema opera sobre o dataset Bank Account Fraud (BAF) Suite, publicado no NeurIPS 2022, que simula dados reais de aberturas de conta com rotulagem binaria (0 = legitima, 1 = fraude). A taxa de fraude e extremamente baixa (~1%), exigindo tecnicas especializadas de balanceamento e avaliacao. |
| **Tarefa de Mineracao**  | Classificacao Binaria Supervisionada (O problema e resolvido testando multiplos algoritmos e os unindo em um comite de Ensemble, mas a _"natureza"_ do problema matematico ainda e de Classificacao Binaria).                                                                                         |
| **Metodologia**          | CRISP-DM (Cross-Industry Standard Process for Data Mining)                                                                                                                                                                                                                                            |

---

# Como Executar o Projeto

CRIAR DOCKER

## Preparacao dos Dados

Baixe o dataset BAF Suite do Kaggle e coloque em `data/raw/Base.csv`:

> [!NOTE]
> Link para download da base de dados usada (drive): https://drive.google.com/file/d/1KWKHddAwpZ2HAwsWmL0HWlUXFw8HWf9N/view?usp=sharing
>
> Link para download da base de dados usada (kaggle): https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

## Execucao

```bash
# Pipeline completo (primeira vez)
python main.py

# Sem EDA (mais rapido)
python main.py --skip-eda

# Apenas modelos especificos
python main.py --skip-eda --models xgb,rf

# Com simulacao de producao
python main.py --predict

# Execucao limitando a base crua a 100 mil registros (Ideal para Testes Rapidos DEV)
python main.py --max-samples 100000

# Execucao com Random Undersampling Teorico (Ex: 0.5 mantem as fraudes e sorteia o dobro de sadios no Treino)
python main.py --undersampling-ratio 0.5

# Sem limpeza (reuso de dados processados)
python main.py --no-reset --skip-eda --models xgb

# Ajuste fino de Precision
python src/models/force_precision.py 0.20
```

# Sumário

- [Fraud Sentinel - Sistema Avancado de Deteccao de Fraudes Bancarias](#fraud-sentinel---sistema-avancado-de-deteccao-de-fraudes-bancarias)
- [Visao Geral do Projeto](#visao-geral-do-projeto)
- [Como Executar o Projeto](#como-executar-o-projeto)
  - [Preparacao dos Dados](#preparacao-dos-dados)
  - [Execucao](#execucao)
- [Sumário](#sumário)
- [1 Resultados da Analise Exploratoria (EDA)](#1-resultados-da-analise-exploratoria-eda)
  - [1.1 Carga e Qualidade Visão Geral](#11-carga-e-qualidade-visão-geral)
  - [1.2 O Problema do Desbalanceamento](#12-o-problema-do-desbalanceamento)
  - [1.3 Significância Estatística das Variáveis](#13-significância-estatística-das-variáveis)
  - [1.4 Importância de Features via Mutual Information (MI)](#14-importância-de-features-via-mutual-information-mi)
  - [1.5 Análise de Risco Categórico: Onde mora a Fraude?](#15-análise-de-risco-categórico-onde-mora-a-fraude)
  - [1.6 O Problema de Outliers: Escala é Essencial](#16-o-problema-de-outliers-escala-é-essencial)
  - [1.7 Downcasting de Memória Otimizada](#17-downcasting-de-memória-otimizada)
- [2. Arquitetura Geral](#2-arquitetura-geral)
  - [2.1 Tipo de Arquitetura](#21-tipo-de-arquitetura)
  - [2.2 Diagrama da Arquitetura](#22-diagrama-da-arquitetura)
  - [2.3 Fluxo Macro (Requisicao ate Resposta)](#23-fluxo-macro-requisicao-ate-resposta)
    - [1. Ingestão de Dados](#1-ingestão-de-dados)
    - [2. Análise Exploratória (EDA)](#2-análise-exploratória-eda)
    - [4. Treinamento e Otimização](#4-treinamento-e-otimização)
    - [5. Avaliação Final](#5-avaliação-final)
    - [6. Previsão Discreta (Opcional)](#6-previsão-discreta-opcional)
  - [2.4 Separacao de Camadas](#24-separacao-de-camadas)
- [3. Estrutura de Diretorios](#3-estrutura-de-diretorios)
  - [3.1 Descricao Detalhada de Cada Arquivo](#31-descricao-detalhada-de-cada-arquivo)
    - [main.py - Orquestrador Principal](#mainpy---orquestrador-principal)
    - [src/data/make_dataset.py - Engenharia de Dados](#srcdatamake_datasetpy---engenharia-de-dados)
    - [src/features/build_features.py - Pipeline de Features (EDA-Driven)](#srcfeaturesbuild_featurespy---pipeline-de-features-eda-driven)
    - [src/models/trainers/reg_log_model.py - Logistic Regression](#srcmodelstrainersreg_log_modelpy---logistic-regression)
    - [src/models/trainers/decision_tree_model.py - Decision Tree](#srcmodelstrainersdecision_tree_modelpy---decision-tree)
    - [src/models/trainers/random_forest_model.py - Random Forest](#srcmodelstrainersrandom_forest_modelpy---random-forest)
    - [src/models/trainers/xgboost_model.py - XGBoost](#srcmodelstrainersxgboost_modelpy---xgboost)
    - [src/models/trainers/mlp_model.py - MLP Neural Network](#srcmodelstrainersmlp_modelpy---mlp-neural-network)
    - [src/models/trainers/isolation_forest_model.py -- Isolation Forest](#srcmodelstrainersisolation_forest_modelpy----isolation-forest)
    - [src/serving/simulate_production.py -- Simulacao de Producao](#srcservingsimulate_productionpy----simulacao-de-producao)
    - [src/models/force_precision.py -- Ajuste de Precision-Alvo](#srcmodelsforce_precisionpy----ajuste-de-precision-alvo)
    - [src/visualization/generate_eda_report.py -- EDA Automatizada](#srcvisualizationgenerate_eda_reportpy----eda-automatizada)
    - [src/visualization/visualize.py -- Avaliacao Final](#srcvisualizationvisualizepy----avaliacao-final)
- [4. Fluxos Detalhados](#4-fluxos-detalhados)
  - [4.1 Fluxo Principal do Sistema](#41-fluxo-principal-do-sistema)
  - [4.2 Fluxo de Treinamento de Modelo (Generico)](#42-fluxo-de-treinamento-de-modelo-generico)
  - [4.3 Fluxo de Inferencia (Predicao com Ensemble)](#43-fluxo-de-inferencia-predicao-com-ensemble)
  - [4.4 Fluxo de Tratamento de Erros](#44-fluxo-de-tratamento-de-erros)
  - [4.5 Fluxo de Logs](#45-fluxo-de-logs)
- [5. Banco de Dados](#5-banco-de-dados)
- [6. Regras de Negocio](#6-regras-de-negocio)
  - [6.1 Regras de Classificacao](#61-regras-de-classificacao)
  - [6.2 Regras de Balanceamento](#62-regras-de-balanceamento)
  - [6.3 Regras de Validacao](#63-regras-de-validacao)
  - [6.4 Regras de Versionamento de Modelos](#64-regras-de-versionamento-de-modelos)
  - [6.5 Regras de Motor de Decisao](#65-regras-de-motor-de-decisao)
- [7. Logica e Algoritmos](#7-logica-e-algoritmos)
  - [7.1 Otimizacao de Memoria (Downcasting)](#71-otimizacao-de-memoria-downcasting)
  - [7.2 Threshold Tuning](#72-threshold-tuning)
  - [7.3 IForestWrapper (Adapter Pattern)](#73-iforestwrapper-adapter-pattern)
  - [7.4 Amostragem Estratificada para GridSearch](#74-amostragem-estratificada-para-gridsearch)
  - [7.5 Informacao Mutua (MI)](#75-informacao-mutua-mi)
  - [7.6 EDAFeatureEngineer (Feature Engineering Orientado por Dados)](#76-edafeatureengineer-feature-engineering-orientado-por-dados)
  - [7.7 Abstração de Treinamento (`BaseTrainer`)](#77-abstração-de-treinamento-basetrainer)
  - [7.8 Motor de Simulação (Ensemble PoV \& ROI)](#78-motor-de-simulação-ensemble-pov--roi)
- [8. Configuracoes e Variaveis de Ambiente](#8-configuracoes-e-variaveis-de-ambiente)
- [9. Estrategia de Logs e Monitoramento](#9-estrategia-de-logs-e-monitoramento)
  - [9.1 Logs em Console](#91-logs-em-console)
  - [9.2 Log Persistido (experiments_log.json)](#92-log-persistido-experiments_logjson)
  - [9.3 Diagnostico de Problemas](#93-diagnostico-de-problemas)
- [10. Teoria Tecnica Envolvida](#10-teoria-tecnica-envolvida)
  - [10.1 Padrões de Projeto (Engenharia de Software em MLOps)](#101-padrões-de-projeto-engenharia-de-software-em-mlops)
  - [10.2 A Teoria dos Modelos: Por que Foram Selecionados e Como Operam](#102-a-teoria-dos-modelos-por-que-foram-selecionados-e-como-operam)
    - [1. Regressão Logística (Logistic Regression)](#1-regressão-logística-logistic-regression)
    - [2. Árvores de Decisão \& Random Forest (Ensambles de Bagging)](#2-árvores-de-decisão--random-forest-ensambles-de-bagging)
    - [3. Máquinas de Vetores de Gradiente (Gradient Boosting - XGBoost e LightGBM)](#3-máquinas-de-vetores-de-gradiente-gradient-boosting---xgboost-e-lightgbm)
    - [4. Redes Neurais Profundas (Multilayer Perceptron - MLP)](#4-redes-neurais-profundas-multilayer-perceptron---mlp)
    - [5. Isolation Forest (Não-Supervisionado / Geométrico)](#5-isolation-forest-não-supervisionado--geométrico)
  - [10.3 O Enfrentamento do Desbalanceamento Extremo Sistemático](#103-o-enfrentamento-do-desbalanceamento-extremo-sistemático)
  - [10.4 Threshold Tuning vs Corte Euclidiano Clássico](#104-threshold-tuning-vs-corte-euclidiano-clássico)
  - [10.5 O Porquê da Estatística não-Paramétrica Adotada na EDA](#105-o-porquê-da-estatística-não-paramétrica-adotada-na-eda)
- [11. Melhorias Futuras](#11-melhorias-futuras)
  - [11.1 Melhorias de Performance](#111-melhorias-de-performance)
  - [11.2 Refatoracoes Recomendadas](#112-refatoracoes-recomendadas)
- [12. Análise Crítica e Explicação dos Experimentos](#12-análise-crítica-e-explicação-dos-experimentos)
  - [12.1 Os Casos de Fracasso (Exemplos Negativos)](#121-os-casos-de-fracasso-exemplos-negativos)
    - [12.1.1 Árvore de Decisão Simples (`DecisionTreeClassifier`)](#1211-árvore-de-decisão-simples-decisiontreeclassifier)
    - [12.1.2 Floresta de Isolamento (`IForestWrapper` / Isolation Forest)](#1212-floresta-de-isolamento-iforestwrapper--isolation-forest)
  - [12.2. A Transição e o Progresso](#122-a-transição-e-o-progresso)
    - [12.2.1 Floresta Aleatória (`RandomForestClassifier`)](#1221-floresta-aleatória-randomforestclassifier)
    - [12.2.2 Regressão Logística (`LogisticRegression`)](#1222-regressão-logística-logisticregression)
  - [12.3 A Vanguarda Numérica (Exemplos Positivos)](#123-a-vanguarda-numérica-exemplos-positivos)
    - [12.3.1 Redes Neurais Cíbridas (`MLPClassifier` - Perceptron Multicamadas)](#1231-redes-neurais-cíbridas-mlpclassifier---perceptron-multicamadas)
    - [12.3.2 Máquinas de Gradient Boosting (`XGBClassifier` \& `LGBMClassifier`)](#1232-máquinas-de-gradient-boosting-xgbclassifier--lgbmclassifier)

# 1 Resultados da Analise Exploratoria (EDA)

<details>
<summary>Clique para expandir a Análise Exploratória de Dados (EDA)</summary>

A partir do arquivo `generate_eda_report.py` criamos um relatorio textual (`reports/eda_summary.txt`) com o sumario completo da base de dados. A seguir estao todas as informacoes e descricoes geradas a partir da analise da base inicial.

## 1.1 Carga e Qualidade Visão Geral

O dataset analisado simula aberturas de contas bancárias com uma variável alvo (`fraud_bool`) indicando legitimidade (0) ou fraude (1). O volume de dados é massivo, o que garante representatividade estatística.

- **Volume de Dados:** 1.000.000 de registros únicos.
- **Dimensionalidade:** 32 features (9 Numéricas Contínuas, 18 Numéricas Discretas, 5 Categóricas).
- **Qualidade Básica:** 0 nulos e 0 duplicados confirmados. A integridade estrutural da base economiza etapas de _imputation_ massivas.

<br>

<details>
<summary><b> Ver Tabela Completa de Cardinalidade e Tipagem</b></summary>

| Coluna                         | Tipo    | Cardinalidade |     | Coluna                      | Tipo    | Cardinalidade |
| ------------------------------ | ------- | ------------- | --- | --------------------------- | ------- | ------------- |
| `fraud_bool`                   | int64   | 2             |     | `credit_risk_score`         | int64   | 551           |
| `income`                       | float64 | 9             |     | `email_is_free`             | int64   | 2             |
| `name_email_similarity`        | float64 | 998.861       |     | `housing_status`            | object  | 7             |
| `prev_address_months_count`    | int64   | 374           |     | `phone_home_valid`          | int64   | 2             |
| `current_address_months_count` | int64   | 423           |     | `phone_mobile_valid`        | int64   | 2             |
| `customer_age`                 | int64   | 9             |     | `bank_months_count`         | int64   | 33            |
| `days_since_request`           | float64 | 989.330       |     | `has_other_cards`           | int64   | 2             |
| `intended_balcon_amount`       | float64 | 994.971       |     | `proposed_credit_limit`     | float64 | 12            |
| `payment_type`                 | object  | 5             |     | `foreign_request`           | int64   | 2             |
| `zip_count_4w`                 | int64   | 6.306         |     | `source`                    | object  | 2             |
| `velocity_6h`                  | float64 | 998.687       |     | `session_length_in_minutes` | float64 | 994.887       |
| `velocity_24h`                 | float64 | 998.940       |     | `device_os`                 | object  | 5             |
| `velocity_4w`                  | float64 | 998.318       |     | `keep_alive_session`        | int64   | 2             |
| `bank_branch_count_8w`         | int64   | 2.326         |     | `device_distinct_emails_8w` | int64   | 4             |
| `date_of_birth_distinct...`    | int64   | 40            |     | `device_fraud_count`        | int64   | 1             |
| `employment_status`            | object  | 7             |     | `month`                     | int64   | 8             |

</details>

## 1.2 O Problema do Desbalanceamento

O principal desafio deste Dataset não é a qualidade do dado, mas sim a assimetria da variável alvo:

- **Contas Legítimas:** **98.90%** (988.971)
- **Fraudes:** Apenas **1.10%** (11.029)

**🔹 O que isso significa para o Negócio?**
Um modelo "burro" que negue todas as aberturas de conta teria 98.90% de "Acurácia", mas quebraria a instituição bancária ao rejeitar todos os clientes legítimos. A prioridade matemática passa a ser métricas como **Recall** (quantas das fraudes reais nós pegamos?) controlando o **F1-Score / Precision** (quantos clientes bons nós negamos por engano?). Isso nos força a usar técnicas de _Cost-Sensitive Learning_ na modelagem.

<br>

## 1.3 Significância Estatística das Variáveis

Buscamos entender se o padrão de quem comete fraude é genuinamente diferente do cliente comum. Para variáveis contínuas que raramente seguem Distribuição Normal (Gaussiana), não testamos apenas médias, mas toda a "forma" da curva através do Teste U de Mann-Whitney.

> \*ℹ️ **Glossário Explicativo: Teste U de Mann-Whitney\***
> _Um teste estatístico não-paramétrico. Em vez de calcular qual a média da idade dos fraudadores vs legítimos, ele ranqueia todos os clientes ordenados por idade e avalia se os fraudadores consistentemente ocupam posições mais altas rankeadas de forma sistemática._

**Resultado:** 24 de 26 variáveis numéricas apresentam padrões matemáticos significativamente diferentes para o grupo de fraude (p-value < 0.05). As exclusões foram `session_length_in_minutes` e `device_fraud_count`.

<details>
<summary><b> Expandir Resultados do Teste de Hipótese (P-Values)</b></summary>

| Variavel                           | Mann-Whitney Stat | P-Value   | Conclusao              |
| ---------------------------------- | ----------------- | --------- | ---------------------- |
| `customer_age`                     | 65.564.824,5      | ~0.00     | Significativo (p<0.05) |
| `credit_risk_score`                | 66.698.262,5      | ~0.00     | Significativo (p<0.05) |
| `prev_address_months_count`        | 39.783.177,5      | 6.23e-300 | Significativo (p<0.05) |
| `proposed_credit_limit`            | 63.877.277,0      | 8.18e-296 | Significativo (p<0.05) |
| `income`                           | 64.323.855,0      | 4.86e-281 | Significativo (p<0.05) |
| `current_address_months_count`     | 63.932.505,5      | 2.91e-255 | Significativo (p<0.05) |
| `keep_alive_session`               | 38.390.000,0      | 3.35e-238 | Significativo (p<0.05) |
| `date_of_birth_distinct_emails_4w` | 37.493.070,0      | 5.98e-207 | Significativo (p<0.05) |
| `has_other_cards`                  | 42.870.000,0      | 1.46e-170 | Significativo (p<0.05) |
| `name_email_similarity`            | 39.904.832,0      | 5.43e-135 | Significativo (p<0.05) |
| `phone_home_valid`                 | 41.975.000,0      | 6.33e-128 | Significativo (p<0.05) |
| `bank_branch_count_8w`             | 41.159.136,0      | 2.41e-105 | Significativo (p<0.05) |
| `email_is_free`                    | 56.960.000,0      | 3.28e-89  | Significativo (p<0.05) |
| `device_distinct_emails_8w`        | 53.088.699,0      | 4.00e-66  | Significativo (p<0.05) |
| `intended_balcon_amount`           | 45.446.320,0      | 6.85e-29  | Significativo (p<0.05) |
| `velocity_6h`                      | 45.459.988,5      | 9.98e-29  | Significativo (p<0.05) |
| `foreign_request`                  | 51.275.000,0      | 1.52e-21  | Significativo (p<0.05) |
| `days_since_request`               | 46.319.316,5      | 1.96e-19  | Significativo (p<0.05) |
| `month`                            | 53.497.183,5      | 5.90e-18  | Significativo (p<0.05) |
| `phone_mobile_valid`               | 47.975.000,0      | 1.85e-17  | Significativo (p<0.05) |
| `velocity_4w`                      | 46.741.244,0      | 1.44e-15  | Significativo (p<0.05) |
| `bank_months_count`                | 46.855.019,5      | 4.28e-15  | Significativo (p<0.05) |
| `velocity_24h`                     | 47.016.792,0      | 2.73e-13  | Significativo (p<0.05) |
| `zip_count_4w`                     | 51.794.874,5      | 1.10e-05  | Significativo (p<0.05) |
| `session_length_in_minutes`        | 50.570.011,0      | 1.63e-01  | Nao Significativo      |
| `device_fraud_count`               | 50.000.000,0      | 1.00      | Nao Significativo      |

</details>

<br>

## 1.4 Importância de Features via Mutual Information (MI)

Correlação de Pearson (linear) falha miseravelmente em contextos de fraude onde os preditores são catérgicos binários ou não lineares. O Score de "Informação Mútua" (MI) é matematicamente imune a esses problemas estatísticos.

> \*ℹ️ **Glossário Explicativo: Mutual Information (MI)\***
> _Mede a redução da nossa "incerteza" estatística sobre a fraude ao conhecermos uma dada variável. É a quantidade de informação (em bits ou nats) que uma variável compartilha sobre a outra, podendo capturar sinergias ocultas._

**🔹 O que isso significa para o Negócio?**
As três variáveis de mais alto poder preditivo de fraude não são "idade" ou "renda" do cliente, mas sim sua "Jornada Digital" no momento da abertura da conta. O ranking de MI nos revela um atacante que:

1. Reitera o uso do dispositivo com e-mails massivos diferenciados (`device_distinct_emails`).
2. Utiliza e-mails gratuitos (`email_is_free`) que são muito fáceis de serem gerados em lote por robôs.
3. Não mantém engajamento duradouro com a sessão (`keep_alive_session`).

<details>
<summary><b> Expandir Ranking de Mutual Information</b></summary>

| Posicao | Variavel                                                    | MI Score |
| ------- | ----------------------------------------------------------- | -------- |
| 1       | `device_distinct_emails_8w`                                 | 0.010217 |
| 2       | `email_is_free`                                             | 0.010028 |
| 3       | `keep_alive_session`                                        | 0.009970 |
| 4       | `phone_mobile_valid`                                        | 0.007698 |
| 5       | `phone_home_valid`                                          | 0.006016 |
| 6       | `proposed_credit_limit`                                     | 0.004636 |
| 7       | `customer_age`                                              | 0.004512 |
| 8       | `income`                                                    | 0.003630 |
| 9       | `has_other_cards`                                           | 0.001935 |
| 10      | `credit_risk_score`                                         | 0.001892 |
| ...     | demais contidas na matriz integral do arquivo `eda_summary` | ...      |

</details>

<br>

## 1.5 Análise de Risco Categórico: Onde mora a Fraude?

Os modelos de regressão de árvore de decisão irão iterar sobre fatias dos dados (Splits de nós). A análise das frequências revelam de antemão por onde as partições iniciarão o corte das variáveis de alto impacto na triagem:

- **Sistema Operacional do Fraudadador (`device_os`):** Acessos através de navegadores Desktop Windows possuíram taxa de Fraude base de **2.47%** (mais do que o dobro do padrão global da base).
- **Emprego (`employment_status`):** Usuários classificados pela flag opaca "CC" tem incidência de **2.47%**.
- **Moradia (`housing_status`):** Atenção crítica à tag de moradia "BA". Ela puxa estonteantes **3.75%** de volume de fraude na sub-amostra; Este é um nicho quase 3.4x mais tóxico estatisticamente do que um usuário não pertencente a esta classe.

**🔹 O que isso significa para o Negócio?**
Um solicitante via _Teleapp_ (Aplicativo Telefonico - Fraude Base = 1.59%), que utilize Windows (Fraude= 2.47%) e seja da base Habitacional Categoria _BA_ (Fraude=3.75%) deve ser sumariamente bloqueado ou submetido à triagem manual pesada de prevenção antes da emissão de cartões temporários.

<br>

## 1.6 O Problema de Outliers: Escala é Essencial

Em modelos não baseados em ramificação em Árvores (ex. Regressão Logística e Redes Neurais - Multilayer Perceptron), outliers extremos causam explosões nos vetores de otimização de gradiente.

> \*ℹ️ **Glossário Explicativo: Método IQR (Interquartile Range)\***
> _Ao invés de definir que um valor é anômalo só porque ele é maior que a média + Desvio Padrão, o método inter-quartil ordena todo o dataset, observa estritamente os usuários do "meio" (do percentil 25 ao 75) traçando uma fronteira em seu tamanho, punindo dados fora dessa redoma da "normalidade"._

Aferimos o dataset usando o limite "Boxplot" padrão de 1.5x o IQR.

**🔹 O que isso significa para o Negócio?**
Variáveis como `proposed_credit_limit` tem um surto: **24.17%** da base é qualificada estatisticamente como Outlier. A Feature `bank_branch_count_8w` passa disso, chegando a distorções gravíssimas em 17% dos clientes. Estes limites astronômicos farão as Redes Neurais perderem o rumo (Backpropagation instability). Teremos obrigatoriamente que aplicar `MinMaxScaler` ou `RobustScaler` com percentuais _clipados (Clipping)_ rigorosos no desenvolvimento das pipelines de engenharia de dados.

<details>
<summary><b> Tabela Completa de Outliers por Features </b></summary>

| Variavel                       | Outliers | % Outliers | Limite Inferior | Limite Superior |
| ------------------------------ | -------- | ---------- | --------------- | --------------- |
| `proposed_credit_limit`        | 241.742  | 24.17%     | -250.00         | 950.00          |
| `has_other_cards`              | 222.988  | 22.30%     | 0.00            | 0.00            |
| `intended_balcon_amount`       | 222.702  | 22.27%     | -10.43          | 14.23           |
| `bank_branch_count_8w`         | 175.243  | 17.52%     | -35.00          | 61.00           |
| `prev_address_months_count`    | 157.320  | 15.73%     | -20.50          | 31.50           |
| `phone_mobile_valid`           | 110.324  | 11.03%     | 1.00            | 1.00            |
| `days_since_request`           | 94.834   | 9.48%      | -0.02           | 0.06            |
| `session_length_in_minutes`    | 78.789   | 7.88%      | -5.54           | 17.51           |
| `zip_count_4w`                 | 59.871   | 5.99%      | -681.00         | 3519.00         |
| `current_address_months_count` | 41.001   | 4.10%      | -147.50         | 296.50          |
| `device_distinct_emails_8w`    | 31.933   | 3.19%      | 1.00            | 1.00            |
| `foreign_request`              | 25.242   | 2.52%      | 0.00            | 0.00            |

</details>

<br>

## 1.7 Downcasting de Memória Otimizada

O uso de memória de 240MB pode não parecer grande na base bruta, mas quando um pipeline efetua One-Hot Encoding ou SMOTE (Aumento de dimensionalidade gerando fatiamento de arrays no C++ / Cython), geramos sobrecarga no kernel ou até mesmo _Memory Leaks_.

> \*ℹ️ **Glossário Explicativo: Memory Downcasting\***
> _Consiste em ler e analisar as varáveis estáticas nas memórias e perceber se suas propriedades se apoiam perfeitamente num espaço alocado imensamente menor da arquitetura padrão._

A maior parte dos booleanos do projeto não requer vetores do tipo `int64` alocados, e sim `int8`.
O _Footprint_ da base de dados será drasticamente reduzido pelos scripts em Python para que o tuning e simulações com os modelos de Árvore Random Forest ocorram lisos e otimizados pelo hardware durante a Pipeline.

</details>

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
+----------------+ +----------+ +-------------+
| make_dataset   | | EDA      | | train_*     |
| (Data Eng.)    | | Reporter | | _model.py   |
+----------------+ +----------+ +-------------+
    |                  |              |
    v                  v              v
+--------+      +-----------+  +------------+
| data/  |      | reports/  |  | models/    |
|processed|     | figures/  |  | *.pkl      |
| *.csv  |      | *.png     |  | *.txt      |
+--------+      +-----------+  +------------+
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

### 1. Ingestão de Dados

- **Arquivo Responsável:** `make_dataset.py`
- **Entrada:** `data/raw/Base.csv` (Dataset BAF Suite - NeurIPS 2022)
  - É a base bruta contendo features comportamentais de aberturas de conta e o rótulo de fraude (`fraud_bool`). O pipeline todo depende dessa massa de dados para o treinamento.
- **Descrição da Atividade:**
  1. Carrega o CSV bruto.
  2. Aplica _downcasting_ de tipos numéricos (ex: `float64` para `float32`) visando reduzir intensamente o consumo de RAM.
  3. Valida se a coluna target existe.
  4. Separa os dados de Forma Estratificada (80/20) para preservar o percentual de fraude (aprox. 1%).
- **Saída Gerada:** Quatro artefatos isolados para o framework scikit-learn (`data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`).

### 2. Análise Exploratória (EDA)

- **Arquivo Responsável:** `generate_eda_report.py`
- **Entrada:** `data/raw/Base.csv` (Original, sem aplicar limpezas intermediárias).
- **Descrição da Atividade:**
  1. Cria um check-up dos dados calculando estatísticas descritivas básicas.
  2. Otimiza quantificação de Outliers pela regra do IQR.
  3. Aciona o cálculo estatístico de Mann-Whitney U e o agrupamento condicional `Mutual Information` rankeando a capacidade de previsão de cada variável.
  4. Monta as correlações visuais e cria o HTML Sweetviz.
- **Saída Gerada:** Tabelas de suporte em `reports/data/*.csv`, Gráficos PNG das variações (em `reports/figures/eda/`), Relatório textual `reports/eda_summary.txt` e o painel dinâmico `reports/sweetviz_report.html`.

### 4. Treinamento e Otimização

- **Arquivos Responsáveis:** `reg_log_model.py`, `decision_tree_model.py`, `random_forest_model.py`, `xgboost_model.py`, `mlp_model.py`, `isolation_forest_model.py`
- **Entrada:** `data/processed/X_train.csv`, `y_train.csv`.
- **Descrição da Atividade:**
  1. Para cada modelo acionado, cria a `Pipeline` (junta o Transformador/Escalonador com o Classificador).
  2. Executa a _GridSearchCV_ procurando o melhor pacote de Hiperparâmetros.
  3. Consolida ensinando ao modelo a base inteira num retreino _Champion_.
  4. Varre a Curva _Precision-Recall_ para calibrar um threshold (limiar de decisão) otimizado.
- **Saída Gerada:** Os cérebros treinados `models/*_best_model.pkl` prontos para inferência em disco rígido, seus versionamentos para histórico, os textos dos limiares de decisão `*_threshold.txt`, as configs ótimas `*_best_model_params.txt`, e a persistência total dos scores operacionais via `reports/experiments_log.json`.

### 5. Avaliação Final

- **Arquivo Responsável:** `visualize.py`
- **Entrada:** `data/processed/X_test.csv`, `y_test.csv` (Os dados intocados de blind-test). Puxa também os `.pkl` gerados no passo anterior.
- **Descrição da Atividade:**
  A etapa expõe o modelo ao Teste Cego para extrair suas probabilidades e classe predita. Processa as métricas definitivas (ROC-AUC, Precision, Recall, F1) e consolida os gráficos avaliativos.
- **Saída Gerada:** Gráficos cruciais da capacidade em tempo real (`reports/figures/confusion_matrix_*.png`, etc) e uma atualização direta da linha final constando as métricas no log unificado `experiments_log.json`.

### 6. Previsão Discreta (Opcional)

- **Arquivo Responsável:** `predict_model.py` (Orquestrado por `main.py` com a flag `--predict`)
- **Entrada:** Pipeline Final Serializado `models/*_best_model.pkl` do algoritmo validado.
- **Descrição da Atividade:**
  1. Carrega amostras do Dataset de Teste.
  2. Simula o ambiente real e faz predições contendo a probabilidade para as contas não serem legítimas em tempo real.
- **Saída Gerada:** Exibição em modo console interativa para acompanhamento individual (Aprovado, Bloqueado, etc) do pipeline executório.

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
|-- main.py                    # Orquestrador principal (CLI paramétrico: --max-samples, --undersampling-ratio)
|-- requirements.txt           # Dependencias do projeto (incluindo scikit-learn e imbalanced-learn)
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
|   |   |-- build_features.py  # Pipeline EDA-driven (EDAFeatureEngineer + Scaler + OneHot)
|   |
|   |-- models/
|   |   |-- __init__.py

|   |   |-- force_precision.py     # Ajuste fino de threshold por Precision-alvo
|   |   |-- threshold_utils.py     # Utilitarios anti-leakage para Thresholds
|   |   |-- trainers/              # Submodulo de Algoritmos Otimizados
|   |       |-- base_trainer.py        # Classe Abstrata (Orquestracao IO e Otimizacao)
|   |       |-- reg_log_model.py       # Treinamento Logistic Regression
|   |       |-- decision_tree_model.py # Treinamento Decision Tree
|   |       |-- random_forest_model.py # Treinamento Random Forest
|   |       |-- xgboost_model.py       # Treinamento XGBoost
|   |       |-- mlp_model.py           # Treinamento MLP (Rede Neural)
|   |       |-- isolation_forest_model.py # Treinamento Isolation Forest
|   |       |-- lightgbm_model.py      # Treinamento LightGBM (Campeao de F1/Recall)
|   |       |-- stacking_model.py      # Treinamento de Stacking Ensemble
|   |
|   |-- serving/
|   |   |-- __init__.py
|   |   |-- predict_ensemble.py    # Motor de Inferencia MLOps (Comite de Votos)
|   |   |-- simulate_production.py # Simulador CLI de Producao (Streaming Console)
|   |   |-- predict_model.py       # Predicao single-model legado
|
|-- reports/
|   |-- data/                  # CSVs de metricas (qualidade, correlacao, MI, etc.)
|   |-- figures/               # Graficos PNG (EDA, avaliacao, comparacao)
|   |-- eda_summary.txt        # Relatorio textual consolidado da EDA
|   |-- model_comparison_report.txt # Relatorio do benchmark
|   |-- experiments_log.json   # Historico unificado de todos os experimentos
|   |-- sweetviz_report.html   # Dashboard interativo HTML
|   |-- simulation_summary.txt # Relatorio executivo financeiro (ROI) do Ensemble
|
|-- tests/                     # Suite de testes unitarios funcionais (Data e Features)
|   |-- test_data.py           # Testes automatizados do build features
|   |-- test_features.py       # Testes da preparacao otimizada
|
|-- venvmine/                  # Ambiente virtual Python (nao versionado)
```

## 3.1 Descricao Detalhada de Cada Arquivo

### main.py - Orquestrador Principal

| Atributo         | Descricao                                                           |
| ---------------- | ------------------------------------------------------------------- |
| **Funcao**       | Orquestra todo o pipeline na ordem correta via CLI (argparse)       |
| **Funcoes**      | `reset_project_artifacts()`, `main()`                               |
| **Entradas**     | Argumentos CLI: `--no-reset`, `--skip-eda`, `--predict`, `--models` |
| **Saidas**       | Execucao sequencial de todos os modulos                             |
| **Dependencias** | Todos os modulos em `src/`                                          |

Flags disponiveis:

| Flag         | Efeito                              |
| ------------ | ----------------------------------- |
| `--no-reset` | Pula a limpeza de artefatos antigos |
| `--skip-eda` | Pula a analise exploratoria         |

| `--predict` | Roda simulacao de inferencia ao final |
| `--models` | Seleciona modelos especificos (ex: `--models xgb,rf`) |

Identificadores de modelos: `logreg`, `dt`, `rf`, `xgb`, `mlp`, `if`.

### src/data/make_dataset.py - Engenharia de Dados

| Atributo     | Descricao                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------- |
| **Funcoes**  | `optimize_memory_usage(df)`, `load_and_split_data(max_samples=None)`                        |
| **Entrada**  | `data/raw/Base.csv`                                                                         |
| **Saida**    | `data/processed/{X_train, X_test, y_train, y_test}.csv`                                     |
| **Excecoes** | `FileNotFoundError` se Base.csv nao existir; `ValueError` se coluna alvo nao for encontrada |

Fluxo interno:

1. Carrega CSV bruto com `pd.read_csv`
2. Aplica truncate limitador da base geral via `df.sample()` (se `--max-samples` ativado em DEV)
3. Valida existencia da coluna target (fallback para `is_fraud`)
4. Separa features (X) e target (y)
5. Executa `train_test_split` com `stratify=y` para manter proporcao de fraude em Teste Cego
6. Executa Otimização de downcasting em chunks descolados de tempo real
7. Salva 4 PKLs processados otimizados

### src/features/build_features.py - Engenharia de Features e Pré-processamento (EDA-Driven)

| Atributo    | Descricao                                                                                |
| ----------- | ---------------------------------------------------------------------------------------- |
| **Classe**  | `EDAFeatureEngineer(BaseEstimator, TransformerMixin)` -- Transformer sklearn customizado |
| **Funcoes** | `get_preprocessor(X)`, `build_pipeline(X_train, model)`, `process_features()`            |
| **Entrada** | DataFrame X com features brutas                                                          |
| **Saida**   | `Pipeline` completo de 3 etapas (EDAFeatureEngineer -> ColumnTransformer -> Modelo)      |

O pipeline foi reestruturado com base nos insights da Analise Exploratoria (EDA) e agora possui 3 camadas:

**Camada 1 - EDAFeatureEngineer** (Transformer Customizado):

A inteligência de negócios do EDA foi rigidamente codificada matematicamente neste Transformer. Ele executa em série:

| Transformação Orgânica               | Fundamentação Acadêmica e Impacto do Algoritmo                                                                                                                                                                   | Defesa do Método Prático (Baseado no EDA)                                                                                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Poda de Sinais Mortos (Drip-out)** | Algoritmos com dimensionalidade ociosa sofrem com o _Cárcere da Dimensionalidade_ e lentidão matricial.                                                                                                          | Remoção sumária de `device_fraud_count` (Variância Zero detectada) e `session_length_in_minutes` (Refutado via _Mann-Whitney_ com P-Value insignificante).                                      |
| **Imputação de Sentinelas (NaNs)**   | Modelos lineares não percebem o valor `-1` como uma "falha no banco de dados", e sim como um "déficit financeiro". Extraímos o -1, convertemos a métrica em Flags Booleanas isoladas e o imputamos para **NaN**. | A Mediana matemática deixava de ser puxada violentamente para baixo em `prev_address_months_count` (onde +50% dos dados estavam nulos e preenchidos equivocadamente por `-1`).                  |
| **Clipping Dinâmico Estrito**        | O _Exploding Gradient_ destrói as derivadas parciais de Redes Neurais quando expostas a vetores estourados infinitos.                                                                                            | Aplica o clipping (limitação) implacável nos percentis de 1% e 99% em variáveis com surto superior a >15% de Outliers (Pelo desvio IQR), cimentando segurança estatística no Limite de Crédito. |
| **Codificação Discreta de Risco**    | Ao invés de dependermos unicamente do `OneHotEncoder` fragmentar a malha dimensional e criar features inexpressivas, alavancamos nós puros da Árvore de Decisão antecipadamente.                                 | Tags identificadas com _Prior Fraud Rate_ altíssimo na EDA (Ex: Uso de Windows, Moradia classe 'BA') recebem rotulagem prioritária discreta no Topo.                                            |
| **Extração de Fatores Ocultos**      | Captura sinergias ocultas cruzando vetores que individualmente apresentavam baixa ameaça linear, gerando features densas de _Mutual Information (MI)_.                                                           | Células novas criadas baseadas em Fraude Comportamental (Ex: Combos de Uso simultâneo de E-mails gratuitos com troca veloz de Device OS).                                                       |

**Camada 2 - ColumnTransformer** (Preprocessamento Paramétrico e Imunização):

- **Pipeline Numérico:** `SimpleImputer(median)` -> `RobustScaler()`.
  > _Por que não StandardScaler?_ O uso de Z-Score da escala Média Clássica seria despedaçado pelas aproximações do Limite propostos dos cartões de crédito sujos (>24% outliers). O Scaling via Mediana + IQR (_RobustScaler_) é imune estruturalmente a isso.
- **Pipeline Categórico:** `SimpleImputer(constant='missing')` -> `OneHotEncoder(handle_unknown='ignore')`.
  > A supressão de crashes de produção: caso uma nova categoria surja em produção que o modelo não conheça, os coeficientes irão silenciá-la evitando pânicos no Back-end.

**Camada 3 - O Hibridismo `imblearn` (Mitigação do Desbalanceamento Agressivo):**

- (Se ativado via parâmetro global) Instancia `imblearn.pipeline.Pipeline`.
- Frita amostras em proporções `Ratio` exclusivamente durante os _Folds_ estritos do GridSearchCV no X_train, blindando as métricas de validação real mantendo seus datasets intocados.

**Camada 4 - Modelo** (classificador):

- O classificador especifico de cada modelo instanciado que obedece todo o histórico anterior (LogReg, XGBoost, RF, etc.)

Decisao tecnica: O `EDAFeatureEngineer` e um `BaseEstimator` do scikit-learn, sendo serializado junto com o modelo via `joblib.dump()`. Isso garante que as mesmas transformacoes sejam aplicadas automaticamente em treino, validacao cruzada e inferencia.

### src/models/trainers/reg_log_model.py - Logistic Regression

| Atributo                           | Descricao                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Funcao principal**               | `train_logistic_regression()`                                                                    |
| **Estrategia de desbalanceamento** | `class_weight='balanced'` (sem SMOTE)                                                            |
| **Grid Search**                    | `C`: [0.01, 0.1, 1, 10]; `penalty`: ['l1', 'l2']                                                 |
| **Otimizacao**                     | Amostra estratificada de 100k linhas para GridSearch; retreino final com dataset completo        |
| **Saidas**                         | `logreg_best_model.pkl`, `logreg_threshold.txt`, `best_model_params.txt`, `experiments_log.json` |

### src/models/trainers/decision_tree_model.py - Decision Tree

| Atributo             | Descricao                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_decision_tree()`                                                                   |
| **Grid Search**      | `max_depth`: [5, 10, None]; `min_samples_split`: [2, 5]; `criterion`: ['gini', 'entropy'] |
| **Saidas**           | `dt_best_model.pkl`, `dt_threshold.txt`, `dt_best_model_params.txt`                       |

### src/models/trainers/random_forest_model.py - Random Forest

| Atributo             | Descricao                                                                            |
| -------------------- | ------------------------------------------------------------------------------------ |
| **Funcao principal** | `train_random_forest()`                                                              |
| **Grid Search**      | `n_estimators`: [100, 200]; `max_depth`: [10, 20, None]; `min_samples_split`: [2, 5] |
| **Nota**             | RF usa `n_jobs=-1` internamente; GridSearch usa `n_jobs=1` para evitar conflito      |
| **Saidas**           | `rf_best_model.pkl`, `rf_threshold.txt`, `rf_best_model_params.txt`                  |

### src/models/trainers/xgboost_model.py - XGBoost

| Atributo             | Descricao                                                                     |
| -------------------- | ----------------------------------------------------------------------------- |
| **Funcao principal** | `train_xgboost()`                                                             |
| **Estrategia**       | `scale_pos_weight=90` para compensar desbalanceamento                         |
| **Grid Search**      | `learning_rate`: [0.01, 0.1]; `n_estimators`: [100, 200]; `max_depth`: [3, 6] |
| **Otimizacao**       | Amostra estratificada de 100k para GridSearch; retreino no dataset completo   |
| **Saidas**           | `xgb_best_model.pkl`, `xgb_threshold.txt`, `xgb_best_model_params.txt`        |

### src/models/trainers/mlp_model.py - MLP Neural Network

| Atributo             | Descricao                                                                                                                                          |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_mlp()`                                                                                                                                      |
| **Grid Search**      | `hidden_layer_sizes`: [(50,), (100,), (50,25)]; `activation`: ['relu','tanh']; `alpha`: [0.0001, 0.001, 0.01]; `learning_rate_init`: [0.001, 0.01] |
| **Nota**             | Usa `early_stopping=True` com 10% de validacao interna                                                                                             |
| **Saidas**           | `mlp_best_model.pkl`, `mlp_threshold.txt`                                                                                                          |

### src/models/trainers/isolation_forest_model.py -- Isolation Forest

| Atributo             | Descricao                                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_isolation_forest()`                                                                                      |
| **Classe auxiliar**  | `IForestWrapper(BaseEstimator, ClassifierMixin)` -- Wrapper que converte `decision_function` em `predict_proba` |
| **Nota**             | Algoritmo nao-supervisionado adaptado para pipeline supervisionado. Sem GridSearch.                             |
| **Parametros fixos** | `n_estimators=200`, `contamination=0.01`                                                                        |
| **Saidas**           | `if_best_model.pkl`, `if_threshold.txt`                                                                         |

A classe `IForestWrapper` inverte o score de anomalia (`-decision_function`), normaliza com `MinMaxScaler` para [0,1] e empacota no formato `predict_proba` padrao `(n_samples, 2)`.

### src/serving/simulate_production.py -- Simulacao de Producao

| Atributo                     | Descricao                                                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| **Funcoes**                  | `load_inference_artifacts()`, `load_threshold()`, `explain_prediction()`, `predict_sample()` |
| **Motor de decisao**         | Votos Maioria >= 2 -> BLOQUEIO; Voto Isolado MLP -> REVISAO MANUAL; Senao -> APROVADO        |
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
[simulate_production] (se --predict ativo)
    Inicia Comite Ensemble (predict_ensemble.py)
    Le os modelos XGB, MLP, LGBM e seus thresholds
    Sorteia batch de transacoes do X_test (embaralhado)
    Aplica motor Voting com Veto Financeiro
    Imprime feed visual da previsao no console
    Computa e Salva simulation_summary.txt (Relatorio em $)
```

## 4.2 Fluxo de Treinamento de Modelo (Generico)

```
train_*():
    [BaseTrainer - Classe Abstrata Orquestradora]
    1. run_id = timestamp atual
    2. _load_data(): Carrega X_train e y_train do PKL e ravel().
    3. pipeline = build_pipeline(X_train, clf)
    4. _get_sample(): Amostragem rigorosa (`stratify=y`) para Busca Rápida de Hiperparâmetros.
    5. compute_sample_weight(): Aplica balanceamento de Pesos Cost-Sensitive para suprir Desbalanceamento.
    6. GridSearchCV / RandomizedSearchCV configurado conforme `config.py` injetado pelo modelo.
    7. Retreina melhor estimador em Dataset Completo.
    8. threshold_utils.compute_optimal_threshold() -> argmax(F1).
    9. joblib.dump do Modelo Campeão e Versionado `model_*_{run_id}.pkl`.
   10. Appenda log de métricas da validação cruzada no `experiments_log.json`.
```

## 4.3 Fluxo de Inferencia (Predicao com Ensemble)

```
simulate_production() -> predict_ensemble() :
    1. predictor = FraudEnsemblePredictor() carrega modelos .pkl e threshold.txt (ex: LightGBM, XGBoost, MLP)
    2. X_test, y_test são carregados e embaralhados.
    3. Para cada transacao fornecida em streaming:
       a. Calcula proba para os 3 modelos e computa contra os seus 3 thresholds otimizados.
       b. Majority Vote Logic (Smart Ensemble c/ Veto Especial):
          - Se Fraud_Votes >= Majority_Threshold (ex: 2/3): BLOQUEIO (Alta Confiança)
          - Se Fraud_Votes > 0 E Votante Unico == 'mlp': REVISÃO MANUAL (Veto de Precisão)
          - Senao: APROVADO (Nivel de Risco Assumido)
    4. Atualiza os TPs, FPs, TNs e FNs em Real Time.
    5. Imprime resultado consolidado visual Terminal CLI.
    6. Quando finalizado, computa ROI e Atrito -> reports/simulation_summary.txt
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
- Algoritmos dependem de balanceador semântico estatístico e pesos (`class_weight`, `scale_pos_weight`) aplicados a nível de CV Fold.
- GridSearch avalia por `roc_auc` (metrica independente de threshold)

## 6.4 Regras de Versionamento de Modelos

- Modelo "latest" em `{nome}_best_model.pkl` (sempre sobrescrito)
- Modelo historico em `model_{nome}_{timestamp}.pkl` (nunca sobrescrito)
- Pasta de modelos nao e limpa pelo reset para manter historico

## 6.5 Regras de Motor de Decisao

```
Se (Votos de Fraude do Comitê) >= Maioria (2+ de 3): --> BLOQUEIO AUTOMATICO (Alto Risco)
Se (Votos = 1) E (Voto de Fraude originado pela MLP): --> REVISAO MANUAL (Médio/Alto Risco - Veto de Campeão)
Outros Casos (Voto 0 ou voto único de modelo fraco):  --> APROVADO (Baixo Risco Aceitável)
```

---

# 7. Logica e Algoritmos

## 7.1 Otimizacao de Memoria (Downcasting)

O algoritmo em `optimize_memory_usage()` itera por todas as colunas e verifica se o range de valores cabe em tipos menores (`int8`, `int16`, `float32`), reduzindo significativamente o footprint de memoria para datasets com milhoes de linhas.

## 7.2 Threshold Tuning

Todos os modelos executam apos o treinamento:

1. Calculam `predict_proba` no conjunto de treino
2. Geram a curva Precision-Recall com todos os thresholds possiveis
3. Calculam F1 = 2*(P*R)/(P+R) para cada threshold
4. Selecionam o threshold que maximiza F1

Isso substitui o corte ingenuo de 0.5 por um ponto de operacao otimizado para o problema.

## 7.3 IForestWrapper (Adapter Pattern)

O Isolation Forest nao implementa `predict_proba` nativamente. O `IForestWrapper` aplica o padrao Adapter:

1. `fit()`: Treina o IF e ajusta um `MinMaxScaler` nos scores de decisao invertidos
2. `predict()`: Converte -1 (anomalia) para 1 (fraude) e 1 (normal) para 0
3. `predict_proba()`: Normaliza scores para [0,1] e retorna formato `(n, 2)`

## 7.4 Amostragem Estratificada para GridSearch

LogReg e XGBoost usam amostra de 100k linhas para GridSearch (economia de horas de processamento), seguida de retreino no dataset completo com os parametros vencedores.

## 7.5 Informacao Mutua (MI)

A EDA calcula Mutual Information com `mutual_info_classif` para ranquear features por capacidade preditiva, capturando relacoes nao-lineares que correlacao de Pearson/Spearman ignora.

## 7.6 EDAFeatureEngineer e o Hibridismo imblearn (Pré-processamento e Engenharia de Features Orientados por Dados)

A essência matemática do nosso tratamento é consolidada numa Orquestração Direcionada baseada em insights acadêmicos da nossa Análise Exploratória. O `EDAFeatureEngineer` e um transformer customizado do scikit-learn que imobiliza o "Efeito Borboleta" da base suja e protege a Inteligência Artificial:

```text
       Dados Brutos Massivos (31 features)
                     |
                     v
 1. Expulsão de Ruído Branco e Inutilidade Geométrica
    [-2 features] Variância estrita a 0 (device_fraud_count constante ignorada).
                     |
                     v
 2. Transmutação de Marcadores Bancários Falsos (Imputação de "Sentinelas")
    Detectado "-1" fingindo ser saldo ou meses na conta. O modelo transforma
    estas lacunas de sistema bancário antigo em marcações de Ausência Estrutural
    booleanas fortes, e cede vazios (NaN) para a Redes Neurais interpolarem na Mediana real.
                     |
                     v
 3. Castração Gradiente Extremista (Winsorizing/Clipping)
    Redes Neurais (MLP) falham mediante valores absurdos (Exploding Gradient).
    Em colunas com +15% de Outliers nativos confirmados via Range Inter-Quartil (IQR),
    aplica teto inflexível nos limites do espectro de 1% a 99%.
                     |
                     v
 4. Evidência Discreta Condicional (Destaque Arbóreo para Árvores Cegas)
    [+5 features booleanas demarcando nichos ultra-fraude no alvo categórico cru].
                     |
                     v
 5. Informação Mútua (Feature Cross-Interaction Digital)
    O Modelo cria interações de peso exponencial simulando o ecossistema mental da fraude.
    (+ O Fraudadador automatiza via Email_Comercial_Free iterando em vários O.S.)
                     |
                     v
       Dados Puros Envernizados (38 eixos matematicos limpos)
    |
    v
6. Opcional (Se Ratio Ativado): Injeção do RandomUnderSampler na Pipeline hibrida do imblearn.
   (Apenas em tempo de ajuste de Validação Cruzada, jamais em Inferência)
```

O transformer implementa `fit()` para aprender limites de clipping no conjunto de treino e `transform()` para aplicar todas as transformacoes. Por ser um `BaseEstimator`, e automaticamente serializado junto com o modelo.

## 7.7 Abstração de Treinamento (`BaseTrainer`)

Refatora o problema de duplicação que existia na versão primária, gerando Orientação a Objetos. A classe orquestra subamostragem `_get_sample()` rigorosa estratificada, injeção de dependências de modelos arbitrária, resolve união entre `GridSearchCV` e `RandomizedSearchCV`, define pesos usando `compute_sample_weight` e automatiza a trilha de Logs (`experiments_log.json`).

## 7.8 Motor de Simulação (Ensemble PoV & ROI)

Implementado com vetores `pandas` de altíssima velocidade, prevê dados em "Batch Array" diretamente pelos 3 Modelos Comitê otimizados simultaneamente (MLP, Xgb, LightGBM). Utiliza CLI (Command Line Interface) Visual para reportar:

- O Veredito de Maioria Simples ou Veto Especial, traduzidos em True Positives e False Negatives.
- Uma Inteligência de Negócio Financeira (`simulate_production.py`) cruza os erros contra um Ticket Médio de negócio predefinido e entrega ao usuário e Stakeholders um Documento final (`simulation_summary.txt`) contendo Patrimônio Salvo, Custo de Atrito, Taxa da Operação e o Lucro Resgatado em Dinheiro Requerido.

---

# 8. Configuracoes e Variaveis de Ambiente

| Variavel       | Arquivo             | Finalidade                     | Valor        |
| -------------- | ------------------- | ------------------------------ | ------------ |
| `RANDOM_STATE` | `config.py`         | Semente para reprodutibilidade | 42           |
| `TEST_SIZE`    | `config.py`         | Proporcao do split de teste    | 0.2          |
| `TARGET_COL`   | `config.py`         | Coluna alvo                    | `fraud_bool` |
| `SAMPLE_SIZE`  | Modelos individuais | Amostra para GridSearch        | 100000       |
| `CV_FOLDS`     | Modelos individuais | Folds de validacao cruzada     | 3            |
| `.env`         | Raiz                | Reservado (vazio)              | --           |

---

# 9. Estrategia de Logs e Monitoramento

## 9.1 Logs em Console

Todos os modulos utilizam `logging.basicConfig` com nivel `INFO` e saida para `stdout`. O formato padrao e `%(asctime)s - %(levelname)s - %(message)s`.

## 9.2 O Funcionamento do Treinamento e do Log Persistido (`experiments_log.json`)

Uma curiosidade comum sobre a orquestração do projeto é como a miríade de parâmetros é testada e de que forma o arquivo preserva essa memória.

O orquestrador central do sistema (`src/models/trainers/base_trainer.py`) efetua essa operação dividida em três etapas vitais, concebidas para garantir estabilidade e evitar que o repositório de logs infle com combinações mortas e ineficientes.

**1. A Competição de Parâmetros (A fase interna de Busca em Memória)**
Os algoritmos do sistema nunca recebem apenas um número seco (Ex: `learning_rate = 0.1`). Eles absorvem do `config.py` uma verdadeira "malha espacial" de combinações. Submetemos estas combinações ao `GridSearchCV` (ou `RandomizedSearchCV`). Internamente (em poder de CPU e alocação de RAM), a máquina testa as dezenas e dezenas de arranjos distintos. Efetua essa submissão particionando amostras de dados na validação cruzada (`CV Folds`), buscando a arquitetura mais equilibrada.

**2. A Coroação da Máquina, e o Descarte (Identificação do Campeão)**
Muitos sistemas cometem o erro de logar absolutamente tudo gerando ruído. Nossa arquitetura, ao finalizar o processo da Busca em Grade anterior, avalia o desempenho de todas as topologias pela ótica da métrica de blindagem assíncrona `roc_auc`. Findada a varredura, o sistema **destrói silenciosamente** os piores e os médios desempenhos. Identificado o "Hyperparâmetro Campeão" (`best_params_`), cria-se um cérebro virgem utilizando-o, **retreinando o modelo do zero na base de dados inteira e total do sistema** para fixar a aprendizagem em escalabilidade. Apenas este cérebro é serializado para virar o `.pkl` no disco.

**3. Injeção Unificada no Banco de Logs (`experiments_log.json`)**
Para proteger a leitura clara e a avaliação sintética do Analista e dos Stakeholders, o script aciona a interface `log_experiment()`. Ela funciona injetando exclusivamente as condições de vitória do modelo. Apenas a arquitetura eleita grava suas métricas, sendo populado:

- O **`run_id`** constando o timestamp do treinamento.
- O **`model_type`** (Da qual família herda).
- A **`best_params`** e **`best_threshold`**: O DNA exato gerado, e sua casa decimal milimétrica exigida pra maximizar as capturas de bandidos contra Falsos Positivos.
- As **Métricas Finalizadas**: Como os vetores do blind-test do script `visualize.py` complementando as taxas fixas da matriz de transição (Recall Testado, F1 Testado e Área Real ROC testada).

## 9.3 Diagnostico de Problemas

1. **Modelo nao encontrado**: Verificar se `main.py` foi executado antes de `simulate_production.py`
2. **Memoria insuficiente**: Reduzir `SAMPLE_SIZE` nos modelos ou usar `--models` para treinar menos modelos
3. **Threshold nao encontrado**: O sistema faz fallback para 0.5 com warning

---

# 10. Teoria Tecnica Envolvida

Esta seção é dedicada a aprofundar academicamente e cientificamente as decisões arquiteturais, matemáticas e de modelagem tomadas durante a construção do **Fraud Sentinel**. A engenharia de um pipeline de detecção de fraudes em _onboarding_ (Application Fraud) requer muito mais do que empilhar bibliotecas de código; exige uma defesa teórica sólida para lidar com o ruído massivo e a raridade implícita do ataque.

## 10.1 Padrões de Projeto (Engenharia de Software em MLOps)

A base de código transcende _scripts_ soltos, abraçando **Object-Oriented Programming (OOP)** e padrões consolidados da Engenharia de Software clássica aplicados ao Aprendizado de Máquina:

| Padrão Aplicado             | Como e Por Que Foi Utilizado                                                                                                                                                                                                                                                                                                                                                                          |
| :-------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Facade (Fachada)**        | O orquestrador `BaseTrainer` simplifica brutalmente a subamostragem, a injeção de dependências e a aplicação de _Cost-Sensitive Learning_ por debaixo dos panos. O usuário final só interage com uma fachada simples (`train_xgboost()`), enquanto o subsistema complexo resolve a otimização de matriz e as salvaguardas de IO.                                                                      |
| **Adapter (Adaptador)**     | Utilizado ativamente no `IForestWrapper`. O algoritmo não-supervisionado _Isolation Forest_ do _scikit-learn_ não retorna probabilidades (não possui a interface `predict_proba`). O adaptador contorna essa lacuna, capturando o `decision_function`, invertendo seus sinais e o espremendo via `MinMaxScaler(0,1)`, compatibilizando a anomalia cega com o resto da arquitetura estrita do sistema. |
| **Pipeline (Encadeamento)** | Imunização arquitetural contra o **Data Leakage** (vazamento de dados estatísticos do futuro para o fold de treino). A construção usa `sklearn.pipeline.Pipeline`, embutindo limpezas complexas (IQR Clipping) dentro do Cross-Validation, garantindo que nenhum modelo veja as métricas da base de teste.                                                                                            |
| **Strategy (Estratégia)**   | Cada classe no diretório `trainers/` obedece a um contrato comum: inicializa um dicionário de Grid, encapsula sua classe nativa (ex: `XGBClassifier`) e lança na rotina mestra.                                                                                                                                                                                                                       |

---

## 10.2 A Teoria dos Modelos: Por que Foram Selecionados e Como Operam

O _Fraud Sentinel_ não aposta "todas as fichas" em um cálculo só. Treinamos um ecossistema variado de algoritmos, apostando que as fraquezas geométricas de um modelo sejam compensadas pela capacidade matemática de outro.

### 1. Regressão Logística (Logistic Regression)

- **A Matemática:** Um modelo paramétrico linear que calcula o log-odds (logaritmo das chances) da combinação das features e as espreme em uma curva sigmoide entre probabilidade 0 e 1.
- **Papel no Projeto:** Atua como a **Linha de Base Segura (Baseline)**. Por ser rigorosamente convexa e linear, se uma variável isolada (ex: `income`) possui forte sinal monótono de fraude, a LogReg detectará de forma estável. No entanto, ela falha miseravelmente em interações não-lineares (ex: quando `idade > 30` E `OS == 'Windows'` indicam fraude, mas isoladamente não sinalizam nada).

### 2. Árvores de Decisão & Random Forest (Ensambles de Bagging)

- **A Matemática:** Árvores cortam o espaço matemático das features em blocos retangulares (splits) maximizando a "Pureza" (Gini Impurity) e os Ganhos de Informação. A _Random Forest_ gera centenas de Árvores profundas de forma aleatorizada e amarra suas previsões via votação (_Bagging_), destruindo a altíssima variância característica de uma única árvore isolada.
- **Papel no Projeto:** Robustez bruta e identificação de não-linearidades. O Random Forest raramente sofre _overfitting_ crasso e mapeia nichos ("bolhas" específicas de usuários do sistema). Seu principal revés neste contexto é o alto peso (Modelos pesados, consumindo centenas de MBs) e a dificuldade orgânica em lidar com fatias minúsculas (1%) de fraude, já que as amostragens aleatórias com frequência geram árvores "cegas" onde nenhuma fraude existiu no subset.

### 3. Máquinas de Vetores de Gradiente (Gradient Boosting - XGBoost e LightGBM)

- **A Matemática:** Ao invés de criar árvores em paralelo como na floresta aleatória, o _Boosting_ constrói as árvores estritamente em série. A Árvore B é otimizada sob o resíduo contínuo (o "erro de gradiente") originado pela Árvore A.
- **Papel no Projeto:** Constituem o **coração preditivo de precisão**. O **LightGBM** é um campeão de particionamento veloz (processamento via histogramas – Otimização por GOSS - Gradient-based One-Side Sampling), focando sua CPU de forma implacável exatamente onde ele mais errou anteriormente (_Hard Examples_). O **XGBoost** compensa com penalizações elásticas (L1/L2 estritos nos pesos da folha) e atua perfeitamente capturando o limiar de fraude invisível aos algoritmos lineares. São eles a força motriz do "Veto Especial de Fraude" na Arquitetura do Comitê de Ensemble.

### 4. Redes Neurais Profundas (Multilayer Perceptron - MLP)

- **A Matemática:** Inspirado fisiologicamente por neurônios abstratos. Aplica centenas de pesos e vieses através de operações matriciais consecutivas, retropropagando (Backpropagation) os erros nas derivadas parciais.
- **Papel no Projeto:** Serve para capturar Padrões Topológicos de Altíssima Dimensão. Como nosso dataset é tabular e possui categorias densas, uma Rede Neural simples apanha consideravelmente do Gradient Boosting (O fenômeno da não-localidade dos dados categóricos contra a força bruta do aprendizado por partição das árvores). Contudo, a MLP no Comitê traz ganhos de **Sensibilidade (Recall)** genérica, já que suas conexões contínuas costumam não sofrer do viés agudo de bordos rígidos da decisão que assombra a Árvore de Decisão convencional.

### 5. Isolation Forest (Não-Supervisionado / Geométrico)

- **A Matemática:** A anomalia é "mais fácil de isolar" na floresta. Quantos nós (splits matemáticos) precisaram ser feitos até isolar aquele usuário em uma única folha? Se ele for "Normal", ele cairá no meio do emaranhado junto da maioria (demorando vários splits). Se ele for Anômalo, em apenas dois cortes da árvore aleatória ele ficará isolado no espaço vazio do vetor.
- **Papel no Projeto:** Usado para provar que a Fraude de Identidade Sintética (Onboarding) não é apenas um outlier geométrico matemático. Criminosos profissionais tentam mimetizar ao máximo o comportamento natural de uma conta corrente para despistar modelos de IA, caindo na média. Portanto, rodar "Análise de Outliers" isoladamente é insuficiente contra fraudes modernas.

---

## 10.3 O Enfrentamento do Desbalanceamento Extremo Sistemático

Lidamos com uma prevalência hostil de apenas `1.10%` da classe Fraude (Classe 1). Em problemas preditivos, IAs sempre otimizam preguiçosamente convergindo para chutar a Classe Majoritária (aprovando todo mundo) alcançando ~99% de Falsa Acurácia.
Enfrentamos de duas formas cruzadas:

1. **A Queda das Resampling (Decadência do SMOTE)**
   O projeto experimentou criar fraude falsa via _SMOTE (Synthetic Minority Over-sampling Technique)_. A matemática por trás espalha dados artificiais utilizando KNN vizinhos ao redor das fraudes. Contudo, percebemos por testes práticos de laboratório o efeito **Fábrica de Fantasmas**: O SMOTE gerava "fraudes redondas e ideais", causando _Over-Confidence_ artificial nas Árvores de Classificação ao validarem em dados de produção, produzindo falsos alarmes generalizados nos clientes bons. Desligamos a técnica.
2. **Implementação de Cost-Sensitive Learning (Pesos Heurísticos)**
   Ativamos o método analítico: Nenhuma linha tabular é artificialmente adicionada ou excluída. Em contrapartida, ensinamos para a Máquina uma matriz punitiva de dor. Na função perda _(Loss Function/LogLoss)_ do nosso XGBoost, o erro em classificar um cidadão bom como Fraude causa à derivada penalidade padrão `1.0`. Todavia, classificar o Fraudador como Cliente Bom inflige penalidade severa multiplicada (`scale_pos_weight = ~90`). Esse arranjo inclina brutalmente as forças da Descida de Gradiente (Gradient Descent), forçando a IA estritamente à cautela rigorosa buscando o menor sinal possível da classe escassa para evitar sua penalização astronômica.
3. **Random Undersampling Robusto (A Biblioteca imbalanced-learn)**
   Para além da matriz de custos, foi acoplada ao sistema a capacidade de parametrizar o rebalanceamento dinâmico estrito de classes com perda induzida de dados normais via **CLI (`--undersampling-ratio`)**. O motor orquestrador substitui a "Pipeline Clássica" pela _ImbPipeline_ (`imbalanced-learn`). Matemeticamente isso garante a blindagem perfeita contra o **Data Leakage na Validação**. A subamostragem (ex: 10 mil fraudes e 20 mil classes sadias) opera sua limpeza _apenas_ e _estritamente_ nos Folds de Treino (Training Splits). O sub-pacote de Teste local que afere as métricas permanece hiper-desbalanceado como no Mundo Real, forçando o motor de busca (GridSearchCV) a validar a verdade inalterada sem ilusões de um ambiente equalitário falsificado.

---

## 10.4 Threshold Tuning vs Corte Euclidiano Clássico

O Default clássico da inteligência de código aberto dita o limiar estático: A IA afirma que há fraude se _P(Fraude) >= 0.5 (50%)_.
A teoria de ML Operations deste nosso sistema renega isso agressivamente: Diferentes IAs com funções de custo escaladas absurdamente (como a de ~90 do passo anterior) costumam distorcer brutalmente suas saídas probabilísticas numéricas puras (_Proba Calibration_ distorcido e natural das _Decision Trees_).

**Arquitetura de Limiar Computado:** Realizamos a rotina forçada buscando do índice _`0.001`_ ao `0.999`. Compilamos internamente uma malha calculando os Falsos Positivos e Verdadeiros Positivos de forma simultânea. Selecionamos dinamicamente (no pacote `threshold_utils.py`) a casa decimal (Geralmente orbitando entre as probabilidades baixíssimas de `0.06` à `0.15` e não os ideais _0.5_) que representa o pico extremo da montanha do **Score F1** — harmonicamente mediando o limite exato onde a precisão de prender bandidos decai menos face à captura total pretendida do Recall. Este número decimal é extraído e arquivado estático no `.txt` em disco lógico a ser consumido compulsoriamente na simulação online pela interface do Sistema do Banco.

---

## 10.5 O Porquê da Estatística não-Paramétrica Adotada na EDA

Nenhuma feature importante (Ex: Idades ou Velocidade do Onboarding 24h) obedeceu ao Sino de Gauss Puro (Teste padrão de Normalidade / Skewness). Dessa forma, a validação não podia ser básica:

1. **Mann-Whitney U Test ao longo do T-Test Clássico das Universidades:** Para garantir que a diferença das métricas comportamentais entre Clientes Normais e Fraudadores era puramente verídica (não um falso acaso), optamos pelo Teste `U` embasado no _Ranking Ordenado de Posição dos Índices_ e não por "Comparações Falsas de Médias Cartesianas", validando se as amostras pertencem à mesma estirpe populacional biológica ou se divergem fortemente pelo espectro P-Value inferirior a `0.05`.
2. **Mutual Information (MI) como Norteador Superior de Pearson/Spearman:** Correlações famosas lineares (A matriz `Pearson`) só enxergam retas unindo X a Y. Muitas de nossas descobertas operavam por "Ganho de Entropia" (Clusters disjuntos sem lógica aparente linear, mas categorizados em frações de nichos perigosos de forma quadrática – A exatidão da detecção explodindo quando o cruzamento de e-mails ocorria sob um determinado número de Telefones). O cálculo do MI estipulando sua fórmula embasada em _Redução da Incerteza_ _(Entropy H(x))_ do target validado pelo ganho condicional embasou com blindagem acadêmica todas as decisões da classe matemática orgânica embutida e serializada em nosso passo estrito da Pipeline (`EDAFeatureEngineer`).

---

# 11. Melhorias Futuras

## 11.1 Melhorias de Performance

1. **Substituir GridSearchCV por Optuna/BayesSearchCV** para busca ainda mais robusta de hiperparametros em hipercubos densos.
2. **Implementar cache de preprocessamento avançado** para evitar recomputacao entre diferentes partições no motor de CV do `BaseTrainer`.
3. **Usar Parquet em vez de Pickle** caso ocorra imprecisão massiva e gargalo nas escritas do disco por datasets multibilionários.

## 11.2 Refatoracoes Recomendadas

1. **Extrair logica de persistencia de experimentos** para um container online (`MLflow` tracking).
2. **Centralizar configuracao de modelos** em um unico YAML/JSON para integração direta a pipelines de Deploy CI/CD em Nuvem (Airflow, KubeFlow).
3. **Implementar Explainable AI (SHAP/LIME)** diretamente conectada na resposta do Comitê de Decisão (`predict_ensemble.py`) informando qual feature barrou ou aprovou o risco.

---

# 12. Análise Crítica e Explicação dos Experimentos

O arquivo `reports/experiments_log.json` consolida os resultados quantitativos de cada ciclo de modelagem. Embora o número da Acurácia de quase todos os testes seja de 98% a 99%, essa métrica é uma ilusão derivada da assimetria extrema do dataset (apenas 1.1% de sub-representação). Em sistemas de fraude impera a necessidade de balizar o trade-off entre bloquear fraudadores e aprovar clientes bons.

Para compreender os resultados, é imperativo o domínio dos seguintes conceitos macro-avaliativos:

| Métrica                              | Significado no Contexto de Fraude                                                                                                                                        | Impacto de Negócio                                                                                     |
| :----------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| **Recall** _(Sensibilidade)_         | Mede a proporção de fraudes reais que o sistema conseguiu detectar. Um Recall de 80% significa que de cada 100 fraudes, pegamos 80 e deixamos passar 20.                 | Maximizar o Recall é o **objetivo de segurança** da instituição financeira.                            |
| **Precision** _(Precisão)_           | Avalia a proporção de acertos quando o sistema "apita" uma fraude. Uma Precisão de 20% significa que a cada 100 usuários bloqueados, apenas 20 eram de fato fraudadores. | Minimiza o atrito gerado aos 80 clientes legítimos bloqueados por engano (Falsos Positivos).           |
| **F1-Score**                         | Média harmônica entre Precision e Recall. Pune modelos que tem disparidade extrema entre as duas métricas (ex: Recall 99% mas Precision 1%).                             | É a métrica de "ponto ideal" para encontrar o melhor **Threshold** (limiar de decisão).                |
| **PR-AUC** _(Precision-Recall Area)_ | Mede a performance geral do modelo através de todos os limiares de decisão possíveis.                                                                                    | Métrica mais robusta e segura (muito melhor que a curva ROC tradicional) para datasets desbalanceados. |

Abaixo, detalhamos conceitualmente, tecnicamente e os motivos do sucesso ou fracasso de cada modelo submetido ao laboratório de modelagem.

---

## 12.1 Os Casos de Fracasso (Exemplos Negativos)

### 12.1.1 Árvore de Decisão Simples (`DecisionTreeClassifier`)

> **🔴 Resultado Prático:** Fracasso Crítico. Apesar de boa métrica de treino inicial na validação cruzada (`CV Score = 0.81`), ruiu na base oficial de verificação com `F1-score` colapsando para `0.0`, errando a mão completamente no limiar `0.92`.

- **Por que ocorreu? (A Teoria):** Modelos monótonos foliares baseados em cortes diretos limitados (Gini/Entropy) não suportam desbalanceamento em massa orgânico se não receberem _Pruning_ (poda) severo muito bem delineado. A Árvore tenta minimizar a impureza total do nó, e na estatística macro deste conjunto, é mais fácil/barato e matematicamente recompensador para a folha apenas aglutinar massas de volume gigante "Legítimas", declarando `0`, para maximizar estabilidade, engolindo os parcos e irregulares `1` (fraudes) na multidão como varrimento de ruído orgânico.

**Parâmetros Técnicos Implementados:**

| Parâmetro                               | Valor | Conceito e Impacto                                                                                                                                                                                                                                                                                                                                            |
| :-------------------------------------- | :---- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `max_depth` <br>_(Profundidade Máxima)_ | `5`   | Define o limite de quantos "níveis" a árvore pode descer fazendo subdivisões. Usou-se limitação de teto deliberada visando atenuar a memorização viciosa de longo encadeamento (_Overfitting_). O efeito colateral reverso ocorreu subadestrando generalização no patamar final das partições (o modelo ficou "raso" demais para pegar as nuances da fraude). |

### 12.1.2 Floresta de Isolamento (`IForestWrapper` / Isolation Forest)

> **🔴 Resultado Prático:** Incompatibilidade arquitetural na essência do ataque. Obteve-se `PR_AUC` pífio de `0.025` e `F1-score` baixo na borda otimizada (`0.061`).

- **Por que ocorreu? (A Teoria):** _Isolation forest_ provê isolamento heurístico baseado estritamente na distância topológica (Quantos "cortes/splits" distam para espremer e exilar o nó anômalo longe da densidade padrão dimensional). O problema técnico-financeiro atual reside na essência da fraude de Identidade Sintética (Onboarding Fraud): Os adversários não são exilados mal-feitos, eles se mimetizam pesadamente para imitarem o ser humano padrão limpo do ecossistema e cruzarem o bloqueador da agência.
- **Conclusão Operacional:** Este não-supervisionado falha rotundamente contra comportamentos que mimetizam ou tentam absorver a média estatística. A fraude que nos assola não reflete Anomalia Espacial Pura Outlier.

**Parâmetros Técnicos Implementados:**

| Parâmetro                                | Valor  | Conceito e Impacto                                                                                                                                                                                           |
| :--------------------------------------- | :----- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_estimators` <br>_(Número de Árvores)_ | `200`  | A quantidade de árvores isoladas criadas no vetor (usado para estabilizar o consenso).                                                                                                                       |
| `contamination` <br>_(Contaminação)_     | `0.01` | Define a estimativa predefinida do número de outliers na base de dados (1%). Orienta o algoritmo a separar precocemente o quantitativo da massa isolada que se refere a nossa proporção de fraude conhecida. |

---

## 12.2. A Transição e o Progresso

### 12.2.1 Floresta Aleatória (`RandomForestClassifier`)

> **🟡 Resultado Prático:** Avanço metódico superando os blocos cegos monótonos do modelo de árvore de decisão (`F1 = 0.189`, `PR_AUC = 0.115`).

- **Por que ocorreu a melhora? (A Teoria):** Resolve o colapso unitário pela premissa pesada do _Bagging_ (Bootstrap Aggregation + Feature Subsampling). Cada uma das sub-árvores injetadas absorve pacotes orgânicos paralelos misturados randomicamente do conjunto total e reage a amostras microscópicas de colunas pré-sorteadas, gerando decorrelação forçada. O consenso demográfico maciço do modelo oblitera a variabilidade errante e provê generalização muito estável.
- **Ponto Fraco Metódico:** Random Forests não possuem aprendizado corretivo de repasse matricial temporal. Eles votam por consenso populacional inerte perante os `"Hard Examples"` (casos extremamente difíceis que compõem o fraudador escasso avançado). Eles batem um teto de cristal da complexidade preditiva se estagnando.

**Parâmetros Técnicos Implementados:**

| Parâmetro                                 | Valor  | Conceito e Impacto                                                                                                                                                                                                                                           |
| :---------------------------------------- | :----- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_estimators` <br>_(Nº de Estimadores)_  | `200`  | Quantas árvores avulsas compõem a floresta. Mais árvores garantem estabilidade de consenso populacional, mas aumentam pesadamente o tempo de processamento.                                                                                                  |
| `max_features` <br>_(Features Sorteadas)_ | `sqrt` | Determina o número de variáveis que o algoritmo visualizará em um único nó divisório (raiz quadrada). Cada árvore fica "cega" à vasta maioria das características, prevenindo que features fortíssimas dominem todas as árvores e trazendo diversidade real. |

### 12.2.2 Regressão Logística (`LogisticRegression`)

> **🟡 Resultado Prático:** Performance assombrosa perante o preconceito técnico do meio. Superou o limite orgânico do Random Forest de forma limpa (`F1 = 0.212`, `PR_AUC = 0.137`).

- **Por que funcionou e surpreendeu? (A Teoria):** Executou o _Baseline Benchmark_ provando que nossa modelagem de Data Engineering (`Mutual Information` para colunas baseadas no risco orgânico interativo do negócio) possui força colossal. O algoritmo traça linearmente hiperplanos nas 38 dimensões, apoiando-se unicamente nas reações combinadas.

**Parâmetros Técnicos Implementados:**

| Parâmetro                             | Valor  | Conceito e Impacto                                                                                                                                                                                                                                     |
| :------------------------------------ | :----- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `C` <br>_(Regularização Inversa)_     | `0.01` | Controla quão rígido o modelo será contra erros na base. Um valor baixo (0.01) é intensamente restritivo. Ele obriga o modelo a achar hiperplanos fracos, generalistas, mitigando _overfitting_. Evita que o modelo se adeque a ruídos desnecessários. |
| `penalty` <br>_(Estratégia Punitiva)_ | `'l2'` | Ridge Penalty. Impede que as multiplicações dos hiperparâmetros (como dar pontuação muito alta à idade do cliente) explodam, mantendo os pesos das características encolhidos suavemente próximos a zero.                                              |

---

## 12.3 A Vanguarda Numérica (Exemplos Positivos)

### 12.3.1 Redes Neurais Cíbridas (`MLPClassifier` - Perceptron Multicamadas)

> **🟢 Resultado Prático:** Captura de padrões ultra densos. Apresentou altíssima recuperação de captura sensível para o Pipeline Ensemble (`F1 = 0.220`).

- **Como atua e o por que (A Teoria):** Redes Neurais (Adeptos não-lineares puristas) absorvem correlações espaciais ocultas e sequenciamento invisível relacional. Requerem volumes titânicos de dados para extrair esses padrões sutis.

**Parâmetros Técnicos Implementados:**

| Parâmetro                                          | Valor    | Conceito e Impacto                                                                                                                                                                                                                  |
| :------------------------------------------------- | :------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hidden_layer_sizes` <br>_(Topologia Oculta)_      | `[128]`  | Quantas conexões formam os neurônios "abstratos". Adotou-se o padrão "Skinny but Wide", processando largura densa de uma vez sem descer níveis paralelos infindáveis, mantendo fôlego em um vetor tabular.                          |
| `alpha` <br>_(Taxa Punitiva L2)_                   | `0.001`  | Análogo à penalidade da LogReg, estabelece teto reacional contra pesos folgados de _Overfitting_ latente, limitando a força bruta excessiva do Backpropagation.                                                                     |
| `activation` <br>_(Função de Ativação)_            | `'tanh'` | Acionamento de não-linearidade sigmoidal suavizada da Tangente Hiperbólica. Evitou estouros reativos que a ReLU normal causa num sistema estritamente tabular, gerando balanço harmônico em negativo/positivo do gradiente reverso. |
| `learning_rate_init` <br>_(Passo de Convergência)_ | `0.0005` | Definida de forma minimalista estática para decair a "descida da montanha do custo matemático" estritamente a passo de formiga. Evita saltos cegos fora do abismo, preservando as detecções finas da Fraude diluída.                |

### 12.3.2 Máquinas de Gradient Boosting (`XGBClassifier` & `LGBMClassifier`)

> **🟢 Resultados Práticos:** Consagração do _Hype_ tabulado do Aprendizado de Máquina Competitivo atual. Domínio ostensivo Absoluto das métricas focadas nos cenários de desproporção assimétrica (`LGBM: F1 = 0.231, PR_AUC = 0.158` / `XGB: F1 = 0.221, PR_AUC = 0.147`). Limites de bloqueio otimizados na margem probabilística cravada em torno dos `0.88` aos `0.89`.

- **A Essência do Sucesso (Boosting Sequencial Teórico):** Contrastando radicalmente perante a Natureza Aleatória de "Média Popular" gerada em paridade das Árvores Bagging, estes motores geram árvores estritamente encadeadas no espaço-tempo. A Árvore subsequente constrói-se focando seu Gradiente puramente nos Resíduos (O erro contínuo). Elas perseguem de forma predatória os `"Erros Complexos"` — a parcela elitista de fraudadores avançados que despistam todas as avaliações rasas — alocando neles pesos gigantes de penalização forçatória.

**Engenharia de Parâmetros Implementados:**

| Parâmetro                        | Valores Testados                      | Conceito e Impacto                                                                                                                                                                                                                                                                                          |
| :------------------------------- | :------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `learning_rate` & `n_estimators` | `0.03 - 0.10` <br> & <br> `100 - 500` | **Taxa do Gradiente e Passos Temporais.** Formam uma Superfície Suave de Resgate. Ao criar centenas de árvores que aprendem parcelas minúsculas (ex: 3%) do erro anterior, evitamos radicalismos, refinando a complexidade progressiva sem queimar poder de análise num passo só.                           |
| `reg_alpha` & `reg_lambda`       | `0.01` (L1) <br> & <br> `5.0` (L2)    | **Funções de regularização matemáticas.** Elas destroem e impõem taxações mortais sobre galhos profundos soltos hiper específicos, castrando decorismos e propiciando defesas Anti-Explosivas no Teste Orgânico cego.                                                                                       |
| `subsample` & `colsample_bytree` | `0.6 - 0.7`                           | **Amostragem Fracionada Bidimensional.** Ditam a pureza do cegamento. A cada nova árvore, o algoritmo é amordaçado para enxergar unicamente 60% das Colunas randômicas e 60% dos Clientes. Isso adestra a rede a identificar padrões subentendidos, enfraquecendo a "Super-Dependência" em colunas campeãs. |

## 12.4 O Motor de Decisão Final: Ensemble de Votação com Veto Especial

> **🚀 Resultado Prático:** Redução massiva do atrito em produção preservando altíssimo bloqueio de fraudadores complexos, culminando numa operação simulada que barrou fraudes invisíveis à ótica de pilares isolados.

- **Por que ocorreu o sucesso operacional? (A Teoria):** Modelos de Machine Learning, não importando o quão hiper-otimizados estejam estruturalmente, carregam vieses intrínsecos de seus cálculos originais. A Rede Neural capta ruídos interacionais, ao passo que o XGBoost castra árvores específicas em busca do erro puro residual. No mundo corporativo prático, "apostar a empresa" em um único cérebro matemático expõe o negócio à fraqueza natural daquele algoritmo selecionado. O _Ensemble_ no nosso projeto não visa buscar a Acurácia Média Absoluta, e sim formar uma **banca julgadora mitigadora de erros heterogêneos**.

**Mecânica Técnica e Regras Dinâmicas de Convergência da Nossa Arquitetura:**
Ao invés de processar empiricamente as variâncias como Random Forests efetuam por debaixo dos panos (Médias Probabilísticas simples), aplicamos regras estritas de negócios ao _Output_ das três IAs Otimizadas do painel MLOps: **XGBoost, LightGBM e o Classificador MLP**.

A mecânica de Decisão Híbrida do Motor estabelece Posições e Papéis para os Modelos:

- **LightGBM:** O Campeão Global de F1-Score e Recall (Rede de varredura).
- **XGBoost:** Algoritmo de Consenso e Estabilidade (Validador tático).
- **MLP (Rede Neural):** O Campeão Cirúrgico de Precisão (Sniper).

Regras do Comitê de Predição de Onboarding:

1. **Maioria Plural Formadora do Flagrante:** Como cada um dos 3 modelos passou por seu _Threshold Tuning_ individual na calibração histórica, se **2 ou 3 modelos** acusam probabilidade acima de suas margens máximas individualizadas, consideramos consenso massivo da banca matemática: a transação toma **BLOQUEIO AUTOMÁTICO** de on-boarding (A taxa deste acerto é hiper-resiliente no nosso banco de testes).
2. **O Veto do Campeão de Precisão (MLP):** A regra garante blindagem contra Falsos Positivos. Se no decorrer das milhares análises o XGBoost e LightGBM declararem legitimidade (Aprovado), mas a poderosa **Rede Neural MLP** alertar uma Fraude Complexa Sozinha na mesa (Veto Isolado), o Orquestrador nega a liberação pura. Como a precisão da MLP beirava estonteantes 95% nos testes de treino contra os 15% de Falsos Positivos do Booster, não se barra o usuário final sem apoio da banca principal: esse escopo é alocado numa **Revisão Manual Humana Obrigatória**.
3. **Absolvição Tolerada (O Risco Aceitável):** Se a probabilidade aferida quebrou a trava estatística somente no crivo singular do LightGBM ou XGBoost, mas foi julgada natural pelos outros companheiros da bancada, a arquitetura assume a aprovação incondicional, limitando atritos agressivos com bons clientes.

## 12.5 Simulação MLOps: Streaming de Produção

Com o _Ensemble_ desenhado, desenvolvemos um Streaming MLOps interativo para simular as defesas automatizadas do Bank Anti-Fraud (BAF) ao longo de um extenso ciclo natural de aprovação de Onboarding de Crédito no mercado.

O script iterativo vetorizado varreu a base sob as seguintes pressuposições financeiras e estocásticas da vida real:

- **Tamanho do Backlog Processado:** 51.100 perfis virgens tentando abertura de conta online e emissão de cartões.
- **Densidade Realística da Fraude (Ataque Diluído):** Aproximadamente 2.15% (1.100 clientes) consistiam em identidades sintéticas (Fraude Oculta) tentando invadir os servidores em meio a 50.000 clientes íntegros reais.
- **Risco de Crédito Inicial Expandido:** Preestabelecemos cada conta com um Limite / Ticket Médio concedido de estrondosos `R$ 3.500,00`.

### 12.6 Diagnóstico Financeiro da Simulação de Impacto (Return On Investment)

Os resultados matemáticos provam a eficiência abismal de escalarmos Arquiteturas de Redes Ensembles contra regras de banco convencionais. Segundo o nosso **Relatório Gerencial Executivo**, a perfomance comportou-se da seguinte forma:

**Impactos Operacionais Controladores:**

- **Taxa de Assertividade Global (Accuracy):** Brincando na altíssima margem de **97.45% de Acerto** de decisões não-vigiadas pela máquina. Uma prova de valor robusta sobre o motor neural em lote vivo.
- **Atrito Computado Limitadamente:** Falsos Positivos na casa dos meros **488 Atendimentos**. Apenas uma taxa ínfima de **0.98% de Fricção** inflingida sobre a esteira de base limpa inteira. Quase imperceptível no funil prático de aquisição orgânica de clientes (CAC Seguro).
- **Triage e Human-in-the-Loop:** Redução da sobrecarga da mesa gerencial de analistas a pó, retendo unicamente apenas **37 casos severos** empurrados ao "Veto Cirúrgico Exclusivo" da Rede Neural MLP onde os modelos ficaram "emparedados".

**Eficácia Financeira e Mitigação Monetária (Patrimônio):**
Através dos números computados e das dezenas de falsidades barradas sumariamente do ecossistema, o Motor MLOps da Equipe produziu uma Retenção Patrimonial Protegida Definitiva de esmagadores **R$ 969.500,00** estritamente sem envolver qualquer mão de obra humana (Apenas com inferência local de _Threshold Limitados_).

A taxa fluída de escoamento e de aceitação, onde as catracas operacionais deram passagem à perfis corretos e minimizou-se o atrito dos algoritmos que afugentariam negócios limpos, produziu isoladamente a incrível alocação sistêmica de **R$ 175.000.000,00** distribuídos aos _Bons Clientes_, garantindo a robustez do fluxo de crédito da arquitetura em longo prazo.

### 12.7 O Trade-Off do Risco e o Paradoxo do "Glass Ceiling" (Análise do Recall)

Embora a arquitetura apresente Acertos Globais de 97% e atrito quase irrisório de Falsos Positivos, é imperativo que um arquiteto MLOps encare o fato de que uma considerável porção de ataques na simulação (Falsos Negativos) conseguiu cruzar a linha de defesa.

No mercado de Risco e Prevenção, a eficácia do nosso Modelo não atesta uma fraqueza no algoritmo, mas comprova um limite técnico conhecido como **"O Teto de Vidro da Base de Dados de Risco Tabular"**.

**1. A Camuflagem Perfeita (Baixa Information Gain):**
Conforme detalhado em nossa _Análise Exploratória Inicial (EDA)_, o _Mutual Information_ das features tabulares em relação ao Alvo era inerentemente baixo. Fraudadores em _Onboarding_ não reportam dados anômalos. Na ótica bidimensional (Salários, Idade, Tipo Residencial), as identidades sintéticas copiam os parâmetros estritos dos nossos assinantes Premium. A Inteligência Artificial (Mesmo utilizando estado da arte LightGBM/XGBoost) enxerga perfis estatisticamente idênticos.

**2. A Ausência de Rastros Comportamentais (Behavioral Footprint):**
Modelos robustos de validação de fraude limitam-se severamente quando privados de _Telemetria Comportamental_. Na base estática atual do _Fraud Sentinel_ faltam chaves cruciais de quebra de anonimato:

- _Micro-métricas Cognitivas:_ Velocidade e ritmo de digitação na tela do Onboarding (Diferenciar copy/paste de teclado orgânico).
- _Geofencing & IP Reputation:_ Assinatura estática da rede (O IP bate com a cidade do passaporte em mãos?).
- _Velocity Checks:_ Abusividade de hardware (O mesmo IMEI tentando transacionar 15 CPFs diferentes em 40 minutos).

**3. O Trade-Off de Negócios e a Imunização do CAC (User Experience):**
A calibração do Comitê de Ensemble poderia ter puxado _Thresholds_ mais rasos para aniquilar 100% dos _Falsos Negativos_, aprisionando brutalmente os fraudadores. Sob a óptica matemática, resolveríamos o vazamento financeiro do risco de _Credit Default_. No entanto, ao forçar a IA a condenar perfis situados em uma severa "zona cinzenta", estatisticamente bloquearíamos mais de 10.000 clientes legítimos no mercado com parâmetros idênticos.

A arquitetura do **Fraud Sentinel** executou com maestria sua Ordem de Negócio Conservadora: Sacrificou-se algumas centenas de estornos de fraude toleráveis frente ao imperdoável Custo de Aquisição Perdido (CAC) de milhares de negócios orgânicos destruídos pela fricção. O algoritmo se absteve de arruinar o _User Experience_. A futura iteração do modelo foca-se obrigatoriamente no engate da Engenharia de Dados alimentando fluxos estocásticos de Device Fingerprinting.
