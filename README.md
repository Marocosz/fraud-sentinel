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

# Com benchmark de algoritmos (demorado)
python main.py --compare-models

# Com simulacao de producao
python main.py --predict

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
  - [2.4 Separacao de Camadas](#24-separacao-de-camadas)
- [3. Estrutura de Diretorios](#3-estrutura-de-diretorios)
  - [3.1 Descricao Detalhada de Cada Arquivo](#31-descricao-detalhada-de-cada-arquivo)
    - [main.py -- Orquestrador Principal](#mainpy----orquestrador-principal)
    - [src/config.py -- Configuracoes Globais](#srcconfigpy----configuracoes-globais)
    - [src/data/make_dataset.py -- Engenharia de Dados](#srcdatamake_datasetpy----engenharia-de-dados)
    - [src/features/build_features.py -- Pipeline de Features (EDA-Driven)](#srcfeaturesbuild_featurespy----pipeline-de-features-eda-driven)
    - [src/models/trainers/reg_log_model.py -- Logistic Regression](#srcmodelstrainersreg_log_modelpy----logistic-regression)
    - [src/models/trainers/decision_tree_model.py -- Decision Tree](#srcmodelstrainersdecision_tree_modelpy----decision-tree)
    - [src/models/trainers/random_forest_model.py -- Random Forest](#srcmodelstrainersrandom_forest_modelpy----random-forest)
    - [src/models/trainers/xgboost_model.py -- XGBoost](#srcmodelstrainersxgboost_modelpy----xgboost)
    - [src/models/trainers/mlp_model.py -- MLP Neural Network](#srcmodelstrainersmlp_modelpy----mlp-neural-network)
    - [src/models/trainers/isolation_forest_model.py -- Isolation Forest](#srcmodelstrainersisolation_forest_modelpy----isolation-forest)
    - [src/models/compare_models.py -- Benchmark de Algoritmos](#srcmodelscompare_modelspy----benchmark-de-algoritmos)
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
- [7. Integracoes Externas](#7-integracoes-externas)
  - [7.1 Bibliotecas Criticas](#71-bibliotecas-criticas)
- [8. Logica e Algoritmos](#8-logica-e-algoritmos)
  - [8.1 Otimizacao de Memoria (Downcasting)](#81-otimizacao-de-memoria-downcasting)
  - [8.2 Threshold Tuning](#82-threshold-tuning)
  - [8.3 IForestWrapper (Adapter Pattern)](#83-iforestwrapper-adapter-pattern)
  - [8.4 Amostragem Estratificada para GridSearch](#84-amostragem-estratificada-para-gridsearch)
  - [8.5 Informacao Mutua (MI)](#85-informacao-mutua-mi)
  - [8.6 EDAFeatureEngineer (Feature Engineering Orientado por Dados)](#86-edafeatureengineer-feature-engineering-orientado-por-dados)
  - [8.7 Abstração de Treinamento (`BaseTrainer`)](#87-abstração-de-treinamento-basetrainer)
  - [8.8 Motor de Simulação (Ensemble PoV \& ROI)](#88-motor-de-simulação-ensemble-pov--roi)
- [9. Configuracoes e Variaveis de Ambiente](#9-configuracoes-e-variaveis-de-ambiente)
- [11. Estrategia de Logs e Monitoramento](#11-estrategia-de-logs-e-monitoramento)
  - [11.1 Logs em Console](#111-logs-em-console)
  - [11.2 Log Persistido (experiments_log.json)](#112-log-persistido-experiments_logjson)
  - [11.3 Diagnostico de Problemas](#113-diagnostico-de-problemas)
- [12. Pontos Criticos do Sistema](#12-pontos-criticos-do-sistema)
  - [12.1 Gargalos de Performance](#121-gargalos-de-performance)
  - [12.2 Riscos Arquiteturais](#122-riscos-arquiteturais)
  - [12.3 Partes Sensiveis](#123-partes-sensiveis)
- [13. Teoria Tecnica Envolvida](#13-teoria-tecnica-envolvida)
  - [13.1 Padroes de Projeto](#131-padroes-de-projeto)
  - [13.2 Conceitos de ML Aplicados](#132-conceitos-de-ml-aplicados)
  - [13.3 Conceitos Estatisticos](#133-conceitos-estatisticos)
- [14. Melhorias Futuras](#14-melhorias-futuras)
  - [14.1 Melhorias de Performance](#141-melhorias-de-performance)
  - [14.2 Refatoracoes Recomendadas](#142-refatoracoes-recomendadas)
- [15. Trabalhos Recentes Refatorados (Changelog Histórico)](#15-trabalhos-recentes-refatorados-changelog-histórico)

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

### 3. Benchmark de Modelos (Opcional)

- **Arquivo Responsável:** `compare_models.py`
- **Entrada:** `data/processed/X_train.csv`, `y_train.csv` (Uma extração proporcional rápida de 50 mil linhas para agilizar execuções iterativas).
- **Descrição da Atividade:**
  Roda múltiplos algoritmos contra testes cegos em camadas (Stratified K-Fold CV), ativando flags _Cost-Sensitive_ para dar pesos diferentes aos erros, avaliando qual Família de IA perfoma melhor (LogReg, Random Forest, XGBoost etc).
- **Saída Gerada:** Matriz e ranking tabular subistancializado (`reports/data/models_comparison_results.csv`), sumário de performance `reports/model_comparison_report.txt` e um gráfico PNG das métricas F1/ROC-AUC.

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
|   |   |-- build_features.py  # Pipeline EDA-driven (EDAFeatureEngineer + Scaler + OneHot)
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- compare_models.py      # Benchmark comparativo de algoritmos
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
|   |       |-- lightgbm_model.py      # Treinamento LightGBM (Campeao de Precisao)
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

### src/features/build_features.py -- Pipeline de Features (EDA-Driven)

| Atributo    | Descricao                                                                                |
| ----------- | ---------------------------------------------------------------------------------------- |
| **Classe**  | `EDAFeatureEngineer(BaseEstimator, TransformerMixin)` -- Transformer sklearn customizado |
| **Funcoes** | `get_preprocessor(X)`, `build_pipeline(X_train, model)`, `process_features()`            |
| **Entrada** | DataFrame X com features brutas                                                          |
| **Saida**   | `Pipeline` completo de 3 etapas (EDAFeatureEngineer -> ColumnTransformer -> Modelo)      |

O pipeline foi reestruturado com base nos insights da Analise Exploratoria (EDA) e agora possui 3 camadas:

**Camada 1 -- EDAFeatureEngineer** (transformer customizado):

| Transformacao            | Detalhe                                                                                                                             | Justificativa (EDA)                                                                       |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Remocao de features      | Remove `device_fraud_count` e `session_length_in_minutes`                                                                           | Variancia zero (MI=0.0001) e MI=0 com Mann-Whitney nao significativo (p=0.163)            |
| Tratamento de sentinelas | Converte -1 para NaN e cria flags (`has_prev_address`, `has_bank_history`, `has_device_emails`)                                     | Mediana de `prev_address_months_count` = -1 indicava >50% de dados marcados como ausentes |
| Clipping de outliers     | Clip nos percentis 1%/99% de `proposed_credit_limit`, `intended_balcon_amount`, `bank_branch_count_8w`, `prev_address_months_count` | Features com 15-24% de outliers pelo metodo IQR                                           |
| Flags de risco           | Cria `is_high_risk_housing`, `is_high_risk_employment`, `is_high_risk_os`, `is_high_risk_payment`, `is_teleapp_source`              | Categorias com 1.5x a 3.4x a taxa media de fraude                                         |
| Interacao digital        | Cria `digital_risk_score` = `email_is_free` \* `device_distinct_emails_8w`                                                          | Top 3 features por MI Score sao todas de comportamento digital                            |

**Camada 2 -- ColumnTransformer** (preprocessamento):

- Pipeline numerico: `SimpleImputer(median)` -> `RobustScaler()`
- Pipeline categorico: `SimpleImputer(constant='missing')` -> `OneHotEncoder(handle_unknown='ignore')`

**Camada 3 -- Modelo** (classificador):

- O classificador especifico de cada modelo (LogReg, XGBoost, RF, etc.)

Decisao tecnica: O `EDAFeatureEngineer` e um `BaseEstimator` do scikit-learn, sendo serializado junto com o modelo via `joblib.dump()`. Isso garante que as mesmas transformacoes sejam aplicadas automaticamente em treino, validacao cruzada e inferencia.

### src/models/trainers/reg_log_model.py -- Logistic Regression

| Atributo                           | Descricao                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Funcao principal**               | `train_logistic_regression()`                                                                    |
| **Estrategia de desbalanceamento** | `class_weight='balanced'` (sem SMOTE)                                                            |
| **Grid Search**                    | `C`: [0.01, 0.1, 1, 10]; `penalty`: ['l1', 'l2']                                                 |
| **Otimizacao**                     | Amostra estratificada de 100k linhas para GridSearch; retreino final com dataset completo        |
| **Saidas**                         | `logreg_best_model.pkl`, `logreg_threshold.txt`, `best_model_params.txt`, `experiments_log.json` |

### src/models/trainers/decision_tree_model.py -- Decision Tree

| Atributo             | Descricao                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **Funcao principal** | `train_decision_tree()`                                                                   |
| **Grid Search**      | `max_depth`: [5, 10, None]; `min_samples_split`: [2, 5]; `criterion`: ['gini', 'entropy'] |
| **Saidas**           | `dt_best_model.pkl`, `dt_threshold.txt`, `dt_best_model_params.txt`                       |

### src/models/trainers/random_forest_model.py -- Random Forest

| Atributo             | Descricao                                                                            |
| -------------------- | ------------------------------------------------------------------------------------ |
| **Funcao principal** | `train_random_forest()`                                                              |
| **Grid Search**      | `n_estimators`: [100, 200]; `max_depth`: [10, 20, None]; `min_samples_split`: [2, 5] |
| **Nota**             | RF usa `n_jobs=-1` internamente; GridSearch usa `n_jobs=1` para evitar conflito      |
| **Saidas**           | `rf_best_model.pkl`, `rf_threshold.txt`, `rf_best_model_params.txt`                  |

### src/models/trainers/xgboost_model.py -- XGBoost

| Atributo             | Descricao                                                                     |
| -------------------- | ----------------------------------------------------------------------------- |
| **Funcao principal** | `train_xgboost()`                                                             |
| **Estrategia**       | `scale_pos_weight=90` para compensar desbalanceamento                         |
| **Grid Search**      | `learning_rate`: [0.01, 0.1]; `n_estimators`: [100, 200]; `max_depth`: [3, 6] |
| **Otimizacao**       | Amostra estratificada de 100k para GridSearch; retreino no dataset completo   |
| **Saidas**           | `xgb_best_model.pkl`, `xgb_threshold.txt`, `xgb_best_model_params.txt`        |

### src/models/trainers/mlp_model.py -- MLP Neural Network

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

### src/models/compare_models.py -- Benchmark de Algoritmos

| Atributo             | Descricao                                                                                                                                           |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Funcao principal** | `compare_algorithms()`                                                                                                                              |
| **Competidores**     | LogReg, DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting, ExtraTrees, AdaBoost, XGBoost, LightGBM (opcional), CatBoost (opcional) |
| **Metodologia**      | Stratified 5-Fold CV com pipeline focado em Cost-Sensitive Learning (sem SMOTE)                                                                     |
| **Metricas**         | ROC-AUC, Recall, Precision, F1-Score                                                                                                                |
| **Saidas**           | `models_comparison_results.csv`, `model_comparison_report.txt`, `model_comparison_metrics.png`                                                      |

### src/serving/simulate_production.py -- Simulacao de Producao

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
    Roda 5-Fold CV para algoritmos usando Cost-Sensitive Learning
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
          - Se Fraud_Votes > 0 E Votante Unico == 'LightGBM': REVISÃO MANUAL (Veto de Precisão)
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
Se (Votos = 1) E (Voto de Fraude originado pelo LightGBM): --> REVISAO MANUAL (Médio/Alto Risco - Veto de Campeão)
Outros Casos (Voto 0 ou voto único de modelo fraco):  --> APROVADO (Baixo Risco Aceitável)
```

---

# 7. Integracoes Externas

O projeto **nao possui integracoes com APIs externas** em tempo de execucao. Todas as dependencias sao bibliotecas Python instaladas localmente.

## 7.1 Bibliotecas Criticas

| Biblioteca           | Versao          | Finalidade                                  |
| -------------------- | --------------- | ------------------------------------------- |
| scikit-learn         | 1.8.0           | Pipelines, modelos, metricas, preprocessing |
| xgboost              | 3.2.0           | Gradient Boosting otimizado                 |
| lightgbm             | 4.6.0           | Gradient Boosting (benchmark e Ensemble)    |
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

## 8.6 EDAFeatureEngineer (Feature Engineering Orientado por Dados)

O `EDAFeatureEngineer` e um transformer customizado do scikit-learn que aplica 5 transformacoes baseadas nos insights da EDA, em sequencia:

```
Dados Brutos (31 features)
    |
    v
1. Remocao: -2 features (device_fraud_count, session_length_in_minutes)
    |
    v
2. Sentinelas: -1 -> NaN + 3 flags binarias (has_prev_address, has_bank_history, has_device_emails)
    |
    v
3. Clipping: Percentis 1%/99% em 4 features com >15% outliers
    |
    v
4. Flags de Risco: +5 features binarias (housing BA, employment CC, OS windows, payment AC, source TELEAPP)
    |
    v
5. Interacao Digital: +1 feature (digital_risk_score = email_is_free * device_distinct_emails_8w)
    |
    v
Dados Engenheirados (38 features)
```

O transformer implementa `fit()` para aprender limites de clipping no conjunto de treino e `transform()` para aplicar todas as transformacoes. Por ser um `BaseEstimator`, e automaticamente serializado junto com o modelo.

## 8.7 Abstração de Treinamento (`BaseTrainer`)

Refatora o problema de duplicação que existia na versão primária, gerando Orientação a Objetos. A classe orquestra subamostragem `_get_sample()` rigorosa estratificada, injeção de dependências de modelos arbitrária, resolve união entre `GridSearchCV` e `RandomizedSearchCV`, define pesos usando `compute_sample_weight` e automatiza a trilha de Logs (`experiments_log.json`).

## 8.8 Motor de Simulação (Ensemble PoV & ROI)

Implementado com vetores `pandas` de altíssima velocidade, prevê dados em "Batch Array" diretamente pelos 3 Modelos Comitê otimizados simultaneamente (MLP, Xgb, LightGBM). Utiliza CLI (Command Line Interface) Visual para reportar:

- O Veredito de Maioria Simples ou Veto Especial, traduzidos em True Positives e False Negatives.
- Uma Inteligência de Negócio Financeira (`simulate_production.py`) cruza os erros contra um Ticket Médio de negócio predefinido e entrega ao usuário e Stakeholders um Documento final (`simulation_summary.txt`) contendo Patrimônio Salvo, Custo de Atrito, Taxa da Operação e o Lucro Resgatado em Dinheiro Requerido.

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

1. **Modelo nao encontrado**: Verificar se `main.py` foi executado antes de `simulate_production.py`
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

## 14.1 Melhorias de Performance

1. **Substituir GridSearchCV por Optuna/BayesSearchCV** para busca ainda mais robusta de hiperparametros em hipercubos densos.
2. **Implementar cache de preprocessamento avançado** para evitar recomputacao entre diferentes partições no motor de CV do `BaseTrainer`.
3. **Usar Parquet em vez de Pickle** caso ocorra imprecisão massiva e gargalo nas escritas do disco por datasets multibilionários.

## 14.2 Refatoracoes Recomendadas

1. **Extrair logica de persistencia de experimentos** para um container online (`MLflow` tracking).
2. **Centralizar configuracao de modelos** em um unico YAML/JSON para integração direta a pipelines de Deploy CI/CD em Nuvem (Airflow, KubeFlow).
3. **Implementar Explainable AI (SHAP/LIME)** diretamente conectada na resposta do Comitê de Decisão (`predict_ensemble.py`) informando qual feature barrou ou aprovou o risco.

---

# 15. Trabalhos Recentes Refatorados (Changelog Histórico)

O projeto sofria de graves limitações como _Código Duplicado de Alto Impacto_ nos scripts da pasta Modelos, variáveis ociosas nos `dict` originais, importações duplas, ausência de Suíte de Testes (TDD) e dependência engessada a rotinas arcaicas e ineficientes de re-sampling como SMOTE (revertido para _Cost-Sensitive Learning_).

Todas as restrições foram abordadas e neutralizadas nas Sprints recentes pelas implementações de:

- **`BaseTrainer.py`** como classe mestra de Treino que absorve 80% do Boilerplate e engessa o Pipeline End-to-End num túnel só de código.
- Escavação de Testes de Unidade através do Framework `pytest` nativo localizados na pasta `tests/` avaliando exatidão da Pipeline de Dados e Feature Engineering.
- Refatoração total para Batch Vectorization nos simuladores preditivos, derrubando o inferência de horas para segundos na simulação macro.
