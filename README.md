# Fraud Sentinel - Sistema Avancado de Deteccao de Fraudes Bancarias

> [!NOTE]
> Link para download da base de dados usada: https://drive.google.com/file/d/1KWKHddAwpZ2HAwsWmL0HWlUXFw8HWf9N/view?usp=sharing

---

- [Fraud Sentinel - Sistema Avancado de Deteccao de Fraudes Bancarias](#fraud-sentinel---sistema-avancado-de-deteccao-de-fraudes-bancarias)
- [1 Resultados da Analise Exploratoria (EDA)](#1-resultados-da-analise-exploratoria-eda)
  - [1.1 Carga de Dados](#11-carga-de-dados)
  - [1.2 Estrutura e Qualidade dos Dados](#12-estrutura-e-qualidade-dos-dados)
    - [1.2.1 Tipos de Dados e Cardinalidade](#121-tipos-de-dados-e-cardinalidade)
  - [1.3 Dominio das Variaveis Categoricas](#13-dominio-das-variaveis-categoricas)
  - [1.4 Estatisticas Descritivas (Variaveis Numericas)](#14-estatisticas-descritivas-variaveis-numericas)
  - [1.5 Quantificacao de Outliers (Metodo IQR)](#15-quantificacao-de-outliers-metodo-iqr)
  - [1.6 Distribuicao do Target](#16-distribuicao-do-target)
  - [1.7 Testes Estatisticos (Mann-Whitney U -- Fraude vs Legitima)](#17-testes-estatisticos-mann-whitney-u----fraude-vs-legitima)
  - [1.8 Mutual Information (Importancia de Features)](#18-mutual-information-importancia-de-features)
  - [1.9 Analise Temporal (Taxa de Fraude por Mes)](#19-analise-temporal-taxa-de-fraude-por-mes)
  - [1.10 Correlacoes com o Target (Spearman)](#110-correlacoes-com-o-target-spearman)
  - [1.11 Analise de Risco Categorico (Taxa de Fraude por Categoria)](#111-analise-de-risco-categorico-taxa-de-fraude-por-categoria)
    - [1.11.1 Risco por Tipo de Pagamento (`payment_type`)](#1111-risco-por-tipo-de-pagamento-payment_type)
    - [1.11.2 Risco por Status de Emprego (`employment_status`)](#1112-risco-por-status-de-emprego-employment_status)
    - [1.11.3 Risco por Status de Moradia (`housing_status`)](#1113-risco-por-status-de-moradia-housing_status)
    - [1.11.4 Risco por Origem da Solicitacao (`source`)](#1114-risco-por-origem-da-solicitacao-source)
    - [1.11.5 Risco por Sistema Operacional (`device_os`)](#1115-risco-por-sistema-operacional-device_os)
- [2. Visao Geral do Projeto](#2-visao-geral-do-projeto)
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
    - [src/features/build_features.py -- Pipeline de Features](#srcfeaturesbuild_featurespy----pipeline-de-features)
    - [src/models/trainers/reg_log_model.py -- Logistic Regression](#srcmodelsreg_log_modelpy----logistic-regression)
    - [src/models/trainers/decision_tree_model.py -- Decision Tree](#srcmodelsdecision_tree_modelpy----decision-tree)
    - [src/models/trainers/random_forest_model.py -- Random Forest](#srcmodelsrandom_forest_modelpy----random-forest)
    - [src/models/trainers/xgboost_model.py -- XGBoost](#srcmodelsxgboost_modelpy----xgboost)
    - [src/models/trainers/mlp_model.py -- MLP Neural Network](#srcmodelsmlp_modelpy----mlp-neural-network)
    - [src/models/trainers/isolation_forest_model.py -- Isolation Forest](#srcmodelsisolation_forest_modelpy----isolation-forest)
    - [src/models/compare_models.py -- Benchmark de Algoritmos](#srcmodelscompare_modelspy----benchmark-de-algoritmos)
    - [src/serving/simulate_production.py -- Simulacao de Producao](#srcmodelspredict_modelpy----simulacao-de-producao)
    - [src/models/force_precision.py -- Ajuste de Precision-Alvo](#srcmodelsforce_precisionpy----ajuste-de-precision-alvo)
    - [src/visualization/generate_eda_report.py -- EDA Automatizada](#srcvisualizationgenerate_eda_reportpy----eda-automatizada)
    - [src/visualization/visualize.py -- Avaliacao Final](#srcvisualizationvisualizepy----avaliacao-final)
- [4. Fluxos Detalhados](#4-fluxos-detalhados)
  - [4.1 Fluxo Principal do Sistema](#41-fluxo-principal-do-sistema)
  - [4.2 Fluxo de Treinamento de Modelo (Generico)](#42-fluxo-de-treinamento-de-modelo-generico)
  - [4.3 Fluxo de Inferencia (Predicao)](#43-fluxo-de-inferencia-predicao)
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
- [9. Configuracoes e Variaveis de Ambiente](#9-configuracoes-e-variaveis-de-ambiente)
- [10. Como Executar o Projeto](#10-como-executar-o-projeto)
  - [10.1 Requisitos](#101-requisitos)
  - [10.2 Instalacao](#102-instalacao)
  - [10.3 Preparacao dos Dados](#103-preparacao-dos-dados)
  - [10.4 Execucao](#104-execucao)
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
  - [14.1 Sugestoes Estruturais](#141-sugestoes-estruturais)
  - [14.2 Melhorias de Performance](#142-melhorias-de-performance)
  - [14.3 Refatoracoes Recomendadas](#143-refatoracoes-recomendadas)
- [15. Analise Critica da Arquitetura](#15-analise-critica-da-arquitetura)
  - [15.1 Codigo Duplicado (Alto Impacto)](#151-codigo-duplicado-alto-impacto)
  - [15.2 Chave Duplicada no Dicionario](#152-chave-duplicada-no-dicionario)
  - [15.3 Variavel Nao Utilizada](#153-variavel-nao-utilizada)
  - [15.4 Import Duplicado](#154-import-duplicado)
  - [15.5 Testes Nao Implementados](#155-testes-nao-implementados)
  - [15.6 Inconsistencia na Estrategia de Amostragem para GridSearch](#156-inconsistencia-na-estrategia-de-amostragem-para-gridsearch)
  - [15.7 Acoplamento com Sistema de Arquivos](#157-acoplamento-com-sistema-de-arquivos)

# 1 Resultados da Analise Exploratoria (EDA)

A partir do arquivo `generate_eda_report.py` criamos um relatorio textual (`reports/eda_summary.txt`) com o sumario completo da base de dados. A seguir estao todas as informacoes e descricoes geradas a partir da analise da base inicial.

## 1.1 Carga de Dados

O dataset carregado possui **1.000.000 de linhas** e **32 colunas**. A coluna alvo (target) e `fraud_bool`, que indica se a abertura de conta e fraudulenta (1) ou legitima (0).

## 1.2 Estrutura e Qualidade dos Dados

O dataset nao possui **nenhum valor nulo** e **nenhuma linha duplicada** (0.00%). Todos os 1.000.000 de registros sao completos e unicos.

### 1.2.1 Tipos de Dados e Cardinalidade

| Coluna                             | Tipo    | Nulos | % Nulos | Cardinalidade |
| ---------------------------------- | ------- | ----- | ------- | ------------- |
| `fraud_bool`                       | int64   | 0     | 0.0%    | 2             |
| `income`                           | float64 | 0     | 0.0%    | 9             |
| `name_email_similarity`            | float64 | 0     | 0.0%    | 998.861       |
| `prev_address_months_count`        | int64   | 0     | 0.0%    | 374           |
| `current_address_months_count`     | int64   | 0     | 0.0%    | 423           |
| `customer_age`                     | int64   | 0     | 0.0%    | 9             |
| `days_since_request`               | float64 | 0     | 0.0%    | 989.330       |
| `intended_balcon_amount`           | float64 | 0     | 0.0%    | 994.971       |
| `payment_type`                     | object  | 0     | 0.0%    | 5             |
| `zip_count_4w`                     | int64   | 0     | 0.0%    | 6.306         |
| `velocity_6h`                      | float64 | 0     | 0.0%    | 998.687       |
| `velocity_24h`                     | float64 | 0     | 0.0%    | 998.940       |
| `velocity_4w`                      | float64 | 0     | 0.0%    | 998.318       |
| `bank_branch_count_8w`             | int64   | 0     | 0.0%    | 2.326         |
| `date_of_birth_distinct_emails_4w` | int64   | 0     | 0.0%    | 40            |
| `employment_status`                | object  | 0     | 0.0%    | 7             |
| `credit_risk_score`                | int64   | 0     | 0.0%    | 551           |
| `email_is_free`                    | int64   | 0     | 0.0%    | 2             |
| `housing_status`                   | object  | 0     | 0.0%    | 7             |
| `phone_home_valid`                 | int64   | 0     | 0.0%    | 2             |
| `phone_mobile_valid`               | int64   | 0     | 0.0%    | 2             |
| `bank_months_count`                | int64   | 0     | 0.0%    | 33            |
| `has_other_cards`                  | int64   | 0     | 0.0%    | 2             |
| `proposed_credit_limit`            | float64 | 0     | 0.0%    | 12            |
| `foreign_request`                  | int64   | 0     | 0.0%    | 2             |
| `source`                           | object  | 0     | 0.0%    | 2             |
| `session_length_in_minutes`        | float64 | 0     | 0.0%    | 994.887       |
| `device_os`                        | object  | 0     | 0.0%    | 5             |
| `keep_alive_session`               | int64   | 0     | 0.0%    | 2             |
| `device_distinct_emails_8w`        | int64   | 0     | 0.0%    | 4             |
| `device_fraud_count`               | int64   | 0     | 0.0%    | 1             |
| `month`                            | int64   | 0     | 0.0%    | 8             |

Resumo de tipos: 9 colunas float64, 18 colunas int64, 5 colunas object (categoricas). Uso de memoria: ~244 MB.

## 1.3 Dominio das Variaveis Categoricas

| Variavel            | Categorias | Valores                               |
| ------------------- | ---------- | ------------------------------------- |
| `payment_type`      | 5          | AA, AB, AC, AD, AE                    |
| `employment_status` | 7          | CA, CB, CC, CD, CE, CF, CG            |
| `housing_status`    | 7          | BA, BB, BC, BD, BE, BF, BG            |
| `source`            | 2          | INTERNET, TELEAPP                     |
| `device_os`         | 5          | linux, macintosh, other, windows, x11 |

## 1.4 Estatisticas Descritivas (Variaveis Numericas)

| Variavel                           | Media   | Desvio Padrao | Min     | Q1 (25%) | Mediana (50%) | Q3 (75%) | Max      |
| ---------------------------------- | ------- | ------------- | ------- | -------- | ------------- | -------- | -------- |
| `income`                           | 0.5627  | 0.2903        | 0.10    | 0.30     | 0.60          | 0.80     | 0.90     |
| `name_email_similarity`            | 0.4937  | 0.2891        | ~0.00   | 0.2252   | 0.4922        | 0.7556   | ~1.00    |
| `prev_address_months_count`        | 16.72   | 44.05         | -1      | -1       | -1            | 12       | 383      |
| `current_address_months_count`     | 86.59   | 88.41         | -1      | 19       | 52            | 130      | 428      |
| `customer_age`                     | 33.69   | 12.03         | 10      | 20       | 30            | 40       | 90       |
| `days_since_request`               | 1.03    | 5.38          | ~0.00   | 0.007    | 0.015         | 0.026    | 78.46    |
| `intended_balcon_amount`           | 8.66    | 20.24         | -15.53  | -1.18    | -0.83         | 4.98     | 112.96   |
| `zip_count_4w`                     | 1572.69 | 1005.37       | 1       | 894      | 1263          | 1944     | 6700     |
| `velocity_6h`                      | 5665.30 | 3009.38       | -170.60 | 3436.37  | 5319.77       | 7680.72  | 16715.57 |
| `velocity_24h`                     | 4769.78 | 1479.21       | 1300.31 | 3593.18  | 4749.92       | 5752.57  | 9506.90  |
| `velocity_4w`                      | 4856.32 | 919.84        | 2825.75 | 4268.37  | 4913.44       | 5488.08  | 6994.76  |
| `bank_branch_count_8w`             | 184.36  | 459.63        | 0       | 1        | 9             | 25       | 2385     |
| `date_of_birth_distinct_emails_4w` | 9.50    | 5.03          | 0       | 6        | 9             | 13       | 39       |
| `credit_risk_score`                | 130.99  | 69.68         | -170    | 83       | 122           | 178      | 389      |
| `email_is_free`                    | 0.5299  | 0.4991        | 0       | 0        | 1             | 1        | 1        |
| `phone_home_valid`                 | 0.4171  | 0.4931        | 0       | 0        | 0             | 1        | 1        |
| `phone_mobile_valid`               | 0.8897  | 0.3133        | 0       | 1        | 1             | 1        | 1        |
| `bank_months_count`                | 10.84   | 12.12         | -1      | -1       | 5             | 25       | 32       |
| `has_other_cards`                  | 0.2230  | 0.4163        | 0       | 0        | 0             | 0        | 1        |
| `proposed_credit_limit`            | 515.85  | 487.56        | 190     | 200      | 200           | 500      | 2100     |
| `foreign_request`                  | 0.0252  | 0.1569        | 0       | 0        | 0             | 0        | 1        |
| `session_length_in_minutes`        | 7.54    | 8.03          | -1.00   | 3.10     | 5.11          | 8.87     | 85.90    |
| `keep_alive_session`               | 0.5769  | 0.4940        | 0       | 0        | 1             | 1        | 1        |
| `device_distinct_emails_8w`        | 1.02    | 0.18          | -1      | 1        | 1             | 1        | 2        |
| `device_fraud_count`               | 0.00    | 0.00          | 0       | 0        | 0             | 0        | 0        |
| `month`                            | 3.29    | 2.21          | 0       | 1        | 3             | 5        | 7        |

## 1.5 Quantificacao de Outliers (Metodo IQR)

| Variavel                           | Outliers | % Outliers | Limite Inferior | Limite Superior |
| ---------------------------------- | -------- | ---------- | --------------- | --------------- |
| `proposed_credit_limit`            | 241.742  | 24.17%     | -250.00         | 950.00          |
| `has_other_cards`                  | 222.988  | 22.30%     | 0.00            | 0.00            |
| `intended_balcon_amount`           | 222.702  | 22.27%     | -10.43          | 14.23           |
| `bank_branch_count_8w`             | 175.243  | 17.52%     | -35.00          | 61.00           |
| `prev_address_months_count`        | 157.320  | 15.73%     | -20.50          | 31.50           |
| `phone_mobile_valid`               | 110.324  | 11.03%     | 1.00            | 1.00            |
| `days_since_request`               | 94.834   | 9.48%      | -0.02           | 0.06            |
| `session_length_in_minutes`        | 78.789   | 7.88%      | -5.54           | 17.51           |
| `zip_count_4w`                     | 59.871   | 5.99%      | -681.00         | 3519.00         |
| `current_address_months_count`     | 41.001   | 4.10%      | -147.50         | 296.50          |
| `device_distinct_emails_8w`        | 31.933   | 3.19%      | 1.00            | 1.00            |
| `foreign_request`                  | 25.242   | 2.52%      | 0.00            | 0.00            |
| `date_of_birth_distinct_emails_4w` | 9.734    | 0.97%      | -4.50           | 23.50           |
| `velocity_6h`                      | 9.005    | 0.90%      | -2930.16        | 14047.25        |
| `credit_risk_score`                | 8.729    | 0.87%      | -59.50          | 320.50          |
| `velocity_24h`                     | 2.917    | 0.29%      | 354.09          | 8991.67         |
| `customer_age`                     | 1.373    | 0.14%      | -10.00          | 70.00           |
| `name_email_similarity`            | 0        | 0.00%      | -0.57           | 1.55            |
| `income`                           | 0        | 0.00%      | -0.45           | 1.55            |
| `velocity_4w`                      | 0        | 0.00%      | 2438.80         | 7317.66         |

## 1.6 Distribuicao do Target

| Classe       | Total   | Percentual |
| ------------ | ------- | ---------- |
| 0 (Legitima) | 988.971 | 98.90%     |
| 1 (Fraude)   | 11.029  | 1.10%      |

O dataset e **extremamente desbalanceado**: apenas 1.10% das aberturas de conta sao fraudulentas. Isso justifica o uso de tecnicas como Cost-Sensitive Learning (`class_weight='balanced'`, `scale_pos_weight=90`) e metricas como ROC-AUC e Recall em vez de Acuracia.

## 1.7 Testes Estatisticos (Mann-Whitney U -- Fraude vs Legitima)

O teste Mann-Whitney U e um teste nao-parametrico que verifica se a distribuicao de uma variavel e estatisticamente diferente entre os dois grupos (Fraude e Legitima). Se p-value < 0.05, a diferenca e significativa.

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

**Conclusao**: 24 de 26 variaveis numericas apresentam diferenca estatisticamente significativa entre fraudes e contas legitimas. Apenas `session_length_in_minutes` e `device_fraud_count` nao apresentaram significancia (p >= 0.05).

## 1.8 Mutual Information (Importancia de Features)

O score de Mutual Information mede a dependencia estatistica entre cada feature e o target, capturando relacoes nao-lineares que a correlacao tradicional ignora. Quanto maior o score, maior o poder preditivo da variavel.

| Posicao | Variavel                           | MI Score |
| ------- | ---------------------------------- | -------- |
| 1       | `device_distinct_emails_8w`        | 0.010217 |
| 2       | `email_is_free`                    | 0.010028 |
| 3       | `keep_alive_session`               | 0.009970 |
| 4       | `phone_mobile_valid`               | 0.007698 |
| 5       | `phone_home_valid`                 | 0.006016 |
| 6       | `proposed_credit_limit`            | 0.004636 |
| 7       | `customer_age`                     | 0.004512 |
| 8       | `income`                           | 0.003630 |
| 9       | `has_other_cards`                  | 0.001935 |
| 10      | `credit_risk_score`                | 0.001892 |
| 11      | `bank_months_count`                | 0.001779 |
| 12      | `date_of_birth_distinct_emails_4w` | 0.001648 |
| 13      | `name_email_similarity`            | 0.001588 |
| 14      | `prev_address_months_count`        | 0.001465 |
| 15      | `month`                            | 0.001373 |
| 16      | `intended_balcon_amount`           | 0.001333 |
| 17      | `current_address_months_count`     | 0.001302 |
| 18      | `velocity_4w`                      | 0.001273 |
| 19      | `velocity_6h`                      | 0.001098 |
| 20      | `bank_branch_count_8w`             | 0.000955 |
| 21      | `days_since_request`               | 0.000603 |
| 22      | `velocity_24h`                     | 0.000184 |
| 23      | `device_fraud_count`               | 0.000100 |
| 24      | `zip_count_4w`                     | 0.000000 |
| 25      | `session_length_in_minutes`        | 0.000000 |
| 26      | `foreign_request`                  | 0.000000 |

As features com maior poder preditivo sao relacionadas ao comportamento digital (`device_distinct_emails_8w`, `email_is_free`, `keep_alive_session`) e dados de contato (`phone_mobile_valid`, `phone_home_valid`), indicando que o perfil digital do solicitante e um forte indicador de fraude.

## 1.9 Analise Temporal (Taxa de Fraude por Mes)

| Mes | Taxa de Fraude |
| --- | -------------- |
| 0   | 1.13%          |
| 1   | 0.94%          |
| 2   | 0.87%          |
| 3   | 0.92%          |
| 4   | 1.14%          |
| 5   | 1.18%          |
| 6   | 1.34%          |
| 7   | 1.47%          |

A taxa de fraude apresenta uma **tendencia crescente ao longo dos meses** (de 0.87% no mes 2 para 1.47% no mes 7), sugerindo um possivel aumento na atividade fraudulenta ao longo do periodo de coleta ou sazonalidade.

## 1.10 Correlacoes com o Target (Spearman)

| Variavel                           | Correlacao com `fraud_bool` | Direcao                                         |
| ---------------------------------- | --------------------------- | ----------------------------------------------- |
| `credit_risk_score`                | +0.0602                     | Positiva (maior score = mais fraude)            |
| `customer_age`                     | +0.0581                     | Positiva (mais velho = mais fraude)             |
| `proposed_credit_limit`            | +0.0574                     | Positiva (limite maior = mais fraude)           |
| `income`                           | +0.0496                     | Positiva (renda maior = mais fraude)            |
| `current_address_months_count`     | +0.0485                     | Positiva (mais tempo no endereco = mais fraude) |
| `device_distinct_emails_8w`        | +0.0365                     | Positiva                                        |
| `email_is_free`                    | +0.0278                     | Positiva                                        |
| `foreign_request`                  | +0.0169                     | Positiva                                        |
| `month`                            | +0.0129                     | Positiva                                        |
| `keep_alive_session`               | -0.0503                     | Negativa (sessao ativa = menos fraude)          |
| `prev_address_months_count`        | -0.0463                     | Negativa                                        |
| `date_of_birth_distinct_emails_4w` | -0.0456                     | Negativa                                        |
| `name_email_similarity`            | -0.0373                     | Negativa (menor similaridade = mais fraude)     |
| `has_other_cards`                  | -0.0352                     | Negativa                                        |
| `phone_home_valid`                 | -0.0351                     | Negativa                                        |
| `bank_branch_count_8w`             | -0.0322                     | Negativa                                        |
| `device_fraud_count`               | NaN                         | Variancia zero (constante)                      |

As correlacoes sao baixas em valor absoluto (max ~0.06), o que e esperado em problemas de fraude. Isso indica que nenhuma variavel isolada e suficiente para prever fraude, sendo necessario o uso de modelos multivariados.

## 1.11 Analise de Risco Categorico (Taxa de Fraude por Categoria)

### 1.11.1 Risco por Tipo de Pagamento (`payment_type`)

| Categoria | Taxa de Fraude |
| --------- | -------------- |
| AC        | 1.67%          |
| AB        | 1.13%          |
| AD        | 1.08%          |
| AA        | 0.53%          |
| AE        | 0.35%          |

### 1.11.2 Risco por Status de Emprego (`employment_status`)

| Categoria | Taxa de Fraude |
| --------- | -------------- |
| CC        | 2.47%          |
| CG        | 1.55%          |
| CA        | 1.22%          |
| CB        | 0.69%          |
| CD        | 0.38%          |
| CE        | 0.23%          |
| CF        | 0.19%          |

### 1.11.3 Risco por Status de Moradia (`housing_status`)

| Categoria | Taxa de Fraude |
| --------- | -------------- |
| BA        | 3.75%          |
| BD        | 0.86%          |
| BC        | 0.61%          |
| BB        | 0.60%          |
| BF        | 0.42%          |
| BG        | 0.40%          |
| BE        | 0.34%          |

A categoria BA de `housing_status` apresenta taxa de fraude **3.4x maior** que a media geral (3.75% vs 1.10%), sendo o segmento de maior risco em todo o dataset.

### 1.11.4 Risco por Origem da Solicitacao (`source`)

| Categoria | Taxa de Fraude |
| --------- | -------------- |
| TELEAPP   | 1.59%          |
| INTERNET  | 1.10%          |

Solicitacoes via TELEAPP apresentam taxa de fraude 44% maior que via INTERNET.

### 1.11.5 Risco por Sistema Operacional (`device_os`)

| Categoria | Taxa de Fraude |
| --------- | -------------- |
| windows   | 2.47%          |
| macintosh | 1.40%          |
| x11       | 1.12%          |
| other     | 0.58%          |
| linux     | 0.52%          |

Dispositivos com sistema operacional Windows apresentam a maior taxa de fraude (2.47%), mais que o dobro da media geral.

---

# 2. Visao Geral do Projeto

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

| Etapa                               | Entrada                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Arquivo                                                                                                                                 | Descricao                                                                                                                                                                                                                                                                                                                                                                                         | Saida                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Ingestao de Dados                | `data/raw/Base.csv` -- O dataset bruto do BAF Suite (NeurIPS 2022) e o ponto de partida de todo o sistema. Contem todas as features sociodemograficas e comportamentais das aberturas de conta, com rotulagem binaria de fraude. E necessario como fonte primaria porque todo o pipeline depende de dados historicos rotulados para aprender padroes.                                                                                                                    | `make_dataset.py`                                                                                                                       | Carrega o CSV bruto, aplica downcasting de tipos numericos (float64 para float32, int64 para int8) para reduzir consumo de RAM, valida a existencia da coluna target (`fraud_bool`), e executa a divisao estratificada 80/20 que garante matematicamente que a proporcao de fraudes (~1%) seja identica nos conjuntos de treino e teste.                                                          | `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` -- Quatro arquivos CSV limpos e otimizados. Sao separados em features (X) e target (y) porque o scikit-learn exige essa separacao. O split estratificado e salvo em disco para que todas as etapas subsequentes trabalhem sobre exatamente os mesmos dados, garantindo reprodutibilidade.                                                                                                                                                 |
| 2. Analise Exploratoria             | `data/raw/Base.csv` -- O dataset bruto original e carregado novamente (nao os processados) porque a EDA precisa analisar os dados no estado natural, sem transformacoes de escala ou encoding, para identificar problemas reais como nulos, outliers e distribuicoes originais.                                                                                                                                                                                          | `generate_eda_report.py`                                                                                                                | Executa um raio-X completo dos dados: calcula estatisticas descritivas, quantifica outliers (IQR), roda testes de hipotese (Mann-Whitney U) para validar significancia estatistica de cada feature, calcula Mutual Information para ranquear importancia, gera boxplots comparativos, heatmaps de correlacao (Spearman), analises de risco categorico, e um dashboard HTML interativo (Sweetviz). | `reports/data/*.csv` (7 tabelas de metricas), `reports/figures/eda/*.png` (7+ graficos), `reports/eda_summary.txt` (relatorio textual consolidado), `reports/sweetviz_report.html` (dashboard interativo) -- Esses artefatos servem para o cientista de dados tomar decisoes informadas sobre quais features usar, quais tratamentos aplicar, e validar cientificamente que os dados possuem sinal discriminativo para fraude.                                                                                    |
| 3. Benchmark de Modelos (Opcional)  | `data/processed/X_train.csv`, `y_train.csv` -- Os dados de treino processados sao necessarios porque o benchmark precisa avaliar algoritmos sobre dados comparaveis. Uma amostra estratificada de 50k linhas e extraida para viabilizar a execucao em tempo razoavel sem perder representatividade estatistica.                                                                                                                                                          | `compare_models.py`                                                                                                                     | Executa um torneio entre 8 a 10 algoritmos (LogReg, DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting, ExtraTrees, AdaBoost, XGBoost, e opcionalmente LightGBM e CatBoost) usando validacao cruzada estratificada de 5 folds. O SMOTE e aplicado dentro de cada fold via ImbPipeline para prevenir data leakage. Mede ROC-AUC, Recall, Precision e F1.                           | `reports/data/models_comparison_results.csv` (tabela com medias e desvios de todas as metricas), `reports/model_comparison_report.txt` (relatorio executivo com ranking), `reports/figures/model_comparison_metrics.png` (grafico de barras comparativo) -- Esses artefatos permitem escolher objetivamente qual algoritmo tem melhor potencial antes de investir tempo na otimizacao de hiperparametros.                                                                                                         |
| 4. Treinamento e Otimizacao         | `data/processed/X_train.csv`, `y_train.csv` -- Os dados de treino sao necessarios para o modelo aprender os padroes de fraude. Cada script de modelo os carrega para construir o pipeline completo (preprocessamento + classificador) e otimizar hiperparametros via busca exaustiva.                                                                                                                                                                                    | `reg_log_model.py`, `decision_tree_model.py`, `random_forest_model.py`, `xgboost_model.py`, `mlp_model.py`, `isolation_forest_model.py` | Cada script cria um pipeline (preprocessor + modelo), executa GridSearchCV com Stratified K-Fold (3 folds) para encontrar os melhores hiperparametros, retreina o modelo vencedor no dataset completo (quando aplicavel), e executa Threshold Tuning que varre a curva Precision-Recall para encontrar o limiar de decisao que maximiza o F1-Score.                                               | `models/{nome}_best_model.pkl` (modelo serializado pronto para producao), `models/model_{nome}_{timestamp}.pkl` (copia versionada para historico), `models/{nome}_threshold.txt` (threshold otimizado), `models/{nome}_best_model_params.txt` (hiperparametros vencedores), `reports/experiments_log.json` (registro do experimento) -- Cada artefato cumpre um papel: o PKL e o modelo reutilizavel, o threshold define o ponto de operacao, e o JSON garante rastreabilidade completa de todos os experimentos. |
| 5. Avaliacao Final                  | `data/processed/X_test.csv`, `y_test.csv` (dados que o modelo nunca viu) e `models/{nome}_best_model.pkl` (modelo treinado) -- O blind test set e essencial porque simula dados reais de producao. Usar dados de treino para avaliar geraria metricas artificialmente infladas (overfitting). O modelo e carregado serializado para simular exatamente o que aconteceria em producao.                                                                                    | `visualize.py`                                                                                                                          | Carrega o modelo treinado e os dados de teste, gera predicoes de classe e probabilidade, calcula metricas finais (ROC-AUC, Precision, Recall, F1), plota a Matriz de Confusao (visualiza falsos positivos e negativos), a Curva ROC (capacidade de discriminacao) e o grafico de importancia de features (explicabilidade). Atualiza o log de experimentos com as metricas reais.                 | `reports/figures/confusion_matrix_{nome}.png`, `reports/figures/roc_curve_{nome}.png`, `reports/figures/feature_importance_coefficients.png` (se modelo linear) -- Graficos essenciais para validar se o modelo esta pronto para producao e comunicar resultados para stakeholders. O `experiments_log.json` e atualizado com metricas reais de teste, fechando o ciclo de rastreabilidade.                                                                                                                       |
| 6. Simulacao de Producao (Opcional) | Modelos balanceados `models/trainers/*_best_model.pkl` de 3 arquiteturas diferentes (XGBoost, LightGBM, MLP) + limiares otimizados + Conjunto Real `data/processed/X_test.csv` -- O Simulador simula o ambiente Real-Time de banco: a cada milissegundo as features batem no comitê de inferência e a decisão majoritária é tomada. O gabarito (y_test) e usado apenas para mostrar se o comitê acertou e quantificar retorno e atrito financeiro gerado para o cliente. | `predict_ensemble.py`, `simulate_production.py`                                                                                         | Recebe as transações como streaming de micro-serviço e orquestra a Ingestão no Comitê de 3 modelos de ML. Se `Fraud_Votes >= 2`: BLOQUEIO. Se `Fraud_Votes == 1` mas o votante for o _Champion de Negocio_ (LightGBM): CAI PARA REVISÃO MANUAL devido ao altíssimo risco estatístico. Qualquer outra coisa é aprovada para não gerar atrito.                                                      | Saida Dashboard Console interativa com emoji para rastreio transacional unitário da tomada de decisão dos modelos. Finaliza salvando na persistência física o artefato executivo `reports/simulation_summary.txt`, que resume todas as TN, TP, Falsos positivos/negativos e traduz o lucro evitado em $$ vs atrito operacional da operação global.                                                                                                                                                                |

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
|   |       |-- reg_log_model.py       # Treinamento Logistic Regression
|   |       |-- decision_tree_model.py # Treinamento Decision Tree
|   |       |-- random_forest_model.py # Treinamento Random Forest
|   |       |-- xgboost_model.py       # Treinamento XGBoost
|   |       |-- mlp_model.py           # Treinamento MLP (Rede Neural)
|   |       |-- isolation_forest_model.py # Treinamento Isolation Forest
|   |       |-- lightgbm_model.py      # Treinamento LightGBM (Campeao de Precisao)
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
|   |-- simulation_summary.txt # Relatorio executivo financeiro (ROI) do Emsemble
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
| **Metodologia**      | Stratified 5-Fold CV com pipeline SMOTE _dentro_ de cada fold                                                                                       |
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

## 4.3 Fluxo de Inferencia (Predicao com Ensemble)

```
simulate_production() -> predict_ensemble() :
    1. predictor = FraudEnsemblePredictor() carrega modelos .pkl e threshold.txt (ex: LightGBM, XGBoost, MLP)
    2. X_test, y_test são carregados e embaralhados.
    3. Para cada transacao fornecida em streaming:
       a. Calcula proba para os 3 modelos e computa contra os seus 3 thresholds otimizados.
       b. Majority Vote Logic:
          - Se Fraud_Votes >= Majority_Threshold: BLOQUEIO
          - Se Fraud_Votes == 1 AND Votante == 'LightGBM': REVISÃO MANUAL
          - Senao: APROVADO
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
3. **Implementar SHAP/LIME** para explicabilidade (placeholder existe em `simulate_production.py`)

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
