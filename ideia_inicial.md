# Projeto Final: FraudSentinel

## Mineração de Dados Aplicada a Finanças (Detecção de Fraudes Bancárias)

**Dataset Selecionado:** Bank Account Fraud (BAF) Suite (NeurIPS 2022)
**Problema:** Identificação de contas bancárias fraudulentas no ato da abertura.
**Tarefa de Mineração:** Classificação Binária (Supervisionada).

---

## 1. Definição do Problema e Tarefa

Embora o objetivo final seja detectar anomalias, a tarefa formal selecionada é a **Classificação**, visto que o dataset possui dados rotulados (_labeled data_), onde:

- `0`: Conta Legítima (Classe Majoritária)
- `1`: Fraude (Classe Minoritária)

O objetivo é treinar um modelo capaz de aprender os padrões de comportamento e atributos sociodemográficos de fraudadores históricos para prever a probabilidade de fraude em novas solicitações (`predict_proba`).

---

## 2. Etapa I: Pré-processamento (A Fundação)

Devido à natureza complexa do _BAF Suite_, que mistura dados numéricos, categóricos e um forte desbalanceamento, esta etapa é crítica.

### A. Tratamento de Variáveis Categóricas (Encoding)

Os algoritmos de Machine Learning exigem entradas numéricas. O dataset possui campos como `payment_type`, `housing_status`, `source` e `device_os`.

- **Técnica:** _One-Hot Encoding_.
- **Aplicação:** Transformar colunas categóricas em vetores binários.
  - _Exemplo Prático:_ A coluna `device_os` (Sistema Operacional) será desmembrada em `device_os_windows`, `device_os_linux`, `device_os_mac`.
  - _Justificativa:_ Isso permite que o modelo identifique correlações específicas, como o uso desproporcional de Linux/X11 em farms de bots para fraude.

### B. Normalização de Dados (Feature Scaling)

As variáveis numéricas possuem escalas discrepantes (ex: `income` vs `age`).

- **Técnica:** _MinMax Scaler_ ou _Standard Scaler_.
- **Aplicação:** Normalizar colunas como `income` (Renda Anual), `age` (Idade) e `credit_risk_score` para um intervalo comum (0 a 1 ou distribuição normal).
- **Justificativa:** Essencial para garantir a convergência correta dos algoritmos e evitar que variáveis com magnitudes maiores dominem o cálculo de importância.

### C. Tratamento de Desbalanceamento (Handling Imbalance)

Fraudes são eventos raros (espera-se < 5% do dataset). Modelos tradicionais tendem a enviesar para a classe majoritária (legítima).

- **Técnica:** _SMOTE (Synthetic Minority Over-sampling Technique)_.
- **Aplicação:** Geração de exemplos sintéticos da classe "Fraude" (1) no conjunto de _treino_ (apenas no treino, nunca no teste).
- **Justificativa:** O SMOTE cria "novas" fraudes baseadas na interpolação vetorial de fraudes existentes, ensinando o modelo a generalizar o padrão da anomalia em vez de apenas decorar os exemplos repetidos.

---

## 3. Etapa II: Mineração de Dados (Modelagem)

Para atender ao rigor acadêmico, não utilizaremos apenas um algoritmo, mas uma abordagem comparativa entre duas famílias distintas de aprendizado de máquina.

### Técnica 1: Random Forest (Baseline Robusto)

- **Família:** _Bagging (Bootstrap Aggregating)_.
- **Descrição:** Um ensemble de múltiplas Árvores de Decisão independentes.
- **Por que usar:** É altamente robusto contra _overfitting_, lida bem com dados tabulares de alta dimensionalidade (pós-encoding) e fornece uma base sólida de comparação.

### Técnica 2: XGBoost ou LightGBM (Estado da Arte)

- **Família:** _Gradient Boosting_.
- **Descrição:** Constrói árvores sequencialmente, onde cada nova árvore foca em corrigir os erros residuais da anterior.
- **Por que usar:** É o padrão da indústria ("Gold Standard") para competições de dados e detecção de fraudes devido à sua velocidade e capacidade de capturar padrões não-lineares complexos que o Random Forest pode perder.

### Estratégia de Validação

- **Cross-Validation:** Utilização de _Stratified K-Fold_ (ex: 5 folds) para garantir que todas as subdivisões de treino e teste mantenham a proporção original de fraudes, garantindo a validade estatística dos resultados.

---

## 4. Etapa III: Pós-processamento (Avaliação e Interpretação)

Esta etapa foca na tradução dos resultados matemáticos em valor de negócio e insights explicáveis.

### A. Métricas de Desempenho (Além da Acurácia)

Como o dataset é desbalanceado, a "Acurácia" é uma métrica enganosa. O foco será em:

- **Recall (Sensibilidade):** A métrica primária. Mede a capacidade do modelo de encontrar _todas_ as fraudes. (Custo do Falso Negativo > Custo do Falso Positivo).
- **Precision:** Mede a qualidade do alerta de fraude (quantos alertas eram reais).
- **F1-Score:** A média harmônica entre Precision e Recall.
- **AUC-ROC:** A capacidade geral de discriminação do modelo.

### B. Visualização de Erros

- **Matriz de Confusão:** Para visualizar explicitamente os Falsos Negativos (fraudes que passaram) e Falsos Positivos (clientes legítimos bloqueados).

### C. Explicabilidade (XAI - Explainable AI)

- **Técnica:** _SHAP Values (SHapley Additive exPlanations)_.
- **Aplicação:** Gerar gráficos que explicam _o porquê_ de uma decisão.
  - _Exemplo de Insight:_ "O modelo aumentou a probabilidade de fraude em 80% porque a variável `name_email_similarity` está abaixo de 0.2 E a variável `device_os` é igual a Linux".
- **Justificativa:** Atende ao requisito de "análise das características das técnicas" solicitado pelo professor, abrindo a "caixa preta" do modelo.

---

## 5. Sugestão de Texto para o Relatório (Metodologia)

> _"Para abordar o problema de fraude em aberturas de conta, selecionou-se a tarefa de **Classificação Binária**. A metodologia seguiu o pipeline padrão de KDD (Knowledge Discovery in Databases):_
>
> 1. **\*Pré-processamento:** Realizou-se a limpeza de dados e a aplicação de One-Hot Encoding para variáveis categóricas (como Sistema Operacional e Status de Moradia). Para mitigar o viés causado pela raridade dos eventos fraudulentos, utilizou-se a técnica **SMOTE** (Synthetic Minority Over-sampling Technique) exclusivamente no conjunto de treino, garantindo o aprendizado de fronteiras de decisão mais robustas.\*
> 2. **\*Mineração:** Foi conduzido um treinamento comparativo entre algoritmos de Bagging (**Random Forest**) e Boosting (**XGBoost**), otimizados via validação cruzada estratificada (Stratified K-Fold) para assegurar a generalização do modelo.\*
> 3. **\*Pós-processamento:** A avaliação de desempenho priorizou a maximização do **Recall** (sensibilidade), visando a minimização de perdas financeiras. Adicionalmente, empregou-se a técnica de **SHAP Values** para garantir a interpretabilidade das decisões algorítmicas, permitindo identificar quais atributos socioeconômicos ou digitais (ex: Renda vs. Tipo de Dispositivo) exercem maior influência na probabilidade de fraude."\*

https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
