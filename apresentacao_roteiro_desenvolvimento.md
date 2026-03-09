# Roteiro de Fala — Seção de Desenvolvimento (Slides 11–15)
## Fraud Sentinel · Mineração de Dados

> **Como usar este roteiro:** Cada slide tem um bloco de fala principal + pontos de apoio caso a banca pergunte.
> Fale de forma natural — não leia. Use os tópicos como guia de memória.
> **Slides 12 e 15** estão no roteiro mas ainda não estão no PDF — adicionar depois.

---

## Slide 11 — Metodologia e Pipeline

### O que está no slide
Fluxograma do pipeline completo: Dados → EDA → Feature Engineering → Treino → Avaliação → Ensemble → Produção

### O que falar

> *"Antes de entrar nos detalhes, deixa eu dar uma visão geral de como o sistema funciona como um todo."*

O Fraud Sentinel foi desenvolvido seguindo a metodologia **CRISP-DM** — o padrão acadêmico e industrial para projetos de mineração de dados. Ela divide o trabalho em fases bem definidas: entendimento do negócio, entendimento dos dados, preparação, modelagem, avaliação e implantação. Cada etapa do nosso pipeline corresponde diretamente a uma fase do CRISP-DM.

Na prática, o sistema foi construído como um **pipeline modular**: cada etapa tem uma responsabilidade única e se comunica com as demais exclusivamente por **artefatos gravados em disco** — CSVs, modelos pickle, JSONs. Isso garante reprodutibilidade total — qualquer experimento pode ser retomado do ponto onde parou sem re-executar o pipeline inteiro.

O orquestrador é o `main.py`, com interface de linha de comando que permite pular etapas, escolher modelos e limitar o tamanho da base. Isso foi fundamental para iterarmos rápido nos experimentos.

> *"Cada bloco desse fluxo vira um slide. Vou detalhar cada um."*

### Se a banca perguntar
- **"Por que CRISP-DM e não outra metodologia?"** → CRISP-DM é iterativo e orientado ao negócio — não é só uma sequência técnica. Ele força a equipe a voltar à fase de negócio para validar se as decisões de modelagem fazem sentido para o banco. Isso foi o que nos levou a priorizar Recall em vez de Acurácia.
- **"O pipeline roda em produção assim?"** → O pipeline como está é voltado para treinamento e experimentação. Para produção real, os modelos já treinados (arquivos `.pkl`) seriam servidos via API. A simulação que fizemos emula esse comportamento.
- **"Por que artefatos em disco e não em memória?"** → Reprodutibilidade e auditoria. Você consegue inspecionar exatamente o que entrou e saiu de cada etapa, e qualquer cientista de dados pode retomar o experimento de qualquer ponto.

---

## Slide 12 — Dataset & Desafio Central
> ⚠️ *Este slide ainda não está no PDF — adicionar*

### O que está no slide
Volume do dataset, divisão treino/teste, desbalanceamento 1,11% fraude, destaque da armadilha da acurácia

### O que falar

> *"O dataset que usamos é o BAF Suite, publicado no NeurIPS 2022, que simula solicitações reais de abertura de conta bancária."*

São **1 milhão de registros**, com **32 atributos** — uma mistura de dados comportamentais da sessão, dados socioeconômicos e dados de dispositivo. A qualidade da base é excepcional: zero valores nulos, zero duplicatas. Isso eliminou uma etapa inteira de limpeza massiva.

A divisão foi **80/20 estratificada** — isso é importante. Estratificada significa que a proporção de fraude foi preservada nos dois conjuntos. Se você fizer um split aleatório simples num dataset com 1,11% de fraude, corre o risco de o conjunto de teste ter proporções muito diferentes, o que distorce as métricas.

Agora, o desafio central: **apenas 1,11% das instâncias são fraudes**. Isso parece pouco, mas são 11 mil casos numa base de 1 milhão.

Isso cria uma **armadilha estatística**: um modelo que simplesmente aprova todo mundo alcança 98,9% de acurácia — e é completamente inútil para o banco. Ele deixou todos os fraudadores passarem.

> *"Por isso a acurácia foi descartada como métrica principal desde o início. Trabalhamos com Recall, Precisão e F1-score."*

### Se a banca perguntar
- **"O que é Recall na prática aqui?"** → De todas as fraudes reais na base de teste, quantas o modelo conseguiu detectar? Recall de 30% significa que pegamos 30 de cada 100 fraudes reais.
- **"E Precisão?"** → Quando o modelo acusa fraude, quantas vezes ele está certo? Precisão de 20% significa que de cada 100 alertas, 20 eram fraude real e 80 eram clientes legítimos bloqueados por engano.
- **"Por que não usar oversampling para equilibrar o dataset?"** → Testamos. Vamos falar disso no próximo slide.

---

## Slide 13 — Análise Exploratória Orientada a Dados

### O que está no slide
Teste Mann-Whitney, ranking de Mutual Information, análise de risco categórico

### O que falar

> *"Antes de treinar qualquer modelo, precisávamos entender o que os dados estavam nos dizendo. Fizemos uma EDA estruturada em três frentes."*

**Primeira frente — Significância Estatística:**
Usamos o **Teste U de Mann-Whitney** para verificar se os padrões dos fraudadores são genuinamente diferentes dos clientes legítimos. Optamos por esse teste porque a maioria das variáveis não segue distribuição normal — e o teste t clássico assume normalidade. O Mann-Whitney é não-paramétrico: ele ranqueia os dados e verifica se um grupo sistematicamente ocupa posições diferentes do outro.

Resultado: **24 de 26 variáveis numéricas** são estatisticamente distintas entre os dois grupos. Isso validou que existe sinal real nos dados.

**Segunda frente — Mutual Information:**
Correlação de Pearson só enxerga relações lineares. O **MI** mede a redução de incerteza sobre a fraude ao conhecermos uma variável — captura qualquer tipo de dependência, linear ou não.

Os três atributos mais informativos foram:
- `device_distinct_emails_8w` — mesmo dispositivo com vários e-mails em 8 semanas. Isso é robô.
- `email_is_free` — e-mails descartáveis são fáceis de gerar em lote.
- `keep_alive_session` — sessão curtíssima, sem engajamento real.

> *"Esses três atributos pintam o perfil digital do fraudador moderno — ele não age como humano."*

**Terceira frente — Risco Categórico:**
Olhamos a taxa de fraude dentro de cada categoria. O grupo habitacional `BA` tem **3,75% de fraude** — mais de três vezes a média global. Usuários Windows Desktop têm **2,47%** — mais que o dobro da média. Esses nichos viraram features binárias diretas no pipeline.

### Se a banca perguntar
- **"Por que não usar correlação de Pearson?"** → Pearson mede só relações lineares monotônicas. Muitas das nossas variáveis são binárias ou categóricas — a relação com fraude é não-linear. O MI captura qualquer tipo de dependência estatística, baseado em redução de entropia.
- **"E as 2 variáveis que não foram significativas?"** → `session_length_in_minutes` e `device_fraud_count`. A segunda tem cardinalidade 1 — todos os valores são iguais, então não há variação para distinguir nada.
- **"A EDA influenciou diretamente o modelo?"** → Sim, diretamente. As features criadas no passo seguinte são baseadas nos resultados da EDA — os nichos de risco categórico viraram variáveis binárias, e as interações de MI orientaram a criação de features compostas.

---

## Slide 14 — Feature Engineering (EDAFeatureEngineer)

### O que está no slide
Transformações aplicadas, RobustScaler + IQR Clipping, criação de features, descarte do SMOTE, adoção de Cost-Sensitive Learning + Random Undersampling

### O que falar

> *"Com os insights da EDA em mãos, construímos a classe EDAFeatureEngineer — ela centraliza todo o pré-processamento de forma que rode corretamente dentro do cross-validation, sem vazamento de dados."*

Se você normaliza os dados antes de separar os folds de validação, está contaminando o teste com informação do treino — isso se chama Data Leakage. O pipeline do scikit-learn previne isso: o fold de teste nunca vê os parâmetros aprendidos no treino.

O que a classe faz:

**1. Transformação categórica:** One-Hot Encoding nas 5 variáveis categóricas, necessário para algoritmos de gradiente.

**2. Normalização:** **RobustScaler combinado com IQR Clipping**. O Scaler padrão explode quando há outliers extremos — e temos isso: 24% da base tem `proposed_credit_limit` como outlier. O RobustScaler usa a mediana, não a média, sendo imune a esses valores. O Clipping limita os valores antes da escala.

**3. Criação de features:** 5 variáveis binárias de nicho de risco derivadas da EDA — por exemplo, um flag para o grupo habitacional `BA` — mais interações entre variáveis digitais orientadas pelo MI.

**4. Otimização de Memória:** Downcasting de tipos (`float64 → float32`, `int64 → int8`). O dataset bruto ocupa muito mais memória do que precisa — essa otimização foi essencial para o GridSearch não estourar RAM.

Agora, a decisão mais importante dessa etapa — **o combate ao desbalanceamento**. Testamos três abordagens:

> *"O SMOTE foi descartado. Ele cria fraudes sintéticas interpolando pontos vizinhos. Essas fraudes artificiais são perfeitas demais — não existem em produção. Isso causou explosão de falsos positivos quando avaliamos no conjunto real."*

Adotamos **duas estratégias combinadas**:
- **Cost-Sensitive Learning:** Nenhum dado é adicionado ou removido. O XGBoost recebeu `scale_pos_weight=90` — o erro em classificar uma fraude como legítima pesa 90 vezes mais na função de perda.
- **Random Undersampling:** Parametrizado via CLI com `--undersampling-ratio 0.5`, reduz a proporção de legítimos no treino de forma controlada. Importante: opera **apenas nos folds de treino**, nunca no conjunto de teste — que permanece com o desbalanceamento real do mundo.

### Se a banca perguntar
- **"O que é Data Leakage?"** → É quando informação do futuro contamina o treino. Se você calcula a média de uma variável em toda a base e usa para normalizar, o modelo de treino "viu" dados do teste indiretamente.
- **"Por que usar undersampling E cost-sensitive ao mesmo tempo?"** → São estratégias complementares. O cost-sensitive atua na função de perda, penalizando o erro. O undersampling reduz o ruído da classe majoritária durante o treino. Juntos, compensam o desbalanceamento de formas distintas.
- **"O que é scale_pos_weight=90?"** → É aproximadamente a razão entre negativos e positivos: 990.000 legítimos / 11.000 fraudes ≈ 90. Instrui o gradiente a dar peso 90× maior ao erro de classificar uma fraude como legítima.

---

## Slide 15 — Ciclo Experimental & Benchmarking
> ⚠️ *Este slide ainda não está no PDF — adicionar*

### O que está no slide
4 famílias de modelos testados: Logistic Regression (baseline), Random Forest (bagging), XGBoost + LightGBM (boosting), MLP (deep learning)

### O que falar

> *"Com o pré-processamento definido, executamos um ciclo experimental sistemático. Não apostamos num único modelo — testamos quatro famílias matemáticas, cada uma com uma perspectiva distinta sobre os dados."*

Todos os modelos seguiram o mesmo protocolo de rigor científico: **busca de hiperparâmetros com validação cruzada estratificada de 3 folds**, seguida de retreino no dataset completo com os melhores parâmetros encontrados. Para modelos mais complexos como o LightGBM, usamos **RandomizedSearchCV com 60 iterações** — exploração aleatória do espaço de hiperparâmetros, mais eficiente que a busca exaustiva quando o espaço é muito grande.

**Regressão Logística — o Baseline:**
É o ponto de partida obrigatório em qualquer experimento sério. Se um modelo simples já resolve bem, não precisamos de complexidade. Resultado: **F1 = 0,212** — surpreendentemente competitivo. Isso prova que a Feature Engineering fez um trabalho excelente: injetou tanto sinal nos dados que até um modelo linear consegue captá-lo.

**Random Forest — Bagging:**
Cria centenas de árvores em paralelo e vota por consenso. Resolve o overfitting de uma árvore única. Resultado: **F1 = 0,189** — ficou abaixo do baseline linear. O motivo: o Bagging vota por popularidade e não aprende com os erros anteriores. Fraudadores sofisticados que ficam no meio da distribuição passam pelo consenso.

**XGBoost e LightGBM — Boosting:**
Aqui a lógica é oposta ao Random Forest. As árvores são construídas **em série**: cada árvore nova foca exatamente nos erros que a anterior cometeu. Isso é especialmente poderoso em dados desbalanceados.

- **XGBoost**: F1 = 0,221 · PR-AUC = 0,147 · ROC-AUC = 0,887
- **LightGBM**: F1 = 0,231 · PR-AUC = 0,158 · ROC-AUC = 0,893 — melhor modelo individual

**MLP — Rede Neural:**
Captura padrões não-lineares em alta dimensão. Em dados tabulares, geralmente perde para o Boosting em F1. Mas tem uma característica estratégica: **maior Precisão global (24,69%)** e **menor número de falsos positivos (1.211)** entre todos os modelos. Isso a tornou candidata perfeita para um papel específico no Ensemble.

**Descartados:**
- **Decision Tree:** F1 = 0,137. Classifica quase tudo como legítimo — minimizar a impureza global é matematicamente mais fácil do que detectar 1% de fraude.
- **Isolation Forest:** PR-AUC = 0,012. Pressupõe que fraude é um outlier geométrico. Mas fraudadores de onboarding imitam clientes normais propositalmente — não são outliers espaciais.

### Se a banca perguntar
- **"Por que não usar apenas o LightGBM?"** → Porque cada modelo erra em lugares diferentes. O Ensemble combina perspectivas para compensar as fraquezas individuais.
- **"O que é RandomizedSearchCV?"** → Ao invés de testar todas as combinações de hiperparâmetros (GridSearch exaustivo), ele sorteia um número fixo de combinações do espaço. Com 60 iterações, explorou bem o espaço sem custo computacional proibitivo.
- **"Por que o Random Forest ficou abaixo do baseline?"** → Pelo problema do Bagging com desbalanceamento extremo: amostras aleatórias com 1% de fraude frequentemente geram árvores que nunca viram nenhuma fraude no subset de treino delas.

---

## Slide 16 — Estratégia Final: Ensemble & Threshold

### O que está no slide
Regras do Ensemble (BLOQUEIO / REVISÃO / APROVADO), fórmula do threshold, bloco de monitoramento

### O que falar

> *"O passo final do desenvolvimento foi combinar os três melhores modelos num comitê de decisão — e calibrar os limiares de cada um individualmente."*

**Por que Ensemble?**
Cada modelo carrega o viés da sua matemática. O LightGBM é agressivo na detecção — alto Recall, mas mais falsos alarmes. A MLP é conservadora — alta Precisão, mas deixa passar alguns casos. Se apostamos em um só, herdamos suas fraquezas. O Ensemble dilui esses vieses com governança.

Os três modelos do comitê são **LightGBM** (varredura ampla), **XGBoost** (validador tático) e **MLP** (filtro de precisão).

**As regras do comitê:**

*Regra 1 — Maioria:* Se 2 ou 3 modelos acusam fraude acima do seu threshold individual — **BLOQUEIO automático**. O consenso de dois algoritmos com perspectivas matemáticas diferentes é extremamente robusto.

*Regra 2 — Veto da MLP:* Se só a Rede Neural alerta, mas os dois boosters aprovaram — **REVISÃO HUMANA**. A MLP tem a maior Precisão individual do sistema. Não faz sentido ignorar esse sinal isolado, mas também não faz sentido bloquear com base num único voto.

*Regra 3 — Absolvição:* Se apenas um booster alerta e a MLP aprova — **APROVADO**. A MLP funciona como filtro de qualidade final.

> *"Esse design respeita o custo assimétrico dos erros: bloquear um cliente bom tem custo — o banco perde o negócio. Deixar um fraudador passar também tem custo. O Ensemble equilibra isso com regras explícitas."*

**Threshold Tuning:**
O limiar padrão de 0,5 é inadequado aqui. Com `scale_pos_weight=90`, as distribuições de probabilidade dos modelos se deslocam — o ponto ótimo de corte não é o centro da escala.

Varremos 0,001 até 0,999 calculando o F1 em cada ponto da curva Precision-Recall. O pico — onde Recall e Precisão estão mais equilibrados — resultou em limiares **altos** para todos os modelos:

- **LightGBM: 0,89** — só alerta fraude com 89% de confiança
- **XGBoost: 0,997** — o mais exigente
- **Logistic Regression: 0,93**
- **Random Forest: 0,80**

Esses números são salvos em disco e carregados automaticamente na simulação de produção.

**Monitoramento:**
Cada experimento grava no `experiments_log.json` — modelo, hiperparâmetros, threshold, métricas. Matrizes de confusão e curvas Precision-Recall são geradas para cada modelo, permitindo auditoria completa.

### Se a banca perguntar
- **"O Ensemble que usaram é Stacking ou Voting?"** → Voting com regras de negócio customizadas. Não há meta-learner treinado como no Stacking. Isso foi uma escolha deliberada — regras determinísticas são mais interpretáveis e auditáveis em contexto bancário.
- **"Por que os thresholds são tão altos, próximos de 0,9?"** → Porque o `scale_pos_weight` desloca as probabilidades de saída dos modelos. O modelo aprende a ser muito sensível à fraude, então atribui probabilidades altas mesmo a casos borderline. O threshold alto compensa isso, exigindo alta confiança antes de bloquear.
- **"Como escolheram o threshold final?"** → Maximizando o F1-score na curva Precision-Recall do conjunto de validação. O F1 é a média harmônica de Precisão e Recall — pune modelos com desequilíbrio extremo entre as duas métricas.
- **"E se o banco quiser ajustar sem retreinar?"** → Para isso existe o `force_precision.py` — você passa o piso mínimo de Precisão desejado e o script recalcula o threshold automaticamente, varrendo a curva.

---

## Dicas Gerais de Apresentação

- **Tempo sugerido por slide:** 2–3 minutos cada
- **Transições naturais:** Use as frases de gancho entre slides (marcadas com *itálico*)
- **Não leia os bullets do slide** — eles são lembretes visuais para a banca, não seu roteiro
- **Se travar:** volte para o princípio do slide e diga *"O ponto central aqui é..."*
- **Para perguntas difíceis:** *"Boa pergunta — isso foi uma decisão deliberada. O que aconteceu foi..."*
- **Nunca cite um número que você não consegue justificar** — prefira dizer "na ordem de" a dar um número errado com precisão falsa
