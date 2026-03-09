# Roteiro de Fala — Seção de Desenvolvimento (Slides 11–16)
## Fraud Sentinel · Mineração de Dados

> **Como usar este roteiro:** Cada slide tem um bloco de fala principal + pontos de apoio caso a banca pergunte.
> Fale de forma natural — não leia. Use os tópicos como guia de memória.

---

## Slide 11 — Metodologia e Pipeline

### O que está no slide
Fluxograma do pipeline completo: Dados → EDA → Feature Engineering → Treino → Avaliação → Ensemble → Produção

### O que falar

> *"Antes de entrar nos detalhes, deixa eu dar uma visão geral de como o sistema funciona como um todo."*

O Fraud Sentinel foi construído como um **pipeline modular**, onde cada etapa tem uma responsabilidade única e não depende diretamente do código das outras. Elas se comunicam exclusivamente por **artefatos gravados em disco** — CSVs, arquivos pickle, JSONs. Isso garante que qualquer etapa possa ser reexecutada de forma isolada sem quebrar o resto.

A entrada é o dataset bruto. A saída final é o comitê de modelos tomando decisões de bloqueio ou aprovação em tempo real.

O orquestrador de tudo isso é o `main.py`, que expõe uma **interface de linha de comando** — você consegue pular etapas, escolher quais modelos treinar, limitar o tamanho da base. Isso foi essencial para iterarmos rápido nos experimentos.

> *"Cada bloco desse fluxo vira um slide. Vou detalhar cada um."*

### Se a banca perguntar
- **"Por que artefatos em disco e não em memória?"** → Reprodutibilidade. Qualquer experimento pode ser retomado do ponto onde parou, sem re-executar todo o pipeline. Também facilita auditoria — você consegue inspecionar exatamente o que entrou e saiu de cada etapa.
- **"O pipeline roda em produção assim?"** → O pipeline como está é voltado para treinamento e experimentação. Para produção real, os modelos já treinados (arquivos `.pkl`) seriam servidos via API. A simulação que fizemos emula esse comportamento.

---

## Slide 12 — Dataset & Desafio Central

### O que está no slide
Volume do dataset, divisão treino/teste, desbalanceamento 1,11% fraude, destaque da armadilha da acurácia

### O que falar

> *"O dataset que usamos é o BAF Suite, publicado no NeurIPS 2022, que simula solicitações reais de abertura de conta bancária."*

São **1 milhão de registros**, com **32 atributos** — uma mistura de dados comportamentais da sessão, dados socioeconômicos e dados de dispositivo. A qualidade da base é excepcional: zero valores nulos, zero duplicatas. Isso eliminou uma etapa inteira de limpeza massiva.

A divisão foi **80/20 estratificada** — isso é importante. Estratificada significa que a proporção de fraude foi preservada nos dois conjuntos. Se você fizer um split aleatório simples num dataset com 1,11% de fraude, corre o risco de o conjunto de teste ter proporções bem diferentes, o que distorce as métricas.

Agora, o desafio central: **apenas 1,11% das instâncias são fraudes**. Isso parece pouco, mas é 11 mil casos numa base de 1 milhão.

O problema é que isso cria uma **armadilha estatística brutal**: um modelo que simplesmente aprova todo mundo alcança 98,9% de acurácia — e é completamente inútil para o banco. Ele deixou todos os fraudadores passarem.

> *"Por isso a acurácia foi descartada como métrica principal desde o início. Trabalhamos com Recall, Precisão e F1-score, que são métricas que efetivamente capturam o comportamento do modelo frente à classe minoritária."*

### Se a banca perguntar
- **"Por que não usar oversampling para equilibrar o dataset?"** → Testamos. O SMOTE gerou fraudes artificiais muito "ideais" que não existem em produção, causando excesso de falsos positivos. Vamos falar disso no próximo slide.
- **"O que é Recall na prática aqui?"** → Recall é: de todas as fraudes reais que existem na base de teste, quantas o modelo conseguiu detectar? Se o Recall é 30%, o modelo pegou 30 de cada 100 fraudes reais.
- **"E Precisão?"** → Precisão é: quando o modelo grita 'fraude', quantas vezes ele está certo? Uma Precisão de 20% significa que de cada 100 alertas, 20 eram fraude real e 80 eram clientes legítimos sendo bloqueados por engano.

---

## Slide 13 — Análise Exploratória (EDA)

### O que está no slide
Teste Mann-Whitney, ranking de Mutual Information, análise de risco categórico

### O que falar

> *"Antes de treinar qualquer modelo, precisávamos entender o que os dados estavam nos dizendo. Fizemos uma EDA bem estruturada com três frentes."*

**Primeira frente — Significância Estatística:**
Usamos o **Teste U de Mann-Whitney** para verificar se os padrões dos fraudadores são genuinamente diferentes dos clientes legítimos. Optamos por esse teste porque a maioria das variáveis não segue distribuição normal — e o teste t clássico que se aprende na faculdade assume normalidade. O Mann-Whitney é não-paramétrico: ele ranqueia os dados e verifica se um grupo sistematicamente ocupa posições diferentes do outro.

Resultado: **24 de 26 variáveis numéricas** são estatisticamente distintas entre os dois grupos. Isso validou que existe sinal real nos dados para os modelos aprenderem.

**Segunda frente — Mutual Information:**
Correlação de Pearson só enxerga relações lineares. No nosso contexto, muitas relações são não-lineares — por exemplo, a combinação de e-mail gratuito com dispositivo específico indica fraude, mas cada variável isolada não diz nada. O **MI** mede a redução de incerteza sobre a fraude ao conhecermos uma variável.

Os três atributos mais informativos foram:
- `device_distinct_emails_8w` — o mesmo dispositivo sendo usado com vários e-mails em 8 semanas. Isso é robô.
- `email_is_free` — e-mails descartáveis são fáceis de gerar em lote.
- `keep_alive_session` — sessão curtíssima, sem engajamento. Uma pessoa abrindo conta de verdade demora mais.

> *"Esses três atributos pintam o perfil digital do fraudador moderno — ele não age como humano."*

**Terceira frente — Risco Categórico:**
Olhamos a taxa de fraude dentro de cada categoria. Descobrimos nichos perigosos: o grupo habitacional `BA` tem **3,75% de taxa de fraude** — mais de três vezes a média global. Um usuário Windows Desktop tem mais que o dobro da média. Esses nichos viraram features binárias diretas no pipeline.

### Se a banca perguntar
- **"Por que não usar correlação de Pearson?"** → Pearson mede só relações lineares monotônicas. Muitas das nossas variáveis são binárias ou categóricas, e a relação com fraude é não-linear. O MI captura qualquer tipo de dependência estatística.
- **"E as 2 variáveis que não foram significativas?"** → `session_length_in_minutes` e `device_fraud_count`. A segunda tem cardinalidade 1 — todos os valores são iguais, então não há variação para distinguir nada.

---

## Slide 14 — Feature Engineering (EDAFeatureEngineer)

### O que está no slide
Transformações aplicadas, RobustScaler + IQR Clipping, criação de features, descarte do SMOTE, adoção de Cost-Sensitive Learning

### O que falar

> *"Com os insights da EDA em mãos, construímos uma classe chamada EDAFeatureEngineer que centraliza todo o pré-processamento de forma que ele seja aplicado corretamente dentro do cross-validation — sem vazamento de dados."*

Essa parte é crítica em MLOps: se você normaliza os dados antes de separar os folds de validação, você está contaminando o teste com informação do treino. Toda a nossa engenharia de features roda **dentro do pipeline do scikit-learn**, garantindo que o fold de teste nunca veja os parâmetros aprendidos no treino.

O que a classe faz:

**1. Transformação categórica:** One-Hot Encoding nas 5 variáveis categóricas. Necessário porque algoritmos de gradiente não operam sobre strings.

**2. Normalização:** Aplicamos **RobustScaler combinado com IQR Clipping**. O Scaler padrão explode quando há outliers extremos — e temos isso: 24% da base tem `proposed_credit_limit` outlier. O RobustScaler usa a mediana, não a média, sendo imune a esses valores extremos. O Clipping limita os valores antes da escala.

**3. Criação de features:** Com base nos nichos que a EDA revelou, criamos 5 variáveis binárias novas — por exemplo, um flag indicando se o usuário pertence ao grupo habitacional `BA`. Também criamos interações entre variáveis digitais baseadas no MI.

**4. Downcasting:** Convertemos `float64` para `float32` e `int64` para `int8` onde possível. O dataset bruto ocupa muito mais memória do que precisa — essa otimização foi essencial para o Random Forest e o GridSearch não estourarem RAM.

Agora, a decisão mais importante dessa etapa: **descartamos o SMOTE**.

> *"Testamos. O SMOTE cria fraudes sintéticas interpolando pontos vizinhos. O problema é que essas fraudes artificiais são 'redondas' e 'ideais' — não existem em produção. Isso causou excesso de falsos positivos quando avaliamos no conjunto real. Chamamos esse fenômeno de Fábrica de Fantasmas."*

A solução adotada foi o **Cost-Sensitive Learning**: nenhum dado é adicionado ou removido. Em vez disso, ensinamos ao modelo que errar numa fraude real custa muito mais do que errar num cliente bom. O XGBoost recebeu `scale_pos_weight=90` — o erro na classe fraude pesa 90 vezes mais na função de perda.

### Se a banca perguntar
- **"O que é Data Leakage?"** → É quando informação do futuro contamina o treino. Por exemplo, se você calcula a média de uma variável em toda a base e usa isso pra normalizar, o modelo de treino "viu" dados do teste. O pipeline do scikit-learn previne isso.
- **"Por que não usar SMOTE nunca?"** → Não é que nunca funciona. Em outros contextos, pode ser eficaz. No nosso caso específico, o dataset de produção é desbalanceado de verdade, então avaliar num conjunto artificialmente balanceado distorce as métricas.
- **"O que é scale_pos_weight=90?"** → É a razão entre o número de negativos e positivos: ~990.000 legítimos / ~11.000 fraudes ≈ 90. Isso instrui o gradiente a dar peso 90× maior ao erro de classificar uma fraude como legítima.

---

## Slide 15 — Ciclo Experimental & Benchmarking de Modelos

### O que está no slide
4 famílias de modelos testados: Logistic Regression (baseline), Random Forest (bagging), XGBoost + LightGBM (boosting), MLP (deep learning)

### O que falar

> *"Com o pré-processamento definido, executamos um ciclo experimental sistemático. Não apostamos em um único modelo — testamos quatro famílias matemáticas diferentes, cada uma com uma perspectiva distinta sobre os dados."*

**Regressão Logística — o Baseline:**
É o ponto de partida obrigatório. Se um modelo simples já resolve bem, não precisamos de complexidade. Ela traça hiperplanos lineares em 38 dimensões. O resultado foi surpreendentemente bom — F1 de 0,212 — o que prova que a nossa Feature Engineering fez um trabalho excelente: ela injetou tanto sinal nos dados que até um modelo linear consegue captá-lo.

**Random Forest — Bagging:**
Cria centenas de árvores de forma aleatória e paralela, votando por consenso. Resolve o problema de overfitting de uma árvore única. Melhorou o baseline — F1 de 0,189 no teste cego — mas bateu num teto. O problema: o Bagging vota por popularidade. Ele não aprende com os erros anteriores. Fraudadores sofisticados que ficam no meio da distribuição passam pelo consenso.

**XGBoost e LightGBM — Boosting:**
Aqui a lógica é oposta ao Random Forest. As árvores são construídas **em série**: cada árvore nova foca exatamente nos erros que a anterior cometeu. Isso é especialmente poderoso em dados desbalanceados — o modelo persegue os casos difíceis de forma iterativa.

O LightGBM usa histogramas para particionar, sendo muito mais rápido e focando em "exemplos difíceis" pela técnica GOSS. Foi o melhor modelo individual — F1 de 0,231.

**MLP — Rede Neural:**
Captura padrões topológicos em alta dimensão que os outros modelos não enxergam. Em dados tabulares, ela geralmente perde para o Gradient Boosting em ranking absoluto. Mas ela tem uma característica única: **alta Precisão**. Quando ela diz que é fraude, ela costuma estar certa. Isso a tornou o candidato perfeito para um papel específico no Ensemble.

Todos os modelos passaram pelo mesmo protocolo: **GridSearchCV com 3 folds estratificados**, seguido de retreino no dataset completo com os melhores hiperparâmetros encontrados.

### Se a banca perguntar
- **"Por que não usar apenas o LightGBM, que foi o melhor?"** → Porque cada modelo erra em lugares diferentes. O LightGBM pode falhar em padrões que a MLP captura. O Ensemble combina as perspectivas, reduzindo o erro total.
- **"Árvore de Decisão e Isolation Forest não aparecem?"** → Testamos e falharam. A Árvore de Decisão simples colapsou com F1=0 no teste — ela classifica tudo como legítimo porque isso minimiza a impureza global. A Isolation Forest teve AUC-PR de 0,025 — ela pressupõe que fraude é um outlier geométrico, mas fraudadores de onboarding imitam clientes normais propositalmente.
- **"O que é GridSearchCV?"** → É uma busca exaustiva sobre uma grade de hiperparâmetros. Para cada combinação, treina e valida com cross-validation. No final, escolhe a combinação com melhor ROC-AUC médio nos folds.

---

## Slide 16 — Ensemble com Veto & Threshold Tuning

### O que está no slide
Tabela de regras do Ensemble (BLOQUEIO / REVISÃO / APROVADO), fórmula do threshold, bloco de monitoramento

### O que falar

> *"O passo final do desenvolvimento foi combinar os três melhores modelos num comitê de decisão — e calibrar os limiares de cada um individualmente."*

**Por que Ensemble?**
Cada modelo carrega o viés da sua matemática. O LightGBM é agressivo na detecção — alto Recall, mas gera mais falsos alarmes. A MLP é conservadora — alta Precisão, mas deixa passar alguns fraudadores. Se apostamos em um só, herdamos suas fraquezas. O Ensemble dilui esses vieses.

**As regras do comitê — e por que cada uma existe:**

*Regra 1 — Maioria:* Se 2 ou 3 modelos acusam fraude acima do seu threshold individual, é **BLOQUEIO automático**. O consenso matemático de dois algoritmos de perspectivas diferentes é extremamente robusto.

*Regra 2 — Veto da MLP:* Se só a Rede Neural alerta, mas os boosters aprovaram, não bloqueamos — mandamos para **revisão humana**. A MLP tem Precisão próxima de 95% nos casos onde ela vota isolada. Não faz sentido ignorar esse sinal, mas também não faz sentido bloquear com base em um único voto.

*Regra 3 — Absolvição:* Se só um booster alerta e a MLP aprova, é **APROVADO**. A MLP funciona como filtro de qualidade.

> *"Esse design respeita o custo do erro de cada lado: bloquear um cliente bom tem custo — o banco perde o negócio. Deixar um fraudador passar também tem custo. O Ensemble foi desenhado para equilibrar isso com governança."*

**Threshold Tuning:**
O limiar padrão de 0,5 é arbitrário e inadequado aqui. Com `scale_pos_weight=90`, o XGBoost distorce as probabilidades de saída — as probabilidades de fraude ficam concentradas em valores muito baixos. Um corte em 0,5 perderia quase todos os casos.

Varremos 0,001 até 0,999 e calculamos o F1 em cada ponto. O pico do F1 — que é onde Recall e Precisão estão mais equilibrados — ficou entre **0,06 e 0,15** dependendo do modelo. Esse número é salvo em disco e carregado na inferência.

O gestor do banco pode ajustar esse threshold. Quer ser mais conservador? Baixa o threshold, aumenta o Recall, aceita mais falsos positivos. Quer menos atrito com clientes? Sobe o threshold, reduz falsos positivos, aceita perder algumas fraudes.

**Monitoramento:**
Cada experimento grava automaticamente no `experiments_log.json` — modelo, hiperparâmetros, threshold, métricas. Também são geradas matrizes de confusão e curvas Precision-Recall para cada modelo, permitindo auditoria completa de cada decisão de engenharia.

### Se a banca perguntar
- **"O Ensemble que usaram é Stacking ou Voting?"** → É um Voting com regras de negócio customizadas — não é um meta-learner treinado (Stacking), mas um conjunto de regras determinísticas sobre os votos. Isso dá mais controle e interpretabilidade para um contexto bancário.
- **"Como escolheram o threshold final?"** → Maximizando o F1-score na curva Precision-Recall do conjunto de validação. O F1 é a média harmônica de Precisão e Recall — pune modelos com desequilíbrio extremo entre as duas métricas.
- **"E se o banco quiser mudar o comportamento do sistema sem retreinar?"** → Exatamente para isso existe o `force_precision.py` — você passa o piso mínimo de Precisão que quer garantir e o script varre a curva e recalcula o threshold automaticamente, sem retreinar.

---

## Dicas Gerais de Apresentação

- **Tempo sugerido por slide:** 2–3 minutos cada → total da seção: ~15 minutos
- **Transições naturais:** Use as frases de gancho entre slides (marcadas com *itálico*)
- **Não leia os bullets do slide** — eles são lembretes visuais para a banca, não seu roteiro
- **Se travar:** volte para o princípio do slide e diga *"O ponto central aqui é..."*
- **Para perguntas difíceis:** *"Boa pergunta — isso foi uma decisão deliberada. O que aconteceu foi..."*
