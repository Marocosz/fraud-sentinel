import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE

# ==============================================================================
# ARQUIVO: build_features.py
#
# OBJETIVO:
#   Construir o pipeline de transformação de dados (Feature Engineering Automático).
#   Mapeia como atributos textuais e numéricos crus são recodificados em matrizes 
#   matemáticas purificadas para que os algoritmos consigam consumi-las.
#
# PARTE DO SISTEMA:
#   Módulo de Engenharia de Features MLOps (Preprocessing Stage).
#
# RESPONSABILIDADES:
#   - Aplicar feature engineering matemático baseado nos insights descobertos (EDA-driven).
#   - Excluir variáveis zeradas e lidar com falsos-positivos de sentinelas (-1).
#   - Identificar automaticamente colunas numéricas x categóricas via Introspection.
#   - Imputar falhas estruturais (NaNs) via Mediana ou 'missing'.
#   - Proteger a rede neural blindando extremos milionários através do RobustScaler.
#   - Transformar categorias textuais em matrizes densas (One-Hot).
#   - Entregar uma classe serializada (`Pipeline`) pronta para inferência/produção.
#
# INTEGRAÇÕES:
#   - Lê: `X_train.pkl` em memória viva (para calibrar as matrizes de escala e OneHot).
#   - Escreve: `preprocessor.joblib` (O motor de conversão exportado para a API de Inferência).
# ==============================================================================


class EDAFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer customizado do scikit-learn que aplica Feature Engineering
    orientado pelos insights da Analise Exploratoria (EDA).

    Este transformer e incluido como PRIMEIRO passo do Pipeline de cada modelo,
    garantindo que as mesmas transformacoes sejam aplicadas em treino, validacao
    cruzada e inferencia. Por ser um BaseEstimator, e serializado junto com o
    modelo via joblib.

    Transformacoes aplicadas (em ordem):
    -----------------------------------------------------------------------
    1. REMOCAO DE FEATURES SEM SINAL PREDITIVO
       - device_fraud_count: Variancia zero (constante = 0 em todos os registros).
         Cardinalidade = 1, MI Score = 0.0001, correlacao = NaN.
         Nao contribui para nenhum modelo.
       - session_length_in_minutes: MI Score = 0.0000 e Mann-Whitney NAO significativo
         (p = 0.163). Unica feature que falhou em ambos os testes de relevancia.

    2. TRATAMENTO DE VALORES SENTINELA (-1)
       O dataset BAF Suite usa -1 como proxy para "dado ausente/desconhecido"
       em 3 colunas. O pipeline original tratava -1 como valor numerico real,
       distorcendo calculos de mediana e escala. A correcao cria uma flag binaria
       indicando presenca do dado e substitui -1 por NaN (que o SimpleImputer
       trata com mediana dos valores reais).
       - prev_address_months_count: Mediana = -1 (>50% dos registros sao sentinela)
       - bank_months_count: Q1 = -1 (~25% dos registros sao sentinela)
       - device_distinct_emails_8w: Min = -1

    3. CLIPPING DE OUTLIERS EXTREMOS
       Features com >15% de outliers pelo metodo IQR recebem clipping nos
       percentis 1% e 99%, calculados no fit() (conjunto de treino). Isso
       reduz a influencia de valores extremos em modelos sensiveis a escala
       (LogReg, MLP) sem perder informacao de ordenacao para modelos de arvore.
       - proposed_credit_limit: 24.17% outliers
       - intended_balcon_amount: 22.27% outliers
       - bank_branch_count_8w: 17.52% outliers
       - prev_address_months_count: 15.73% outliers

    4. FLAGS BINARAS DE RISCO CATEGORICO
       O EDA identificou categorias com taxa de fraude muito acima da media (1.10%).
       Flags binarias explicitam esses segmentos para o modelo:
       - housing_status == 'BA': 3.75% fraude (3.4x a media)
       - employment_status == 'CC': 2.47% fraude (2.2x a media)
       - device_os == 'windows': 2.47% fraude (2.2x a media)
       - payment_type == 'AC': 1.67% fraude (1.5x a media)
       - source == 'TELEAPP': 1.59% fraude (1.4x a media)

    5. FEATURE INTERACTIONS (COMPORTAMENTO DIGITAL + DEMOGRAFICO)
       As top 3 features por MI Score sao todas digitais. As interacoes capturam
       padroes compostos que features individuais nao expressam.

       5a. digital_risk_score = email_is_free * device_distinct_emails_8w
           Hipotese: Email gratuito + multiplos emails no mesmo dispositivo
           e um forte indicador composto de fraude.

       5b. velocity_anomaly = velocity_6h / (velocity_4w + 1)
           Hipotese: Picos subitos de atividade (alta velocidade recente vs
           historica) indicam comportamento atipico associado a fraude.
           EDA indica que velocity_6h apresenta mudancas mais drasticas
           em registros fraudulentos.

       5c. credit_utilization = intended_balcon_amount / (proposed_credit_limit + 1)
           Hipotese: Ratio alto indica tentativa de usar todo o credito
           disponivel rapidamente, comportamento tipico de fraude.
           Ambas features tem >15% de outliers e correlacao com o target.

       5d. age_income_risk = customer_age * income
           Hipotese: Perfis demograficos especificos (jovens com renda alta
           ou idosos com renda muito baixa) podem indicar identidade falsa.
           Ambas tem correlacao significativa com fraude (p < 0.001).

       5e. multi_risk_flag = is_high_risk_housing * is_high_risk_os * email_is_free
           Hipotese: A combinacao simultanea de multiplos fatores de alto risco
           multiplica a probabilidade de fraude. Fraud Rate isolado:
           BA=3.75%, windows=2.47%, email_free=~1.5%. Combo deve ser >>3%.

       5f. phone_mismatch = phone_home_valid XOR phone_mobile_valid
           Hipotese: Ter APENAS um tipo de telefone valido (mas nao ambos e
           nao nenhum) pode indicar uso de numero temporario.
           phone_home_valid e phone_mobile_valid tem correlacoes opostas
           com fraude no EDA.

       5g. email_velocity_risk = email_is_free * velocity_6h
           Hipotese: Email gratuito combinado com alta velocidade de transacao
           indica atividade automatizada de fraude.

       5h. address_stability = current_address_months_count / (customer_age * 12 + 1)
           Hipotese: Fracao da vida no endereco atual. Valores muito baixos
           (mudancas recentes) combinados com outros fatores indicam risco.
           current_address_months_count tem correlacao de 0.048 com fraude.
    """

    # -------------------------------------------------------------------------
    # CONFIGURACAO (derivada do EDA)
    # -------------------------------------------------------------------------

    # Features a remover (variancia zero ou MI = 0 + Mann-Whitney nao significativo)
    FEATURES_TO_DROP = [
        'device_fraud_count',           # Variancia zero (constante = 0)
        'session_length_in_minutes',    # MI = 0, Mann-Whitney p = 0.163
    ]

    # Mapeamento de colunas com sentinela -1 -> nome da flag binaria
    SENTINEL_COLUMNS = {
        'prev_address_months_count': 'has_prev_address',
        'bank_months_count': 'has_bank_history',
        'device_distinct_emails_8w': 'has_device_emails',
    }

    # Colunas com >15% de outliers para aplicar clipping
    CLIP_COLUMNS = [
        'proposed_credit_limit',
        'intended_balcon_amount',
        'bank_branch_count_8w',
        'prev_address_months_count',
    ]

    # Mapeamento de categorias de alto risco -> nome da flag
    HIGH_RISK_CATEGORIES = {
        'housing_status': ('BA', 'is_high_risk_housing'),
        'employment_status': ('CC', 'is_high_risk_employment'),
        'device_os': ('windows', 'is_high_risk_os'),
        'payment_type': ('AC', 'is_high_risk_payment'),
        'source': ('TELEAPP', 'is_teleapp_source'),
    }

    def fit(self, X, y=None):
        """
        Aprende os limites estatísticos estritos do conjunto orgânico para uso futuro
        em inferências cegas.
        
        Por que existe:
        Calibrações de escala (como achar o percentil 1% e 99% para Clipping) DEVEM OCORRER 
        obrigatoriamente apenas nos dados de treino. Se calcularmos os percentis durante 
        o Data Ingestion, o teste sofre Data Leakage. Este método "trava" os percentis numéricos
        descobertos aqui, para aplicá-los passivamente no Transform depois.

        Recebe:
        X (pd.DataFrame): Dados de treinamento crús com anomalias e Outliers ainda vivos.

        Retorna:
        self: A própria instância calibada (mantendo padrão do scikit-learn).
        """
        # Otimização: Não clona todo o DataFrame em memória na inspeção
        X_work = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Trata sentinelas antes de calcular percentis (para nao incluir -1)
        for col in self.CLIP_COLUMNS:
            if col in X_work.columns:
                col_values = X_work[col].copy()
                if col in self.SENTINEL_COLUMNS:
                    col_values = col_values.replace(-1, np.nan)
                # Ignora NaN no calculo dos percentis
                self.clip_bounds_ = getattr(self, 'clip_bounds_', {})
                self.clip_bounds_[col] = (
                    float(col_values.quantile(0.01)),
                    float(col_values.quantile(0.99))
                )

        return self

    def transform(self, X, y=None):
        """
        Motor ativo do Custom Transformer. Recebe um dataset de produção e devolve
        a matriz purificada baseada nas lógicas rígidas extraídas do EDA.
        
        Por que existe:
        É o escudo anti-fraude que cria e transmuta as colunas em real-time, gerando fatores 
        cruzados (Interações Numéricas/Booleanas) antes que o Estimador veja a luz do dia.

        Recebe:
        X (pd.DataFrame): Payload da API Web ou DataFrame inteiro de validação cruzada.

        Retorna:
        pd.DataFrame X: A base transmutada inplace (ou nova), amplificada com ~7 novas 
        features interativas e cimentada contra valores sentinelas.
        """
        # Otimização MLOps (Mitigação de OOM): Evitamos X.copy() inteiro e mutamos apenas
        # features que chegam. Para o sklearn, instanciamos um dataframe em memória sob demanda.
        X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # ----- 1. REMOCAO DE FEATURES SEM SINAL -----
        cols_to_drop = [c for c in self.FEATURES_TO_DROP if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        for col, flag_name in self.SENTINEL_COLUMNS.items():
            if col in X.columns:
                # Usar np.where ou atribuição no dict do pandas, otimizado inplace ou sem cópia pesada
                X[flag_name] = (X[col] != -1).astype(np.int8)
                # Cast to float32 specifically to avoid float64 memory explosion when filling NaN
                X[col] = X[col].replace(-1, np.nan).astype(np.float32)

        # ----- 3. CLIPPING DE OUTLIERS -----
        if hasattr(self, 'clip_bounds_'):
            for col, (lower, upper) in self.clip_bounds_.items():
                if col in X.columns:
                    X[col] = X[col].clip(lower=lower, upper=upper)

        # ----- 4. FLAGS DE RISCO CATEGORICO -----
        for col, (risk_value, flag_name) in self.HIGH_RISK_CATEGORIES.items():
            if col in X.columns:
                X[flag_name] = (X[col] == risk_value).astype(np.int8)

        # ----- 5. FEATURE INTERACTIONS -----

        # 5a. DIGITAL RISK SCORE (email gratuito * multiplos emails no dispositivo)
        if 'email_is_free' in X.columns and 'device_distinct_emails_8w' in X.columns:
            X['digital_risk_score'] = (
                X['email_is_free'] * X['device_distinct_emails_8w']
            )

        # 5b. VELOCITY ANOMALY (pico de atividade recente vs historico)
        if 'velocity_6h' in X.columns and 'velocity_4w' in X.columns:
            X['velocity_anomaly'] = X['velocity_6h'] / (X['velocity_4w'] + 1)

        # 5c. CREDIT UTILIZATION RATIO (tentativa de usar todo o credito)
        if 'intended_balcon_amount' in X.columns and 'proposed_credit_limit' in X.columns:
            X['credit_utilization'] = X['intended_balcon_amount'] / (X['proposed_credit_limit'] + 1)

        # 5d. AGE-INCOME DEMOGRAPHIC RISK (perfil demografico anomalo)
        if 'customer_age' in X.columns and 'income' in X.columns:
            X['age_income_risk'] = X['customer_age'] * X['income']

        # 5e. MULTI-RISK COMBINATION FLAG (acumulo de fatores de risco)
        if all(col in X.columns for col in ['is_high_risk_housing', 'is_high_risk_os', 'email_is_free']):
            X['multi_risk_flag'] = (
                X['is_high_risk_housing'] * X['is_high_risk_os'] * X['email_is_free']
            ).astype(np.int8)

        # 5f. PHONE MISMATCH (apenas um tipo de telefone valido)
        if 'phone_home_valid' in X.columns and 'phone_mobile_valid' in X.columns:
            X['phone_mismatch'] = (
                X['phone_home_valid'] != X['phone_mobile_valid']
            ).astype(np.int8)

        # 5g. EMAIL-VELOCITY RISK (email gratuito + alta velocidade = automacao)
        if 'email_is_free' in X.columns and 'velocity_6h' in X.columns:
            X['email_velocity_risk'] = X['email_is_free'] * X['velocity_6h']

        # 5h. ADDRESS STABILITY (estabilidade no endereco em relacao a idade)
        if 'current_address_months_count' in X.columns and 'customer_age' in X.columns:
            X['address_stability'] = X['current_address_months_count'] / (X['customer_age'] * 12 + 1)

        return X

    def get_feature_names_out(self, input_features=None):
        """Compatibilidade com sklearn para nomes de features."""
        return input_features


def get_preprocessor(X):
    """
    Sub-módulo de montagem das calhas (Pipeline Base Paramétrica).
    
    Por que existe:
    Modelos matemáticos (Salvo árvores puristas) colapsam ou enviesam com colunas de texto 
    ou matrizes numéricas fora de proporção normalizada. Este orquestrador separa as tipagens, 
    atribui a conversão respectiva e funde tudo num Output multidimensional cego para a Máquina.

    Regra de Negócio Crucial:
    O dataset inputado a ele DEVE JA ASSUMIR A EXISTÊNCIA DAS NOVAS COLUNAS 
    do EDAFeatureEngineer. Por isso invocamos um `fit_transform` prévio nas rotinas lá embaixo,
    senão o Preprocesser ignora features importantes como 'digital_risk_score'.

    Recebe:
    X (pd.DataFrame): DataFrame pré-processado pela Engenharia inicial.
    
    Retorna:
    ColumnTransformer: Objeto Sklearn de canais segregados (Num e Cat), pronto para `.fit()`.
    """
    # Inspeção e tipagem da malha dimensional (Pandas Type Introspection)
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int8', 'float32']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"   [FE] Features Numericas detectadas: {len(numeric_features)}")
    print(f"   [FE] Features Categoricas detectadas: {len(categorical_features)}")

    # --------------------------------------------------------------------------
    # PIPELINE NUMÉRICO: Proteção das Contas 
    #
    # Lógica Imputer Médio: Preenche lacunas restantes com a Mediana, ignorando as sentinelas 
    #   (-1 cortado acima), focando no saldo do mercado. 
    # Lógica Scaler Anti-Fraude: Emprega `RobustScaler` com limite de espalhamento Interquartil (IQR) 
    #   para imunizar contra Bilionários/Ladrões milionários que arruinariam o `StandardScaler`.
    # --------------------------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), 
    ])

    # --------------------------------------------------------------------------
    # PIPELINE CATEGÓRICO: One-Hot Escalável
    #
    # Lógica de Crashing: Se um sistema externo enviar uma feature STRING nova que foi criada no FrontEnd 
    #   após o treino, o parâmetro handle_unknown='ignore' força um vetor booleano ZERO pacífico,
    #   evitando uma fatal exception ("Unseen category") que derrubaria todas as inferências do banco.
    # --------------------------------------------------------------------------
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Unificacao dos Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def build_pipeline(X_train, model, undersampling_ratio=None):
    """
    Construtor oficial da Base Híbrida de Escala & Modelo.
    Fábrica onde os blocos modulares soltos de MLOps são selados dentro de um único pacote.

    Por que existe:
    Permite rodar as GridSearches focadas nos Modelos Otimos SEM PREOCUPAR O DESENVOLVEDOR com vazamento
    de dados nas dobras (Folds) em validação cruzada, encadeando logicamente Transformer > Estimator
    numa rotina sólida (Ou ImbPipeline em situações assíncronas).

    Recebe:
    X_train (pd.DataFrame): Dados de treinamento crús onde o preprocessor validará os dtypes.
    model: Componente não inicializado / estimador ScikitLearn limpo a ser alocado como final do tubo.
    undersampling_ratio (float | None): Flag condicional de MLOps. Se existir uma taxa (Ex `0.5`), muda o cano
        inteiro de Scikit Pipeline Clássico para Imbalanced-Learn Pipe com RandomUnderSampling automático 
        implantado sob demada, focando em lidar com a base brutal (98% Normal x 1% Fraudes).

    Retorna:
    (sklearn.pipeline.Pipeline | imblearn.pipeline.Pipeline): Tubo armado a laser pronto pro `.fit(X,y)` do usuário.
    """
    # Aplica o EDAFeatureEngineer nos dados de treino para que o ColumnTransformer
    # detecte as colunas corretas (incluindo novas flags e sem as removidas)
    eda_engineer = EDAFeatureEngineer()
    X_transformed = eda_engineer.fit_transform(X_train)

    # Constroi o preprocessor baseado nas colunas pos-engenharia
    preprocessor = get_preprocessor(X_transformed)

    # Pipeline final de 3 ou 4 etapas
    if undersampling_ratio is not None:
        if not HAS_IMBLEARN:
            raise ImportError(
                "CRITICO: O pacote imbalanced-learn nao esta instalado. "
                "Para usar o parametro undersampling_ratio (Opcao A), execute: pip install imbalanced-learn"
            )
        
        print(f"   [FE] Aplicando RandomUnderSampler (ratio={undersampling_ratio}) de forma robusta e otimizada...")
        sampler = RandomUnderSampler(sampling_strategy=undersampling_ratio, random_state=RANDOM_STATE)
        
        # O ImbPipeline e fundamental pois ele aplica o sampler APENAS no X_train
        # durante a validacao cruzada, garantindo que o X_test continue representando
        # a populacao real, evitando que as metricas sejam mascaradas.
        pipeline = ImbPipeline(steps=[
            ('eda_features', eda_engineer),
            ('preprocessor', preprocessor),
            ('sampler', sampler),
            ('model', model)
        ])
    else:
        # Modo Classico / Padrao
        pipeline = Pipeline(steps=[
            ('eda_features', eda_engineer),
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    return pipeline


def process_features():
    """
    Modo de Execução Standalone e Geração de Artefatos em Lote (Artifact Exporter).
    
    Por que existe:
    Se os desenvolvedores necessitarem rodar uma bateria de re-treinamento ou apenas injetar o arquivo
    `preprocessor.joblib` na AWS sem treinar um modelo novo, essa porta isolada assume o encargo, 
    rodando apenas o Transformer sem engatar nenhum Boosting, selando o binário no diretório de saída.
    """
    print("Iniciando construcao de features...")
    X_train = pd.read_pickle(PROCESSED_DATA_DIR / "X_train.pkl")
    y_train = pd.read_pickle(PROCESSED_DATA_DIR / "y_train.pkl").values.ravel()

    # Aplica feature engineering EDA-driven
    eda_engineer = EDAFeatureEngineer()
    X_engineered = eda_engineer.fit_transform(X_train)

    # Cria a "receita" de transformacao baseada nas colunas do treino
    preprocessor = get_preprocessor(X_engineered)

    pipeline = Pipeline(steps=[
        ('eda_features', eda_engineer),
        ('preprocessor', preprocessor)
    ])

    print("   Ajustando transformadores (Fit)...")
    X_train_transformed = pipeline.fit_transform(X_train)
    
    # PERSISTENCIA CRITICA:
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    
    print(f"   Features processadas com sucesso!")
    print(f"   Shape Original: {X_train.shape} -> Features Extraidas: {X_train_transformed.shape}")
    print(f"   Pipeline salvo em: {MODELS_DIR / 'preprocessor.joblib'}")
    
    return pipeline

if __name__ == "__main__":
    process_features()