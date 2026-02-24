import pandas as pd
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from src.config import PROCESSED_DATA_DIR, MODELS_DIR, RANDOM_STATE

# ==============================================================================
# ARQUIVO: build_features.py
#
# OBJETIVO:
#   Construir o pipeline de transformacao de features (Feature Engineering Automatico).
#   Define como numeros e textos brutos sao convertidos em matrizes matematicas
#   que os algoritmos conseguem entender.
#
#   ATUALIZADO com base nos insights da Analise Exploratoria (EDA):
#   - Remocao de features com variancia zero e MI = 0
#   - Tratamento de valores sentinela (-1) como dados ausentes
#   - Clipping de outliers extremos (percentis 1% e 99%)
#   - Criacao de flags binarias de risco categorico
#   - Feature interactions para comportamento digital
#
# PARTE DO SISTEMA:
#   Modulo de Engenharia de Features (Preprocessing Stage).
#
# RESPONSABILIDADES:
#   - Aplicar feature engineering orientado por dados (EDA-driven).
#   - Identificar automaticamente tipos de dados (Numerico vs Categorico).
#   - Definir estrategias de imputacao para valores nulos (Median/Missing).
#   - Aplicar normalizacao robusta a outliers (RobustScaler).
#   - Aplicar codificacao One-Hot para variaveis categoricas.
#   - Persistir o objeto transformador (preprocessor.joblib) para uso futuro
#     na etapa de inferencia/producao.
#
# COMUNICACAO:
#   - Le: data/processed/X_train.csv (para "aprender" a escala dos dados)
#   - Escreve: models/preprocessor.joblib (Artefato reutilizavel)
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
        Aprende os limites de clipping (percentis 1% e 99%) a partir do
        conjunto de treino. Esses limites sao fixados e aplicados em
        transform() para treino E teste, evitando data leakage.
        """
        X_work = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

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
        Aplica todas as transformacoes EDA-driven em sequencia.
        Retorna um DataFrame com as features engenheiradas.
        """
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # ----- 1. REMOCAO DE FEATURES SEM SINAL -----
        cols_to_drop = [c for c in self.FEATURES_TO_DROP if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)

        # ----- 2. TRATAMENTO DE SENTINELAS (-1 -> NaN + Flag) -----
        for col, flag_name in self.SENTINEL_COLUMNS.items():
            if col in X.columns:
                X[flag_name] = (X[col] != -1).astype(np.int8)
                X[col] = X[col].replace(-1, np.nan)

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
    Cria o objeto ColumnTransformer que orquestra as transformacoes de dados.
    
    IMPORTANTE: Esta funcao deve receber o DataFrame JA TRANSFORMADO pelo
    EDAFeatureEngineer, pois detecta colunas automaticamente por dtype.
    No pipeline, o EDAFeatureEngineer roda como primeiro step e o
    ColumnTransformer como segundo.

    Logica de Separacao:
    - O scikit-learn nao adivinha tipos nativamente, entao segregamos colunas
      por dtype (int/float -> numerico, object/category -> categorico).
      
    Returns:
        ColumnTransformer: O pipeline completo pronto para .fit().
    """
    # Identifica colunas automaticamente baseado no tipo de dado do Pandas
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int8', 'float32']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"   [FE] Features Numericas detectadas: {len(numeric_features)}")
    print(f"   [FE] Features Categoricas detectadas: {len(categorical_features)}")

    # --------------------------------------------------------------------------
    # PIPELINE NUMERICO
    # 1. SimpleImputer(median): Preenche nulos com a mediana (robusto a outliers).
    #    CRITICO apos o EDAFeatureEngineer: Os sentinelas -1 foram convertidos
    #    em NaN, e o SimpleImputer agora calcula a mediana dos valores REAIS
    #    (sem os -1), resultando em imputacao muito mais precisa.
    # 2. RobustScaler: Normaliza usando (x - mediana) / IQR. 
    #    CRITICO para fraude: Diferente do StandardScaler (media/desvio), o RobustScaler
    #    nao e "esmagado" por valores milionarios extremos.
    # --------------------------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), 
    ])

    # --------------------------------------------------------------------------
    # PIPELINE CATEGORICO
    # 1. SimpleImputer(constant): Preenche nulos com o texto 'missing'.
    # 2. OneHotEncoder: Cria colunas binarias para cada categoria.
    #    handle_unknown='ignore': Se aparecer uma categoria nova em producao que nao
    #    existia no treino, o modelo ignora (tudo zero) em vez de quebrar (Crash).
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


def build_pipeline(X_train, model):
    """
    Constroi o pipeline completo com Feature Engineering (EDA-driven) + Preprocessing + Modelo.

    Esta funcao centraliza a criacao do pipeline de 3 etapas:
    1. EDAFeatureEngineer: Engenharia de features baseada nos insights do EDA
    2. ColumnTransformer: Preprocessamento (Scaler, Imputer, OneHot)
    3. Modelo: Algoritmo de classificacao

    Args:
        X_train (pd.DataFrame): Dados de treino brutos (para detectar tipos de coluna).
        model: Instancia do classificador (LogReg, XGBoost, etc).

    Returns:
        Pipeline: Pipeline completo pronto para .fit() ou GridSearchCV.
    """
    # Aplica o EDAFeatureEngineer nos dados de treino para que o ColumnTransformer
    # detecte as colunas corretas (incluindo novas flags e sem as removidas)
    eda_engineer = EDAFeatureEngineer()
    X_transformed = eda_engineer.fit_transform(X_train)

    # Constroi o preprocessor baseado nas colunas pos-engenharia
    preprocessor = get_preprocessor(X_transformed)

    # Pipeline final de 3 etapas
    pipeline = Pipeline(steps=[
        ('eda_features', eda_engineer),
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def process_features():
    """
    Funcao de execucao isolada (opcional).
    Gera o preprocessor e testa o pipeline completo de resampling (SMOTE).
    
    Nota Importante de Arquitetura:
    - O SMOTE (criacao de dados sinteticos) so deve ser aplicado no TREINO.
    - Por isso, temos um pipeline 'full_pipeline' que inclui SMOTE para validacao,
      mas salvamos em disco apenas o 'preprocessor' (sem SMOTE) para ser usado
      nos dados de teste/producao.
    """
    print("Iniciando construcao de features...")
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()

    # Aplica feature engineering EDA-driven
    eda_engineer = EDAFeatureEngineer()
    X_engineered = eda_engineer.fit_transform(X_train)

    # Cria a "receita" de transformacao baseada nas colunas do treino
    preprocessor = get_preprocessor(X_engineered)

    # Pipeline de Validacao (com Oversampling)
    # Usado apenas para verificar se o SMOTE roda sem erro de memoria/tipo
    full_pipeline = ImbPipeline(steps=[
        ('eda_features', eda_engineer),
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE))
    ])

    print("   Ajustando transformadores (Fit) e aplicando SMOTE...")
    # fit_resample executa: 1.Feature Eng -> 2.Transforma -> 3.Cria Fraudes Falsas (SMOTE)
    X_train_resampled, y_train_resampled = full_pipeline.fit_resample(X_train, y_train)
    
    # PERSISTENCIA CRITICA:
    # Salvamos apenas o 'preprocessor'.
    # Motivo: Em producao, nao queremos gerar dados falsos (SMOTE), apenas transformar os reais.
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    
    print(f"   Features processadas com sucesso!")
    print(f"   Shape Original: {X_train.shape} -> Pos-SMOTE: {X_train_resampled.shape}")
    print(f"   Pipeline salvo em: {MODELS_DIR / 'preprocessor.joblib'}")
    
    return full_pipeline

if __name__ == "__main__":
    process_features()