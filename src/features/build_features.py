import pandas as pd
import joblib
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
#   Construir o pipeline de transformação de features (Feature Engineering Automático).
#   Define como números e textos brutos são convertidos em matrizes matemáticas
#   que os algoritmos conseguem entender.
#
# PARTE DO SISTEMA:
#   Módulo de Engenharia de Features (Preprocessing Stage).
#
# RESPONSABILIDADES:
#   - Identificar automaticamente tipos de dados (Numérico vs Categórico).
#   - Definir estratégias de imputação para valores nulos (Median/Missing).
#   - Aplicar normalização robusta a outliers (RobustScaler).
#   - Aplicar codificação One-Hot para variáveis categóricas.
#   - Persistir o objeto transformador (preprocessor.joblib) para uso futuro
#     na etapa de inferência/produção.
#
# COMUNICAÇÃO:
#   - Lê: data/processed/X_train.csv (para "aprender" a escala dos dados)
#   - Escreve: models/preprocessor.joblib (Artefato reutilizável)
# ==============================================================================

def get_preprocessor(X):
    """
    Cria o objeto ColumnTransformer que orquestra as transformações de dados.
    
    Lógica de Separação:
    - O scikit-learn não adivinha tipos nativamente, então segregamos colunas
      por dtype (int/float -> numérico, object/category -> categórico).
      
    Returns:
        ColumnTransformer: O pipeline completo pronto para .fit().
    """
    # Identifica colunas automaticamente baseado no tipo de dado do Pandas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"   [FE] Features Numéricas detectadas: {len(numeric_features)}")
    print(f"   [FE] Features Categóricas detectadas: {len(categorical_features)}")

    # --------------------------------------------------------------------------
    # PIPELINE NUMÉRICO
    # 1. SimpleImputer(median): Preenche nulos com a mediana (robusto a outliers).
    # 2. RobustScaler: Normaliza usando (x - mediana) / IQR. 
    #    CRÍTICO para fraude: Diferente do StandardScaler (média/desvio), o RobustScaler
    #    não é "esmagado" por valores milionários extremos.
    # --------------------------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()), 
        # ('selector', VarianceThreshold(threshold=0.01)) # Opcional: Remove colunas quase constantes
    ])

    # --------------------------------------------------------------------------
    # PIPELINE CATEGÓRICO
    # 1. SimpleImputer(constant): Preenche nulos com o texto 'missing'.
    # 2. OneHotEncoder: Cria colunas binárias para cada categoria.
    #    handle_unknown='ignore': Se aparecer uma categoria nova em produção que não
    #    existia no treino, o modelo ignora (tudo zero) em vez de quebrar (Crash).
    # --------------------------------------------------------------------------
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Unificação dos Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def process_features():
    """
    Função de execução isolada (opcional).
    Gera o preprocessor e testa o pipeline completo de resampling (SMOTE).
    
    Nota Importante de Arquitetura:
    - O SMOTE (criação de dados sintéticos) só deve ser aplicado no TREINO.
    - Por isso, temos um pipeline 'full_pipeline' que inclui SMOTE para validação,
      mas salvamos em disco apenas o 'preprocessor' (sem SMOTE) para ser usado
      nos dados de teste/produção.
    """
    print("🛠️ Iniciando construção de features...")
    X_train = pd.read_csv(PROCESSED_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DATA_DIR / "y_train.csv").values.ravel()

    # Cria a "receita" de transformação baseada nas colunas do treino
    preprocessor = get_preprocessor(X_train)

    # Pipeline de Validação (com Oversampling)
    # Usado apenas para verificar se o SMOTE roda sem erro de memória/tipo
    full_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=RANDOM_STATE))
    ])

    print("   Ajustando transformadores (Fit) e aplicando SMOTE...")
    # fit_resample executa: 1.Transforma -> 2.Cria Fraudes Falsas (SMOTE)
    X_train_resampled, y_train_resampled = full_pipeline.fit_resample(X_train, y_train)
    
    # PERSISTÊNCIA CRÍTICA:
    # Salvamos apenas o 'preprocessor'.
    # Motivo: Em produção, não queremos gerar dados falsos (SMOTE), apenas transformar os reais.
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    
    print(f"   ✅ Features processadas com sucesso!")
    print(f"   Shape Original: {X_train.shape} -> Pós-SMOTE: {X_train_resampled.shape}")
    print(f"   💾 Pipeline salvo em: {MODELS_DIR / 'preprocessor.joblib'}")
    
    return full_pipeline

if __name__ == "__main__":
    process_features()