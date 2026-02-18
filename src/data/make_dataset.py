import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==============================================================================
# ARQUIVO: make_dataset.py
#
# OBJETIVO:
#   Preparar o dataset bruto para a modelagem. 
#   Realiza a carga, otimização de memória e divisão em Treino/Teste.
#
# PARTE DO SISTEMA:
#   Pipeline de Engenharia de Dados (Data Ingestion & Splitting).
#
# RESPONSABILIDADES:
#   - Carregar o CSV bruto (Base.csv).
#   - Otimizar tipos de dados (downcasting) para reduzir uso de RAM.
#   - Validar a existência do target.
#   - Realizar o split ESTRATIFICADO (mantendo a % de fraudes).
#   - Salvar os artefatos (X_train, X_test, y_train, y_test) prontos para uso.
#
# COMUNICAÇÃO:
#   - Lê: data/raw/Base.csv
#   - Escreve: data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
# ==============================================================================

# Adiciona raiz ao path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Tenta importar configurações
try:
    from src.config import RAW_DATA_PATH, PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE, TARGET_COL
except ImportError:
    # Fallback
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Base.csv"
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    TARGET_COL = "fraud_bool"

def optimize_memory_usage(df):
    """
    Itera por todas as colunas do dataframe e modifica o tipo de dado
    para reduzir o uso de memória (downcasting).
    
    Ex: float64 -> float32, int64 -> int16/int8.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"   💾 Memória antes da otimização: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32) # float16 tem precisão muito baixa, float32 é seguro
                else:
                    df[col] = df[col].astype(np.float32)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    reduction = 100 * (start_mem - end_mem) / start_mem
    print(f"   💾 Memória após otimização: {end_mem:.2f} MB (Redução de {reduction:.1f}%)")
    return df

def load_and_split_data():
    """
    Função principal de processamento.
    """
    print(f"🚀 Iniciando pipeline de dados...")
    print(f"📂 Carregando dataset: {RAW_DATA_PATH}...")
    
    if not RAW_DATA_PATH.exists():
         raise FileNotFoundError(f"Erro: O arquivo {RAW_DATA_PATH} não foi encontrado. Verifique se ele está na pasta data/raw/.")

    # Carrega CSV
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"   ✅ Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")

    # Validação Básica
    if TARGET_COL not in df.columns:
        # Tenta inferir se o nome for diferente
        if 'is_fraud' in df.columns:
            print(f"⚠️ Coluna alvo '{TARGET_COL}' não encontrada, mas 'is_fraud' detectada. Renomeando...")
            df.rename(columns={'is_fraud': TARGET_COL}, inplace=True)
        else:
            raise ValueError(f"CRÍTICO: Coluna alvo '{TARGET_COL}' não encontrada no dataset.")

    # Otimização de Memória (Crucial para datasets grandes de fraude)
    df = optimize_memory_usage(df)

    # Separação X (Features) e y (Target)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Relatório de Classes Antes do Split
    fraud_rate = (y.sum() / len(y)) * 100
    print(f"📊 Distribuição Original: {y.sum()} Fraudes ({fraud_rate:.2f}%) de {len(y)} registros")

    # Divisão Estratificada (Mantém a proporção de fraudes no Treino e Teste)
    # random_state garante que o split seja sempre o mesmo (reprodutibilidade)
    print(f"✂️ Dividindo dados (Teste = {TEST_SIZE*100}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Validação Pós-Split
    train_fraud_rate = (y_train.sum() / len(y_train)) * 100
    test_fraud_rate = (y_test.sum() / len(y_test)) * 100
    print(f"   ✅ Treino: {len(X_train)} linhas (Fraude: {train_fraud_rate:.2f}%)")
    print(f"   ✅ Teste:  {len(X_test)} linhas (Fraude: {test_fraud_rate:.2f}%)")

    # Garante diretório de saída
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Salvando arquivos
    # index=False evita criar uma coluna "Unnamed: 0" inútil
    print("💾 Salvando arquivos processados em data/processed/...")
    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print("🏁 Pipeline de dados concluído com sucesso!")

if __name__ == "__main__":
    load_and_split_data()