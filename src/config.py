# Configurações globais (caminhos, sementes aleatórias, hiperparâmetros)
from pathlib import Path

# Caminhos Base
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Arquivos Específicos
RAW_DATA_PATH = DATA_DIR / "raw" / "Base.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Parâmetros Globais
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "fraud_bool"  # Nome da coluna alvo no BAF Suite

# Criação automática de pastas se não existirem
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)