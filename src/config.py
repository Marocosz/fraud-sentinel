# ==============================================================================
# ARQUIVO: config.py
#
# OBJETIVO:
#   Centralizar todas as configurações globais do projeto em um único local.
#   Define os caminhos absolutos para persistência de arquivos, configurações 
#   base para hiperparâmetros (seed, splits) e demais constantes estáticas.
#
# PARTE DO SISTEMA:
#   Configuração Global / Setup de Infraestrutura.
#
# RESPONSABILIDADES:
#   - Declarar dinamicamente as raízes estruturais através da classe Path.
#   - Expor nomes estáticos essenciais (TARGET_COL, RANDOM_STATE, TEST_SIZE).
#   - Automatizar a criação dos diretórios alvo caso não existam (raw, processed, reports).
#
# COMUNICAÇÃO:
#   - É lido por diversos módulos: orquestração (main), `build_features`, testes e modelos.
# ==============================================================================

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