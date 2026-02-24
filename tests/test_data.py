import unittest
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, RAW_DATA_PATH

class TestDataPipeline(unittest.TestCase):
    def test_processed_files_exist(self):
        """Verifica se os arquivos processados necessarios foram gerados."""
        self.assertTrue((PROCESSED_DATA_DIR / "X_train.pkl").exists(), "X_train.pkl nao encontrado")
        self.assertTrue((PROCESSED_DATA_DIR / "y_train.pkl").exists(), "y_train.pkl nao encontrado")
        self.assertTrue((PROCESSED_DATA_DIR / "X_test.pkl").exists(), "X_test.pkl nao encontrado")
        self.assertTrue((PROCESSED_DATA_DIR / "y_test.pkl").exists(), "y_test.pkl nao encontrado")

    def test_no_missing_values_in_target(self):
        """Verifica se o dataset de treino tem a coluna target perfeitamente preenchida."""
        if (PROCESSED_DATA_DIR / "y_train.pkl").exists():
            y_train = pd.read_pickle(PROCESSED_DATA_DIR / "y_train.pkl")
            self.assertEqual(y_train.isnull().sum().sum(), 0, "Valores nulos encontrados no y_train")
            
    def test_data_shapes(self):
        """Verifica se as dimensoes dos dados de treino fazem sentido estatistico."""
        if (PROCESSED_DATA_DIR / "X_train.pkl").exists() and (PROCESSED_DATA_DIR / "y_train.pkl").exists():
            X_train = pd.read_pickle(PROCESSED_DATA_DIR / "X_train.pkl")
            y_train = pd.read_pickle(PROCESSED_DATA_DIR / "y_train.pkl")
            self.assertEqual(X_train.shape[0], y_train.shape[0], "Dimensao de linhas de X_train e y_train difere")

if __name__ == '__main__':
    unittest.main()
