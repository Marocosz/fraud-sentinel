import unittest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Mockup data based on actual features observed
mock_data = {
    'income': [0.1, 0.9, 0.5],
    'name_email_similarity': [0.1, 0.9, 0.5],
    'prev_address_months_count': [-1, 20, 100],
    'current_address_months_count': [5, 50, 200],
    'customer_age': [20, 40, 60],
    'days_since_request': [0.01, 1.5, 10.0],
    'intended_balcon_amount': [-10.0, 50.0, 100.0],
    'payment_type': ['AA', 'AB', 'AC'],
    'zip_count_4w': [100, 2000, 5000],
    'velocity_6h': [1000, 5000, 10000],
    'velocity_24h': [2000, 6000, 8000],
    'velocity_4w': [3000, 5000, 7000],
    'bank_branch_count_8w': [1, 20, 2000],
    'date_of_birth_distinct_emails_4w': [2, 10, 25],
    'employment_status': ['CA', 'CC', 'CG'],
    'credit_risk_score': [50, 150, 300],
    'email_is_free': [0, 1, 0],
    'housing_status': ['BA', 'BB', 'BC'],
    'phone_home_valid': [0, 1, 0],
    'phone_mobile_valid': [1, 1, 1],
    'bank_months_count': [-1, 10, 30],
    'has_other_cards': [0, 1, 0],
    'proposed_credit_limit': [200, 500, 1500],
    'foreign_request': [0, 0, 1],
    'source': ['INTERNET', 'TELEAPP', 'INTERNET'],
    'session_length_in_minutes': [2.0, 10.0, 50.0],
    'device_os': ['windows', 'linux', 'macintosh'],
    'keep_alive_session': [0, 1, 0],
    'device_distinct_emails_8w': [1, 1, 2],
    'device_fraud_count': [0, 0, 0],
    'month': [1, 5, 7]
}

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        from src.features.build_features import EDAFeatureEngineer
        self.engineer = EDAFeatureEngineer()
        self.mock_df = pd.DataFrame(mock_data)

    def test_eda_feature_engineer_transformation(self):
        """Testa as operacoes customizadas de features, como remocao, clipping e criacao de flags."""
        # fit process learn limit values
        self.engineer.fit(self.mock_df)
        transformed_df = self.engineer.transform(self.mock_df)
        
        # Verify removal of low variance/MI features
        self.assertNotIn('device_fraud_count', transformed_df.columns, "Feature variancia zero nao foi removida")
        self.assertNotIn('session_length_in_minutes', transformed_df.columns, "Feature s/ significancia nao foi removida")
        
        # Verify creation of engineered risk flags
        self.assertIn('digital_risk_score', transformed_df.columns, "Feature de risco digital nao criada")
        self.assertIn('is_high_risk_housing', transformed_df.columns, "Flag de housing BA nao criada")
        self.assertIn('is_high_risk_os', transformed_df.columns, "Flag de windows OS nao criada")
        
    def test_pipeline_integration(self):
        """Verifica se o pipeline completo do get_preprocessor constroi tensores corretamente."""
        from src.features.build_features import get_preprocessor
        
        # Pass data through the engineered process first (which matches the pipeline logic)
        self.engineer.fit(self.mock_df)
        transformed_df = self.engineer.transform(self.mock_df)
        
        preprocessor = get_preprocessor(transformed_df)
        
        # Fit logic of scikit learn robust scaler + one hot encoder
        processed_matrix = preprocessor.fit_transform(transformed_df)
        
        # Output should be sparse robust matrix 
        self.assertGreater(processed_matrix.shape[1], transformed_df.shape[1], "O One-Hot Encoding falhou ou ignorou features categoricas")
        
if __name__ == '__main__':
    unittest.main()
