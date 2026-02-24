# ==============================================================================
# ARQUIVO: predict_ensemble.py
#
# OBJETIVO:
#   Atuar como o Motor de Inferência (Serving) em produção utilizando uma
#   arquitetura MLOps de Comitê de Modelos (Ensemble Voting) com regras de negócio.
#
# PARTE DO SISTEMA:
#   Módulo de Inferência em Produção (MLOps).
#
# RESPONSABILIDADES:
#   - Carregar 3 modelos campeões com objetivos distintos (Precision, Balance, Recall).
#   - Carregar seus respectivos limiares operacionais (Thresholds).
#   - Implementar lógica de "Smart Majority Vote com Veto Especial".
#   - Fornecer um veredito estruturado por transação, tipado com Pydantic/Dataclasses.
#
# COMUNICAÇÃO:
#   - Lê modelos de 'models/'.
#   - Provê interface de predição `predict_batch`.
# ==============================================================================

import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import joblib

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import MODELS_DIR

# Configuração de Logs Profissional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ENSEMBLE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuração para cada modelo membro do Comitê."""
    name: str
    file_name: str
    thresh_file: str
    role: str
    model_obj: Any = None
    threshold: float = 0.5

@dataclass
class TransactionResult:
    """Estrutura de dados para o resultado da inferência de uma transação."""
    transaction_index: Any
    final_decision: str
    confidence: str
    fraud_votes: int
    total_active_models: int
    committee_details: Dict[str, Dict[str, Any]]

class FraudEnsemblePredictor:
    """
    Preditor em formato de Comitê (Ensemble) para transações financeiras.
    Combina as predições de múltiplos modelos otimizados para maximizar a assertividade
    de acordo com as regras de negócio "Smart Majority Vote com Veto Especial".
    """
    
    def __init__(self) -> None:
        self.committee: Dict[str, ModelConfig] = {
            'lightgbm': ModelConfig(name='lightgbm', file_name='lgbm_best_model.pkl', thresh_file='lgbm_threshold.txt', role='Campeão de Precisão'),
            'xgboost': ModelConfig(name='xgboost', file_name='xgb_best_model.pkl', thresh_file='xgb_threshold.txt', role='Campeão de Equilíbrio (F1)'),
            'mlp': ModelConfig(name='mlp', file_name='mlp_best_model.pkl', thresh_file='mlp_threshold.txt', role='Campeão de Recall (Sensibilidade)')
        }
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Carrega os modelos e thresholds persistidos no disco.
        Garante tolerância a falhas caso algum artefato esteja ausente.
        """
        logger.info("Iniciando carregamento do Comitê de Modelos (Ensemble)...")
        loaded_count = 0
        
        for name, config in self.committee.items():
            model_path = MODELS_DIR / config.file_name
            thresh_path = MODELS_DIR / config.thresh_file
            
            # Carregar Modelo
            try:
                if model_path.exists():
                    config.model_obj = joblib.load(model_path)
                    logger.info(f"✅ Modelo carregado: {name.upper()} ({config.role})")
                    loaded_count += 1
                else:
                    logger.warning(f"⚠️ Modelo não encontrado: {model_path.name}. O comitê terá menos membros.")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar {name}: {e}")
                
            # Carregar Threshold
            try:
                if thresh_path.exists():
                    with open(thresh_path, 'r', encoding='utf-8') as f:
                        config.threshold = float(f.read().strip())
                    logger.info(f"   🎯 Threshold Otimizado carregado: {config.threshold:.4f}")
                else:
                    logger.warning(f"   ⚠️ Threshold otimizado não encontrado para {name}. Usando fallback: 0.5")
            except Exception as e:
                logger.warning(f"   ⚠️ Erro ao ler threshold ({e}). Usando fallback: 0.5")

        if loaded_count == 0:
            logger.error("Nenhum modelo foi carregado! O sistema não pode operar.")
            raise ValueError("O comitê está vazio. Falha crítica no motor de inferência.")

    def predict_batch(self, df: pd.DataFrame) -> List[TransactionResult]:
        """
        Recebe um lote de transações (DataFrame) e avalia cada uma através do comitê.
        Aplica a Regra de Negócio de Veto Especial.
        
        Args:
            df (pd.DataFrame): DataFrame contendo features prontas.
            
        Returns:
            List[TransactionResult]: Lista de objetos tipados com o veredito por transação.
        """
        results: List[TransactionResult] = []
        
        for idx in range(len(df)):
            transaction = df.iloc[[idx]]
            committee_details = {}
            fraud_votes = 0
            voted_fraud_models = []
            
            # 1. Realizar Predições Individuais
            for name, config in self.committee.items():
                if config.model_obj is None:
                    continue # Pula modelo indisponível
                
                try:
                    prob = config.model_obj.predict_proba(transaction)[0, 1]
                    is_fraud = bool(prob >= config.threshold)
                    
                    if is_fraud:
                        fraud_votes += 1
                        voted_fraud_models.append(name)
                        
                    committee_details[name] = {
                        "score": round(prob, 4),
                        "threshold": round(config.threshold, 4),
                        "vote_fraud": is_fraud
                    }
                except Exception as e:
                    logger.error(f"Erro na predição do modelo {name}: {e}")
            
            # 2. Lógica de Negócio: Smart Majority Vote com Veto Especial
            # CRÍTICO: denominação usa apenas os modelos que sobreviveram a possíveis falhas
            total_active_models = len(committee_details)
            
            # Default fallback para falha completa
            if total_active_models == 0:
                results.append(TransactionResult(
                    transaction_index=df.index[idx] if 'index' in df.columns else idx,
                    final_decision="ERRO",
                    confidence="ERRO CRÍTICO (Todos modelos falharam)",
                    fraud_votes=0,
                    total_active_models=0,
                    committee_details={}
                ))
                continue
                
            majority_threshold = (total_active_models // 2) + 1
            
            # Default
            final_decision = "APROVAR"
            confidence = "ALTA (Seguro)"
            
            if fraud_votes >= majority_threshold:
                final_decision = "BLOQUEAR"
                confidence = "ALTA (Unanimidade/Maioria Clara)" if fraud_votes == total_active_models else "MÉDIA (Maioria)"
            elif fraud_votes > 0:
                # Regra de Veto: Se APENAS o LightGBM (Campeão de Precisão) disser que é fraude
                if len(voted_fraud_models) == 1 and voted_fraud_models[0] == 'lightgbm':
                    final_decision = "REVISÃO MANUAL"
                    confidence = "MÉDIA (Veto do Campeão de Precisão - LightGBM alerta risco grave)"
                else:
                    final_decision = "APROVAR"
                    confidence = "MÉDIA (Divergência menor - Risco assumido)"

            # Empacotar resultado strongly-typed
            trans_index = df.index[idx] if 'index' in df.columns else idx
            
            result = TransactionResult(
                transaction_index=trans_index,
                final_decision=final_decision,
                confidence=confidence,
                fraud_votes=fraud_votes,
                total_active_models=total_active_models,
                committee_details=committee_details
            )
            results.append(result)
            
        return results

if __name__ == "__main__":
    logger.info("Homologação da Instanciação do FraudEnsemblePredictor...")
    try:
        predictor = FraudEnsemblePredictor()
        logger.info("Sistema de inferência pronto para produção!")
    except Exception as e:
        logger.error(f"Falha de inicialização: {e}")
