# ==============================================================================
# ARQUIVO: predict_ensemble.py
#
# OBJETIVO:
#   Atuar como o Motor de Inferência (Serving) em produção utilizando uma
#   arquitetura MLOps de Comitê de Modelos (Ensemble Voting) embasado em regras de negócio.
#
# PARTE DO SISTEMA:
#   Módulo de Inferência em Produção (Risco de Crédito / Onboarding).
#
# RESPONSABILIDADES:
#   - Carregar e instanciar os top 3 modelos na memória do servidor.
#   - Resgatar Thresholds otimizados blindando a operação contra perdas financeiras.
#   - Orquestrar a regra mitigadora "Smart Majority Vote com Veto do Campeão".
#   - Retornar Vereditos transacionais formatados para a API de consumo do Banco.
#
# INTEGRAÇÕES:
#   - Lê arquivos binários e limites de `/models`.
#   - Provê: Uma classe `FraudEnsemblePredictor` ativa com o método público `predict_batch`
#     usado diretamente por aplicações, filas Kafka ou pelo Simulador de Produção.
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
            'lightgbm': ModelConfig(name='lightgbm', file_name='lgbm_best_model.pkl', thresh_file='lgbm_threshold.txt', role='Campeão Global F1 e Recall'),
            'xgboost': ModelConfig(name='xgboost', file_name='xgb_best_model.pkl', thresh_file='xgb_threshold.txt', role='Voto de Consenso/Estabilidade'),
            'mlp': ModelConfig(name='mlp', file_name='mlp_best_model.pkl', thresh_file='mlp_threshold.txt', role='Campeão de Precisão Cirúrgica')
        }
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """
        Rotina crítica de Bootstrap do Comitê. Carrega os artefatos binários dos modelos e 
        seus limiares matemáticos persistidos após o último treinamento.
        
        Por que existe:
        Um motor de inferência falha estrepitosamente se os pilares matemáticos não estiverem
        sincronizados. Este método isola a leitura de I/O em bloco de `try...except` para garantir
        que a quebra de um arquivo (ex: XGBoost corrompido) não crashe a inicialização imediata.

        Retorna:
        None.
        Efeito colateral: Abastece o dicionário `self.committee` com os objetos preditivos.
        Levanta um `ValueError` crítico se NENHUM modelo for carregado (Paralisação Preditiva).
        """
        logger.info("Iniciando carregamento do Comitê de Modelos (Ensemble)...")
        loaded_count = 0
        
        for name, config in self.committee.items():
            model_path = MODELS_DIR / config.file_name
            thresh_path = MODELS_DIR / config.thresh_file
            
            try:
                if model_path.exists():
                    config.model_obj = joblib.load(model_path)
                    logger.info(f"✅ Modelo carregado: {name.upper()} ({config.role})")
                    loaded_count += 1
                else:
                    logger.warning(f"⚠️ Modelo não encontrado: {model_path.name}. O comitê terá menos membros.")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar {name}: {e}")
                
            try:
                if thresh_path.exists():
                    with open(thresh_path, 'r', encoding='utf-8') as f:
                        config.threshold = float(f.read().strip())
                    logger.info(f"   🎯 Threshold Otimizado carregado: {config.threshold:.4f}")
                else:
                    logger.warning(f"   ⚠️ Threshold otimizado ausente para {name}. Usando fallback 0.5")
            except Exception as e:
                logger.warning(f"   ⚠️ Erro ao ler threshold ({e}). Usando fallback 0.5")

        if loaded_count == 0:
            logger.error("Nenhum modelo foi carregado! O sistema não pode operar às escuras.")
            raise ValueError("O comitê está vazio. Falha crítica no motor de inferência.")

    def predict_batch(self, df: pd.DataFrame) -> List[TransactionResult]:
        """
        Avalia um lote contínuo de matrizes tabulares e instaura julgamento comutado de Fraude vs Legitímidade.
        
        Por que existe:
        Predições linha a linha (`.predict_proba(series)`) estrangulam a engine matemática em cenários 
        assíncronos. Esta função suporta a injeção Vectorizada de milhares de linhas e aplica a Regra 
        de Negócio do negócio do Banco.

        Recebe:
        df (pd.DataFrame): Dados massivos enfileirados recém saídos do pipeline.
        
        Regra de Negócio Financeira:
        1. "Smart Majority Vote": Requer 2+/3 Modelos cruzando o threshold Tunado para ditar Bloqueio Automático.
        2. "Veto Exclusivo (Glass-Ceiling)": Se as árvores passarem a transação, mas a Rede Neural MLP
           (Dona de 95% de precisão nos testes) gritar "Fraude", não rejeita sumariamente para evitar CAC perdido.
           Ele empurra o cliente para "REVISÃO MANUAL" Humana (Back-Office).

        Retorna:
        List[TransactionResult]: Coleção DTO serializada com o status transparente e Voto a Voto da transação.
        """
        results: List[TransactionResult] = []
        n_samples = len(df)
        
        if n_samples == 0:
            return results

        # 1. Realiza Inferência Vetorizada Numpy
        predictions = {}
        for name, config in self.committee.items():
            if config.model_obj is not None:
                try:
                    probs = config.model_obj.predict_proba(df)[:, 1]
                    is_frauds = probs >= config.threshold
                    
                    predictions[name] = {
                        "probs": probs,
                        "is_frauds": is_frauds,
                        "threshold": config.threshold
                    }
                except Exception as e:
                    logger.error(f"Erro na predição em lote do modelo {name}: {e}")
                    
        total_active_models = len(predictions)
        
        # 2. Aplica Cruzamento de Regras nas Predições
        for idx in range(n_samples):
            committee_details = {}
            fraud_votes = 0
            voted_fraud_models = []
            
            for name, preds in predictions.items():
                prob = preds["probs"][idx]
                is_fraud = preds["is_frauds"][idx]
                
                if is_fraud:
                    fraud_votes += 1
                    voted_fraud_models.append(name)
                    
                committee_details[name] = {
                    "score": round(float(prob), 4),
                    "threshold": round(preds["threshold"], 4),
                    "vote_fraud": bool(is_fraud)
                }
            
            # Sub-Rotina de Desastre
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
            
            final_decision = "APROVAR"
            confidence = "ALTA (Seguro)"
            
            # Verificação de Flagrante Massivo Limitador de Fraude
            if fraud_votes >= majority_threshold:
                final_decision = "BLOQUEAR"
                confidence = "ALTA (Unanimidade/Maioria Clara)" if fraud_votes == total_active_models else "MÉDIA (Maioria)"
            elif fraud_votes > 0:
                # Regra Financeira de Custo-Oportunidade (Veto da MLP Sniper)
                if len(voted_fraud_models) == 1 and voted_fraud_models[0] == 'mlp':
                    final_decision = "REVISÃO MANUAL"
                    confidence = "MÉDIA (Veto do Campeão de Precisão - MLP alerta risco grave)"
                else: # Singularidade de árvore duvidosa sem poder
                    final_decision = "APROVAR"
                    confidence = "MÉDIA (Divergência menor - Risco assumido)"

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
