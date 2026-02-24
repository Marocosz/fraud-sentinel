# ==============================================================================
# ARQUIVO: predict_model.py
#
# OBJETIVO:
#   Simular um ambiente de produção (inferência em tempo real) para detecção de fraudes.
#   Recebe dados brutos (simulados do conjunto de teste), aplica o modelo treinado
#   e decide se aprova ou bloqueia a transação com base em Regras de Negócio.
#
# PARTE DO SISTEMA:
#   Módulo de Inferência / Serving (Model Scoring).
#
# RESPONSABILIDADES:
#   - Carregar o modelo final otimizado (.pkl).
#   - Carregar um subconjunto de dados blind (Teste) para simular novas transações.
#   - Calcular o Score de Risco (Probabilidade) para cada transação.
#   - Aplicar Thresholds de Negócio (ex: >0.8 = Bloqueio) sobre o score estatístico.
#   - Exibir logs detalhados para auditoria (ID da transação, Decisão, Gabarito).
#
# COMUNICAÇÃO:
#   - Lê: models/*_best_model.pkl (Modelo serializado)
#   - Lê: data/processed/X_test.csv, y_test.csv (Dados para simulação)
# ==============================================================================

import pandas as pd
import numpy as np
import joblib
import sys
import logging
import random
from pathlib import Path

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Imports do Projeto
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# Configuração de Logs (Formato de Produção)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [INFERENCE] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_inference_artifacts(model_name="logreg"):
    """
    Função Helper de Descarregamento de Artefatos em Lote (Memory-Loader).
    
    - O que ela faz: Vai até a pasta da arquitetura e traz em memória `Pipeline` Picklizado, 
      Array de Features formatada e Alvo.
    - O que retorna: tuple: (modelo_carregado, dataframe_X, array_y)
    - Por que carregar dados aqui (mockup de produção)? Em um sistema real transacional, receberíamos um JSON via API REST.
      Como estamos simulando um Teste Cego para provar pro board executivo, carregamos o conjunto 
      separado na etapa 1 para sortear e bater.
    """
    try:
        # Construção dinâmica do caminho do modelo (padrão definido em train_model.py)
        model_path = MODELS_DIR / f"{model_name}_best_model.pkl"
        
        # Validação de existência do arquivo (Evita crash feio se o usuário não treinou)
        if not model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")

        logger.info(f"Carregando modelo: {model_path.name}...")
        model = joblib.load(model_path)
        
        logger.info(f"Carregando dados para simulação (Blind Test Set)...")
        # Carregamos X_test (atributos) e y_test (gabarito)
        # O gabarito serve apenas para mostrar no log se o modelo "Acertou" ou "Errou",
        # num cenário real de produção, obviamente não teríamos o y_test.
        X_test = pd.read_pickle(PROCESSED_DATA_DIR / "X_test.pkl")
        y_test = pd.read_pickle(PROCESSED_DATA_DIR / "y_test.pkl").values.ravel()
        
        return model, X_test, y_test
    
    except Exception as e:
        logger.error(f"Erro crítico ao carregar artefatos: {e}")
        sys.exit(1)

def load_threshold(model_name="xgb"):
    """
    Carrega o threshold otimizado (F1-Score) salvo durante o treinamento.
    Se não encontrar, usa o padrão 0.5.
    """
    try:
        threshold_path = MODELS_DIR / f"{model_name}_threshold.txt"
        if threshold_path.exists():
            with open(threshold_path, "r") as f:
                threshold = float(f.read().strip())
            logger.info(f"🎯 Threshold Otimizado carregado: {threshold:.4f}")
            return threshold
        else:
            logger.warning("⚠️ Threshold otimizado não encontrado. Usando padrão 0.5.")
            return 0.5
    except Exception as e:
        logger.warning(f"Erro ao ler threshold ({e}). Usando padrão 0.5.")
        return 0.5

def explain_prediction(model, sample_row, feature_names):
    """
    Placeholder para XAI (Explainable AI).
    
    Objetivo:
    - Explicar ao analista humano POR QUE o sistema bloqueou a transação.
    - É uma exigência regulatória em muitos países (Direito à explicação).
    
    Nota Técnica:
    - Atualmente implementado como pass, pois requer acesso complexo aos nomes
      das features após o pipeline de transformação (OneHotEncoding, etc).
    - Em versões futuras, implementar SHAP ou LIME aqui.
    """
    explanation = []
    
    # Verifica se é LogReg (modelo linear tem coeficientes interpretáveis diretamente)
    if hasattr(model.named_steps['model'], 'coef_'):
        classifier = model.named_steps['model']
        coefs = classifier.coef_[0]
        
        # TODO: Mapear feature_name -> coeficiente * valor_da_instancia
        # Isso revelaria quais campos específicos (ex: 'valor_alto', 'ip_estrangeiro')
        # puxaram o score para cima.
        pass

def predict_sample(model_name="xgb", n_samples=5):
    """
    Função Principal de Simulação.
    Cria um loop de requisições simula o comportamento do motor de decisão.
    
    Args:
        model_name (str): Qual modelo usar (default: 'logreg').
        n_samples (int): Quantas transações processar nesta bateria de teste.
    """
    model, X_test, y_test = load_inference_artifacts(model_name)
    threshold = load_threshold(model_name)
    
    logger.info(f"🚀 Iniciando Simulação de Produção ({n_samples} transações)...")
    logger.info(f"⚙️  Critério de Decisão Otimizado (F1-MAX): > {threshold:.4f}")
    print("-" * 80)
    
    # --------------------------------------------------------------------------
    # ESTRATÉGIA DE AMOSTRAGEM
    # Para fins de demonstração, não queremos apenas zeros (legítimos), pois
    # 99% dos dados são legítimos. Se pegarmos aleatório puro, talvez não
    # vejamos nenhuma fraude.
    # Por isso, forçamos uma mistura balanceada de índices de fraude e não-fraude.
    # --------------------------------------------------------------------------
    fraud_indices = np.where(y_test == 1)[0]
    legit_indices = np.where(y_test == 0)[0]
    
    # Tenta garantir 50% de fraudes na amostra para visualização
    n_frauds = min(len(fraud_indices), max(1, n_samples // 2))
    n_legits = n_samples - n_frauds
    
    sample_indices = np.concatenate([
        np.random.choice(fraud_indices, n_frauds, replace=False),
        np.random.choice(legit_indices, n_legits, replace=False)
    ])
    np.random.shuffle(sample_indices) # Embaralha para simular ordem de chegada aleatória
    
    # --------------------------------------------------------------------------
    # LOOP DE INFERÊNCIA
    # Simula o recebimento de requests um a um.
    # --------------------------------------------------------------------------
    for i, idx in enumerate(sample_indices):
        # Extrai os dados da transação como um DataFrame de 1 linha
        transaction_data = X_test.iloc[[idx]]
        true_label = y_test[idx] # Gabarito (apenas para validação visual)
        
        # 1. SCORE DE RISCO (Probabilidade)
        # O modelo retorna [prob_legitimo, prob_fraude]. Pegamos o índice [1].
        # Este é o valor mais importante para o banco.
        proba = model.predict_proba(transaction_data)[0, 1]
        
        # 2. MOTOR DE DECISÃO (Business Rules Layer)
        # O modelo diz "Risco 70%". O Negócio decide o que fazer com isso.
        # Estes thresholds são configuráveis e mudam de acordo com o apetite de risco do banco.
        decision = "APROVADA"
        
        # Lógica adaptativa baseada no Threshold Otimizado (T)
        if proba > threshold:
            decision = "🔴 BLOQUEADO (ALTO RISCO)"
            icon = "🚨"
        elif proba > (threshold * 0.8):
            decision = "⚠️ REVISÃO MANUAL (MÉDIO RISCO)"
            icon = "👀"
        else:
            decision = "🟢 APROVADO (BAIXO RISCO)"
            icon = "✅"
            
        print(f"ID Transação: {idx:06d} | Verdadeiramante: {'🔴 Fraude' if true_label == 1 else '🟢 Legítima'}")
        print(f"   Score de Risco (Modelo): {proba:.4f} ({proba*100:.1f}%)")
        print(f"   Decisão do Sistema: {decision} {icon}")
        
        # Verifica acerto (Considerando o threshold otimizado)
        pred_label = 1 if proba > threshold else 0
        match = "✅ ACERTOU" if pred_label == true_label else "❌ ERROU"
        print(f"   Resultado: O modelo {match}")
        print("-" * 80)

if __name__ == "__main__":
    predict_sample()
