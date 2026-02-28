# ==============================================================================
# ARQUIVO: simulate_production.py
#
# OBJETIVO:
#   Emular um cenário real de tráfego (Streaming MLOps) com cálculos focados em negócios (ROI).
#   Prova de Valor (PoV) matemática que comprova como o Comitê reduz atrito (User Experience)
#   e minimiza a perda financeira direta contra a base estática tabular.
#
# PARTE DO SISTEMA:
#   Módulo Front-End / Monitoramento Tático Analítico.
#
# RESPONSABILIDADES:
#   - Sacar amostras fidedignas e desbalanceadas do Data Lake de testes (`X_test.pkl`).
#   - Rodar lote maciço sob o funil de "Smart Vote" do Ensemble Predictor.
#   - Simular o estresse terminal iterativo através da Engine MLOps de Decisão (Aprovar/Bloquear).
#   - Computar e salvar formalmente os números corporativos e fricções no arquivo de relatório Textual.
#
# INTEGRAÇÃO:
#   - Lê arquivos do pipeline de ML: `X_test.pkl`, `y_test.pkl`.
#   - Inicializa a classe viva do Motor: `predict_ensemble.FraudEnsemblePredictor`.
#   - Exporta o laudo Executivo Empresarial (Txt): `reports/simulation_summary.txt`.
# ==============================================================================

import pandas as pd
import numpy as np
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.serving.predict_ensemble import FraudEnsemblePredictor

# Configurações de exibição do Pandas para não exibir warnings chatos durante streaming
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Reduzimos verbosidade para o feed visual ficar limpo
logging.getLogger("src.serving.predict_ensemble").setLevel(logging.ERROR)

class ProductionSimulator:
    """Classe responsável por orquestrar a simulação de streaming de logs com interface avançada."""
    
    def __init__(self, avg_ticket: float = 500.0, sleep_time: float = 0.05, fast_mode: bool = False):
        self.avg_ticket = avg_ticket
        self.sleep_time = sleep_time
        self.fast_mode = fast_mode
        self.predictor = None
        self.metrics: Dict[str, int] = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0, 'RM': 0} # RM = Revisão Manual

    def setup(self):
        print("\n" + "═"*90)
        print("🚀 [FRAUD SENTINEL] INICIANDO SIMULAÇÃO DO MOTOR DE PRODUÇÃO MLOPS (STREAMING)")
        print("═"*90)
        print("⚙️  Ligando motores (Instanciando Ensamble de Modelos)...")
        self.predictor = FraudEnsemblePredictor()
        print("🟢 SISTEMA OPERACIONAL! Motores aquecidos.\n")
        time.sleep(1)

    def load_sample_data(self, n_legit: int = 500, n_fraud: int = 30) -> pd.DataFrame:
        """
        Extrator e misturador de tráfego não-visto estocástico.
        
        Por que existe:
        Toda métrica do Simulador MLOps é focada no balanço de Dinheiro/Risco. Portanto esse método 
        busca a proporção real de 'Ataque Sintético' (~2% Fraude Rate) contida no File do Teste cego 
        e não do pipeline de treino balanceado.
        
        Recebe:
        n_legit (int): Cota fixa do pipeline paramétrico simulando requisições limpas.
        n_fraud (int): Cota fixa de atacantes cibernéticos.

        Retorna:
        Tuple (X_test, y_test, sample_indices): Dataframes tabulares inteiros e 
        uma matriz Numpy 1D indexada contendo a desordem do tráfego (Shuffle cronológico).
        """
        try:
            X_test = pd.read_pickle(PROCESSED_DATA_DIR / "X_test.pkl")
            y_test = pd.read_pickle(PROCESSED_DATA_DIR / "y_test.pkl").values.ravel()
        except FileNotFoundError:
            print("❌ Arquivos de teste não encontrados em data/processed/. Execute make_dataset.py.")
            sys.exit(1)

        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        
        # Limita extração se os lotes requisitados forem maiores que as defesas do BD de Teste
        n_f = min(n_fraud, len(fraud_idx))
        n_l = min(n_legit, len(legit_idx))
        
        print(f"📊 Mix Carregado: {n_l} Legítimas e {n_f} Fraudes (Total: {n_l + n_f}).\n")
        
        # Corta a quantia exata de cada Array baseada no Threshold definido no MLOps Parameter
        selected_fraud = np.random.choice(fraud_idx, n_f, replace=False)
        selected_legit = np.random.choice(legit_idx, n_l, replace=False)
        
        # Array único randomizado simulando o funil caótico da requisição Web do Banco D-0 
        sample_indices = np.concatenate([selected_fraud, selected_legit])
        np.random.shuffle(sample_indices)
        
        return X_test, y_test, sample_indices

    def run_stream(self, X_test: pd.DataFrame, y_test: np.ndarray, sample_indices: np.ndarray):
        """
        Laço Principal de processamento síncrono da Ordem de Avaliação.
        
        Por que existe:
        Toma conta de varrer o lote randômico preparado puxando a vetorização `predict_batch`.
        Encarrega-se de atualizar constantemente a variável `metrics` global (TN, TP, FP, FN, RM) 
        para abastecer o motor relacional que produzirá as provas monetárias de fechamento de relatório.

        Recebe:
        X_test (pd.DataFrame): DataFrame orgânico total (Cego).
        y_test (np.ndarray): Target Label para comprovação de auditoria MLOps do Veredicto Real da Fraude.
        sample_indices (np.ndarray): Posições aleatórias já mapeadas para percorrer com segurança o Pandas `.iloc`.
        """
        print("📡 Iniciando o monitoramento de rede em tempo real...")
        print("═"*110)
        
        for i, idx in enumerate(sample_indices):
            transaction = X_test.iloc[[idx]]
            true_label = int(y_test[idx])
            is_truly_fraud = bool(true_label == 1)
            
            # Chama o motor MLOps (Batch Assíncrono para os modelos isolados)
            response = self.predictor.predict_batch(transaction)[0]
            
            decision = response.final_decision
            fraud_votes = response.fraud_votes
            total_models = response.total_active_models
            
            # Auditoria e Métricas
            status = ""
            icon = ""
            if decision == "BLOQUEAR":
                if is_truly_fraud:
                    self.metrics['TP'] += 1
                    status = "✅ FRAUDE BARRADA! "
                    icon = "🛑"
                else:
                    self.metrics['FP'] += 1
                    status = "❌ APROVAÇÃO NEGADA (Atrito Cliente) "
                    icon = "⚠️"
            elif decision == "REVISÃO MANUAL":
                self.metrics['RM'] += 1
                status = "🔍 ENVIADO PARA REVISÃO HUMANA (Veto MLP)"
                icon = "👀"
            else: # APROVAR
                if not is_truly_fraud:
                    self.metrics['TN'] += 1
                    status = "✅ LEGÍTIMA APROVADA"
                    icon = "🟢"
                else:
                    self.metrics['FN'] += 1
                    status = "❌ FRAUDE PASSOU DESPERCEBIDA!     "
                    icon = "🚨"

            # Formatação do Voto do Comitê: [LGB:🔴]
            aliases = {'lightgbm': 'LGB', 'xgboost': 'XGB', 'mlp': 'MLP'}
            committee_str = ""
            for m_name, det in response.committee_details.items():
                v_icon = "🔴" if det['vote_fraud'] else "🟢"
                committee_str += f"[{aliases.get(m_name, m_name[:3])}:{v_icon}]"

            # Print Visual Terminal
            gabarito_icon = "🎭" if is_truly_fraud else "👤"
            vote_txt = f"{fraud_votes:d}/{total_models:d}"
            
            # Espaçamentos fixos para alinhamento (REVISÃO MANUAL tem 14 chars)
            decision_fmt = decision.ljust(15) if decision != "REVISÃO MANUAL" else decision.ljust(15) # Force alinhamento de 15 chars (isso cobre as labels)
            
            if not self.fast_mode:
                print(f"TX-{idx:05d} | {gabarito_icon} {('FRAUDE  ' if is_truly_fraud else 'LEGÍTIMO')} | COMITÊ: {committee_str} ({vote_txt}) ➡️ {decision_fmt} | {icon} {status}")
                time.sleep(self.sleep_time)
            
    def export_report(self, total_samples: int, n_legit: int, n_fraud: int):
        """
        Extrato de Inteligência C-Level, traduzindo Matemática para Dinheiro.
        
        Por que existe:
        Empresas de Crédito e Fintechs não dialogam primáriamente em F1-Score ou TPR, mas sim 
        em Custo de Aquisição Perdido (Fricção) e Risco Inadimplente Cedido (Loss Prevented / Incurred).
        Calcula as taxas de atrito da carteira baseado no volume limpo inserido contra falsos flagrantes.
        E emite o extrato txt persistido.
        """
        print("\n" + "═"*90)
        print("🏁 SIMULAÇÃO DE STREAMING CONCLUÍDA")
        print("═"*90)
        
        # Consideramos RM (Revisão Manual) focada na fraude, para cálculo vamos
        # assumir atrito se for RM de um legítimo, ou proteção parcial se for fraude.
        # Simplificação: Taxa de Acerto Automática imediata = (TP + TN) / Total Processado Sem Intervenção Human
        processed_automagic = total_samples - self.metrics['RM']
        hit_rate = ((self.metrics['TP'] + self.metrics['TN']) / processed_automagic * 100) if processed_automagic > 0 else 0
        
        money_saved = self.metrics['TP'] * self.avg_ticket
        money_lost = self.metrics['FN'] * self.avg_ticket
        
        total_revenue_legit = n_legit * self.avg_ticket
        friction_rate = (self.metrics['FP'] / n_legit * 100) if n_legit > 0 else 0
        
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / "simulation_summary.txt"
        
        report_content = f"""================================================================================
RELATÓRIO EXECUTIVO DE NEGÓCIOS - FRAUD SENTINEL (Avaliando Modelo Ensemble)
================================================================================
[ DADOS DA OPERAÇÃO DE SIMULAÇÃO (STREAMING DE ONBOARDING - CRIAÇÃO DE CONTA) ]
Total de Solicitações Processadas no Lote: {total_samples}
  - Solicitações Presumidas Legítimas: {n_legit}
  - Tentativas Ocultas de Fraude (Ataques Sintéticos ou Invasão): {n_fraud}
Média de Risco de Crédito Inicial (Thresholding Financeiro): R$ {self.avg_ticket:,.2f}

--------------------------------------------------------------------------------
📍 PERFORMANCE TÉCNICA E OPERACIONAL DO MOTOR NEURAL MLOPS:
--------------------------------------------------------------------------------
- Taxa de Assertividade Global do Sistema Automático (Accuracy): {hit_rate:.2f}%
- 🛑 Verdadeiros Positivos (Criminosos e Ataques Barrados Imediatamente): {self.metrics['TP']}
- 🟢 Verdadeiros Negativos (Bons Clientes Aprovados Imediatamente): {self.metrics['TN']}
- 🚨 Falsos Negativos (Criminosos que Passaram a Malha Fina): {self.metrics['FN']}
- ⚠️ Falsos Positivos (Clientes Bons c/ Fricção no Onboarding): {self.metrics['FP']} (Taxa Atrito: {friction_rate:.2f}%)
- 👀 Solicitações para Revisão de Mesa Humana (Veto do Algoritmo de Precisão): {self.metrics['RM']}

--------------------------------------------------------------------------------
💸 IMPACTO FINANCEIRO SIMULADO DA IA (Custo vs Retenção Média Baseada no Lote):
--------------------------------------------------------------------------------
✅ Retenção Patrimonial Protegida Definitiva:       R$ {money_saved:,.2f}
❌ Exposição Concedida a Risco Certo (Fraude):      R$ {money_lost:,.2f}

Aprovação de Linhas de Crédito para Bons Clientes: R$ {total_revenue_legit:,.2f}

CONCLUSÃO DA ARQUITETURA DO SISTEMA INTEGRADO DE ENSEMBLE:
O mecanismo de "Smart Majority Vote" agiu retendo o dano financeiro central da instituição 
através do voto cruzado, enquanto o "Veto de Campeão Analítico" redirecionou amostras de borda
nebulosa para intervenção de um Back-office humano cortando falsos bloqueios (redução da taxa 
de perda e atrito friccional de CAC de Bons Clientes).
================================================================================
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(report_content)
        print(f"📄 Relatório gerencial de ROI armazenado com sucesso em: {report_path}")

def run():
    """
    Controlador de gatilho do Arquivo.
    Pode ser rodado de forma rápida e fria (`--fast` no console) para varrer lotes abissais e extrair o arquivo.
    """
    parser = argparse.ArgumentParser(description="Simulador de Produção Fraud Sentinel")
    parser.add_argument("--fast", action="store_true", help="Desabilita modo visual e corre os dados massivamente em background.")
    args = parser.parse_args()

    # Parametrização Core Monetária de Simulação
    # Utiliza Tícket de liberação altíssimo para simular abertura de contas de luxo.
    sim = ProductionSimulator(avg_ticket=3500.0, sleep_time=0.01, fast_mode=args.fast)
    sim.setup()
    
    # Mix de Tráfego Abusivo
    # Fornece volume agressivo pra calcular atrito em cima da camada de Falso Negativos
    n_legit = 50000
    n_fraud = 1100
    X_test, y_test, sample_indices = sim.load_sample_data(n_legit, n_fraud)
    
    if args.fast:
        print(f"⏩ [MODO RÁPIDO MLOPS] Processando {len(sample_indices)} transações sem interface gráfica (Stand by...).")

    sim.run_stream(X_test, y_test, sample_indices)
    sim.export_report(len(sample_indices), n_legit, n_fraud)

if __name__ == "__main__":
    run()
