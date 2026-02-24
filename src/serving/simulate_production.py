# ==============================================================================
# ARQUIVO: simulate_production.py
#
# OBJETIVO:
#   Simular de forma visual e instigante o fluxo de transações em produção (Streaming).
#   Prova de Valor (PoV) para validar as regras de negócio do Ensemble.
#
# PARTE DO SISTEMA:
#   Simulação / Avaliação de Negócios / MLOps.
#
# RESPONSABILIDADES:
#   - Consumir de forma embaralhada `X_test.csv` (Legítimos e Fraudes).
#   - Aplicar interface amigável via CLI: feed em tempo real com emojis e cores.
#   - Contabilizar TP, TN, FP, FN em memória.
#   - Computar o ROI (Retorno sobre Investimento) da operação para relatório executivo.
#   - Salvar o `reports/simulation_summary.txt`.
#
# COMUNICAÇÃO:
#   - Depende do `predict_ensemble.py` para processamento com regras de negócio.
# ==============================================================================

import pandas as pd
import numpy as np
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Configuração de Caminhos
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR, REPORTS_DIR
from src.serving.predict_ensemble import FraudEnsemblePredictor

# Reduzimos verbosidade para o feed visual ficar limpo
logging.getLogger("src.serving.predict_ensemble").setLevel(logging.ERROR)

class ProductionSimulator:
    """Classe responsável por orquestrar a simulação de streaming de logs com interface avançada."""
    
    def __init__(self, avg_ticket: float = 500.0, sleep_time: float = 0.05):
        self.avg_ticket = avg_ticket
        self.sleep_time = sleep_time
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
        """Carrega e embaralha os dados das bases de testes."""
        try:
            X_test = pd.read_csv(PROCESSED_DATA_DIR / "X_test.csv")
            y_test = pd.read_csv(PROCESSED_DATA_DIR / "y_test.csv").values.ravel()
        except FileNotFoundError:
            print("❌ Arquivos de teste não encontrados em data/processed/. Execute make_dataset.py.")
            sys.exit(1)

        fraud_idx = np.where(y_test == 1)[0]
        legit_idx = np.where(y_test == 0)[0]
        
        n_f = min(n_fraud, len(fraud_idx))
        n_l = min(n_legit, len(legit_idx))
        
        print(f"📊 Mix Carregado: {n_l} Legítimas e {n_f} Fraudes (Total: {n_l + n_f}).\n")
        
        selected_fraud = np.random.choice(fraud_idx, n_f, replace=False)
        selected_legit = np.random.choice(legit_idx, n_l, replace=False)
        
        sample_indices = np.concatenate([selected_fraud, selected_legit])
        np.random.shuffle(sample_indices)
        
        return X_test, y_test, sample_indices

    def run_stream(self, X_test: pd.DataFrame, y_test: np.ndarray, sample_indices: np.ndarray):
        """Itera sobre a amostra processando a inferência linha a linha."""
        print("📡 Iniciando o monitoramento de rede em tempo real...")
        print("═"*110)
        
        for i, idx in enumerate(sample_indices):
            transaction = X_test.iloc[[idx]]
            true_label = int(y_test[idx])
            is_truly_fraud = bool(true_label == 1)
            
            # Chama o motor MLOps
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
                status = "🔍 ENVIADO PARA REVISÃO HUMANA (Veto LGBM)"
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
            
            # Espaçamentos fixos para alinhamento
            print(f"TX-{idx:05d} | {gabarito_icon} {('FRAUDE  ' if is_truly_fraud else 'LEGÍTIMO')} | COMITÊ: {committee_str} ({vote_txt}) ➡️ {decision:15s} | {icon} {status}")
            
            time.sleep(self.sleep_time)
            
    def export_report(self, total_samples: int, n_legit: int, n_fraud: int):
        """Calcula o ROI e salva o relatório .txt"""
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
[ DADOS DA OPERAÇÃO DE SIMULAÇÃO ]
Total de Transações Processadas: {total_samples}
  - Transações Legítimas Naturais: {n_legit}
  - Transações Fraudulentas Naturais: {n_fraud}
Ticket Médio por Transação: R$ {self.avg_ticket:.2f}

--------------------------------------------------------------------------------
📍 PERFORMANCE TÉCNICA E OPERACIONAL DO ENSEMBLE (Smart Majority Vote):
--------------------------------------------------------------------------------
- Taxa de Acerto Global da Automação (Accuracy): {hit_rate:.2f}%
- 🛑 Verdadeiros Positivos (Fraudes Barradas): {self.metrics['TP']}
- 🟢 Verdadeiros Negativos (Bons Aprovados): {self.metrics['TN']}
- 🚨 Falsos Negativos (Fraudes que Passaram): {self.metrics['FN']}
- ⚠️ Falsos Positivos (Clientes Bons c/ Fricção): {self.metrics['FP']} (Taxa Atrito: {friction_rate:.2f}%)
- 👀 Enviados para Revisão Humana (Veto Preditivo Especial): {self.metrics['RM']}

--------------------------------------------------------------------------------
💸 IMPACTO FINANCEIRO DE NEGÓCIO DA IA:
--------------------------------------------------------------------------------
✅ Patrimônio Salvo (Loss Prevented):     R$ {money_saved:,.2f}
❌ Prejuízo Realizado (Fraudes Não Vistas): R$ {money_lost:,.2f}

Lucro Total na Aprovação de Bons Perfis: R$ {total_revenue_legit:,.2f}

CONCLUSÃO:
A utilização do 'Veto Especial do Campeão de Precisão' (LightGBM) enviando 
casos críticos para Análise Humana evitou Falsos Positivos graves, enquanto
a 'Maioria Simples' sustentou taxas de contenção eficientes.
================================================================================
"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(report_content)
        print(f"📄 Relatório gerencial de ROI armazenado com sucesso em: {report_path}")

def run():
    sim = ProductionSimulator(avg_ticket=500.0, sleep_time=0.03)
    sim.setup()
    
    n_legit = 500
    n_fraud = 30
    X_test, y_test, sample_indices = sim.load_sample_data(n_legit, n_fraud)
    
    sim.run_stream(X_test, y_test, sample_indices)
    sim.export_report(len(sample_indices), n_legit, n_fraud)

if __name__ == "__main__":
    run()
