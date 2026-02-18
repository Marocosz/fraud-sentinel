import argparse
import sys
import shutil
from pathlib import Path

# Adiciona raiz ao path para garantir imports corretos
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Tenta importar os módulos. Se falhar, avisa o usuário qual arquivo está faltando.
try:
    from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RAW_DATA_PATH
    from src.data.make_dataset import load_and_split_data
    from src.visualization.generate_eda_report import EDAReporter
    from src.models.compare_models import compare_algorithms
    from src.models.reg_log_model import train_logistic_regression
    from src.models.random_forest_model import train_random_forest
    from src.models.xgboost_model import train_xgboost
    from src.models.decision_tree_model import train_decision_tree
    from src.models.mlp_model import train_mlp
    from src.models.isolation_forest_model import train_isolation_forest
    from src.visualization.visualize import evaluate
    from src.models.predict_model import predict_sample
except ImportError as e:
    print(f"❌ ERRO CRÍTICO DE IMPORTAÇÃO: {e}")
    print("Verifique se todos os arquivos (make_dataset.py, reg_log_model.py, etc) estão nas pastas corretas dentro de 'src/'.")
    sys.exit(1)

def reset_project_artifacts():
    """
    Remove todos os artefatos gerados (processados, modelos e relatórios)
    para garantir uma execução limpa e reprodutível.
    """
    print("\n🧹 [MAESTRO] Iniciando limpeza de artefatos antigos...")
    
    first_run_file = PROCESSED_DATA_DIR / ".gitkeep"
    
    # Lista de diretórios a limpar
    # OBSERVACAO: Removemos MODELS_DIR da limpeza para permitir histórico de experimentos.
    dirs_to_clean = [
        PROCESSED_DATA_DIR, 
        # CLAUSULA DE SEGURANÇA: Não limpamos mais a pasta de modelos para manter histórico.
        # MODELS_DIR, 
        # REPORTS_DIR / "data", # Opcional: Se quiser limpar os CSVs de relatório
        # FIGURES_DIR # Opcional: Se quiser limpar os gráficos
    ]
    
    for directory in dirs_to_clean:
        if directory.exists():
            try:
                # Remove todo o conteúdo da pasta, mas mantém a pasta em si
                for item in directory.glob("*"):
                    if item.name == ".gitkeep":
                        continue
                    
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print(f"   - Limpo: {directory.name}")
            except Exception as e:
                print(f"⚠️ Aviso: Não foi possível limpar {directory}: {e}")
                    
    print("✨ Limpeza concluída! Ambiente pronto.")

def main():
    """
    Função Principal (O Maestro).
    Orquestra a execução de todo o pipeline de dados na ordem lógica.
    """
    parser = argparse.ArgumentParser(description="🛡️ Fraud Sentinel - Maestro (Pipeline Orchestrator)")
    
    # Flags de Controle
    parser.add_argument("--no-reset", action="store_true", help="Inibe a limpeza inicial dos arquivos.")
    parser.add_argument("--skip-eda", action="store_true", help="Pula a etapa de Análise Exploratória.")
    parser.add_argument("--compare-models", action="store_true", help="Executa o torneio de modelos (Demorado).")
    parser.add_argument("--predict", action="store_true", help="Roda uma simulação de predição no final.")
    parser.add_argument("--models", type=str, default="all", help="Modelos a rodar separados por vírgula (ex: xgb,rf). Opções: logreg, dt, rf, xgb, mlp, if. Default: all")
    
    args = parser.parse_args()

    print("\n🎼 [MAESTRO] Bem-vindo ao Fraud Sentinel Pipeline.")
    print("===================================================")

    # 1. RESET (Limpeza)
    if not args.no_reset:
        reset_project_artifacts()
    else:
        print("⚠️ [MAESTRO] Limpeza pulada (--no-reset).")

    # 2. DATA ENGINEERING
    print("\n🎹 [MAESTRO] 1. Movimento: Data Engineering (make_dataset.py)")
    load_and_split_data()

    # 3. EDA
    if not args.skip_eda:
        print("\n🎨 [MAESTRO] 2. Movimento: Análise Exploratória (generate_eda_report.py)")
        reporter = EDAReporter(RAW_DATA_PATH)
        reporter.run()
    else:
        print("\n⏩ [MAESTRO] Pulando EDA (--skip-eda)...")

    # 4. MODEL COMPARISON
    if args.compare_models:
        print("\n🥊 [MAESTRO] 3. Movimento: Torneio de Modelos (compare_models.py)")
        print("   Esta etapa pode levar vários minutos...")
        compare_algorithms()
    else:
        print("\n⏩ [MAESTRO] Pulando Torneio de Modelos (Padrão).")

    # 5. MODEL TRAINING & EVALUATION
    print("\n🧠 [MAESTRO] 4. Movimento: Treinamento & Avaliação e Seleção de Modelos")
    
    # Parse models argument
    if args.models == "all":
        selected_models = ["logreg", "dt", "rf", "xgb", "mlp", "if"]
    else:
        selected_models = [m.strip().lower() for m in args.models.split(",")]

    # 4.1 Logistic Regression
    if "logreg" in selected_models:
        print("\n📌 [SUB-TAREFA] 4.1. Logistic Regression")
        train_logistic_regression()
        evaluate(model_name="logreg")

    # 4.2 Decision Tree
    if "dt" in selected_models:
        print("\n📌 [SUB-TAREFA] 4.2. Decision Tree")
        train_decision_tree()
        evaluate(model_name="dt")

    # 4.3 Random Forest
    if "rf" in selected_models:
        print("\n📌 [SUB-TAREFA] 4.3. Random Forest")
        train_random_forest()
        evaluate(model_name="rf")

    # 4.4 XGBoost
    if "xgb" in selected_models:
        print("\n📌 [SUB-TAREFA] 4.4. XGBoost")
        train_xgboost()
        evaluate(model_name="xgb")
        
    # 4.5 MLP (Neural Network)
    if "mlp" in selected_models:
        print("\n📌 [SUB-TAREFA] 4.5. MLP Neural Network")
        train_mlp()
        evaluate(model_name="mlp")
        
    # 4.6 Isolation Forest
    if "if" in selected_models:
        print("\n📌 [SUB-TAREFA] 4.6. Isolation Forest (Anomaly Detection)")
        train_isolation_forest()
        # Nota: IF precisa de tratamento especial no evaluate se não tiver predict_proba,
        # mas nosso Wrapper implementa predict_proba, então deve funcionar!
        evaluate(model_name="if")

    # 7. PREDICTION (Opcional)
    if args.predict:
        print("\n🔮 [MAESTRO] 6. Movimento: Simulação de Produção (predict_model.py)")
        # Usa o primeiro modelo da lista selecionada como default para predição, ou xgb se disponível
        pred_model = "xgb" if "xgb" in selected_models else selected_models[0]
        predict_sample(model_name=pred_model)

    print("\n🏁 [MAESTRO] Sinfonia Concluída! Seu projeto foi executado com sucesso.")
    print(f"📊 Verifique os relatórios em: {REPORTS_DIR}")

if __name__ == "__main__":
    main()

# ==============================================================================
# COMO EXECUTAR ESTE ARQUIVO (MANUAL DE USO)
# ==============================================================================
#
# 1. Execução Padrão (Recomendado para primeira vez):
#    Limpa artefatos antigos, roda EDA, treina TODOS os modelos otimizados e avalia cada um.
#    > python main.py
#
# 2. Execução Rápida (Sem EDA):
#    Pula a análise exploratória (que pode demorar) e vai direto pro treino dos modelos.
#    > python main.py --skip-eda
#
# 3. Execução Completa (Com Torneio de Modelos):
#    Roda também o benchmark inicial de algoritmos antes de treinar os finais.
#    > python main.py --compare-models

#
# 4. Execução de Manutenção (Sem Limpeza):
#    Não apaga os arquivos processados (útil se você já rodou o make_dataset.py).
#    > python main.py --no-reset
#
# 5. Execução com Simulação de Predição:
#    No final, roda um teste de inferência com dados aleatórios simulando produção.
#    > python main.py --predict
#
# Combinando Flags (Exemplo):
#    > python main.py --skip-eda --no-reset --predict
# ==============================================================================