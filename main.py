# ==============================================================================
# ARQUIVO: main.py
#
# OBJETIVO:
#   Atuar como o orquestrador principal (Maestro) do Pipeline de Machine Learning.
#   Controla a execução sequencial das etapas: limpeza, engenharia de dados, 
#   análise exploratória, treinamento de modelos e inferência.
#
# PARTE DO SISTEMA:
#   Orquestrador Central / Entrypoint.
#
# RESPONSABILIDADES:
#   - Limpar artefatos antigos para reprodutibilidade.
#   - Chamar os módulos em ordem lógica (Data -> EDA -> Models -> Predictions).
#   - Receber e decodificar argumentos de linha de comando (CLI).
#   - Treinar os modelos na seleção do usuário.
#
# COMUNICAÇÃO:
#   - Importa constantes e utilitários globais de `src/config.py`.
#   - Centraliza e excuta chamadas das sub-pastas `src/data`, `src/models` e `src/visualization`.
# ==============================================================================

import argparse
import sys
import shutil
from pathlib import Path

# Adiciona raiz ao path para garantir imports corretos
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# Tenta importar os modulos. Se falhar, avisa o usuario qual arquivo esta faltando.
try:
    from src.config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RAW_DATA_PATH
    from src.data.make_dataset import load_and_split_data
    from src.visualization.generate_eda_report import EDAReporter
    from src.models.compare_models import compare_algorithms
    from src.models.trainers.reg_log_model import train_logistic_regression
    from src.models.trainers.random_forest_model import train_random_forest
    from src.models.trainers.xgboost_model import train_xgboost
    from src.models.trainers.decision_tree_model import train_decision_tree
    from src.models.trainers.mlp_model import train_mlp
    from src.models.trainers.isolation_forest_model import train_isolation_forest
    from src.visualization.visualize import evaluate
    from src.serving.predict_model import predict_sample
except ImportError as e:
    print(f"❌ ERRO CRITICO DE IMPORTACAO: {e}")
    print("Verifique se todos os arquivos estao nas pastas corretas dentro de 'src/'.")
    sys.exit(1)

# Imports opcionais (modelos que podem nao estar disponiveis)
try:
    from src.models.trainers.lightgbm_model import train_lightgbm
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from src.models.trainers.stacking_model import train_stacking
    HAS_STACKING = True
except ImportError:
    HAS_STACKING = False


def reset_project_artifacts():
    """
    Remove todos os artefatos gerados (processados, modelos e relatorios)
    para garantir uma execucao limpa e reprodutivel.
    """
    print("\n🧹 [MAESTRO] Iniciando limpeza de artefatos antigos...")
    
    dirs_to_clean = [
        PROCESSED_DATA_DIR, 
    ]
    
    for directory in dirs_to_clean:
        if directory.exists():
            try:
                for item in directory.glob("*"):
                    if item.name == ".gitkeep":
                        continue
                    
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print(f"   - Limpo: {directory.name}")
            except Exception as e:
                print(f"⚠️ Aviso: Nao foi possivel limpar {directory}: {e}")
                    
    print("✨ Limpeza concluida! Ambiente pronto.")


def main():
    """
    Funcao Principal (O Maestro).
    Orquestra a execucao de todo o pipeline de dados na ordem logica.
    """
    parser = argparse.ArgumentParser(description="🛡️ Fraud Sentinel - Maestro (Pipeline Orchestrator)")
    
    # Flags de Controle
    parser.add_argument("--no-reset", action="store_true", help="Inibe a limpeza inicial dos arquivos.")
    parser.add_argument("--skip-eda", action="store_true", help="Pula a etapa de Analise Exploratoria.")
    parser.add_argument("--compare-models", action="store_true", help="Executa o torneio de modelos (Demorado).")
    parser.add_argument("--predict", action="store_true", help="Roda uma simulacao de predicao no final.")
    parser.add_argument(
        "--models", type=str, default="all",
        help=(
            "Modelos a rodar separados por virgula. "
            "Opcoes: logreg, dt, rf, xgb, mlp, if, lgbm, stacking. "
            "Default: all (todos exceto stacking, que requer modelos pre-treinados). "
            "Exemplo: --models xgb,lgbm,stacking"
        )
    )
    
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
        print("\n🎨 [MAESTRO] 2. Movimento: Analise Exploratoria (generate_eda_report.py)")
        reporter = EDAReporter(RAW_DATA_PATH)
        reporter.run()
    else:
        print("\n⏩ [MAESTRO] Pulando EDA (--skip-eda)...")

    # 4. MODEL COMPARISON
    if args.compare_models:
        print("\n🥊 [MAESTRO] 3. Movimento: Torneio de Modelos (compare_models.py)")
        print("   Esta etapa pode levar varios minutos...")
        compare_algorithms()
    else:
        print("\n⏩ [MAESTRO] Pulando Torneio de Modelos (Padrao).")

    # 5. MODEL TRAINING & EVALUATION
    print("\n🧠 [MAESTRO] 4. Movimento: Treinamento & Avaliacao e Selecao de Modelos")
    
    # Parse models argument
    if args.models == "all":
        # Por default, roda todos EXCETO stacking (precisa dos modelos base prontos)
        selected_models = ["logreg", "dt", "rf", "xgb", "mlp", "if"]
        if HAS_LIGHTGBM:
            selected_models.append("lgbm")
    else:
        selected_models = [m.strip().lower() for m in args.models.split(",")]

    # -------------------------------------------------------------------------
    # MODELOS INDIVIDUAIS
    # -------------------------------------------------------------------------
    
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
        evaluate(model_name="if")
    
    # 4.7 LightGBM (NOVO)
    if "lgbm" in selected_models:
        if HAS_LIGHTGBM:
            print("\n📌 [SUB-TAREFA] 4.7. LightGBM")
            train_lightgbm()
            evaluate(model_name="lgbm")
        else:
            print("\n⚠️ LightGBM nao disponivel. Instale: pip install lightgbm")
    
    # -------------------------------------------------------------------------
    # ENSEMBLE (deve rodar APOS os modelos individuais)
    # -------------------------------------------------------------------------
    
    # 4.8 Stacking Ensemble (NOVO)
    if "stacking" in selected_models:
        if HAS_STACKING:
            print("\n📌 [SUB-TAREFA] 4.8. Stacking Ensemble (Meta-Modelo)")
            print("   Nota: Requer modelos base (xgb, rf) ja treinados.")
            train_stacking()
            evaluate(model_name="stacking")
        else:
            print("\n⚠️ Stacking nao disponivel.")

    # 6. PREDICTION (Opcional)
    if args.predict:
        print("\n🔮 [MAESTRO] 6. Movimento: Simulacao de Producao (predict_model.py)")
        pred_model = "xgb" if "xgb" in selected_models else selected_models[0]
        predict_sample(model_name=pred_model)

    print("\n🏁 [MAESTRO] Sinfonia Concluida! Seu projeto foi executado com sucesso.")
    print(f"📊 Verifique os relatorios em: {REPORTS_DIR}")


if __name__ == "__main__":
    main()

# ==============================================================================
# COMO EXECUTAR ESTE ARQUIVO (MANUAL DE USO)
# ==============================================================================
#
# 1. Execucao Padrao (Recomendado para primeira vez):
#    Limpa artefatos antigos, roda EDA, treina TODOS os modelos otimizados.
#    > python main.py
#
# 2. Execucao Rapida (Sem EDA):
#    Pula a analise exploratoria e vai direto pro treino.
#    > python main.py --skip-eda
#
# 3. Execucao Completa com Ensemble:
#    Treina modelos base e depois combina no Stacking.
#    > python main.py --skip-eda --models xgb,rf,lgbm,logreg,stacking
#
# 4. Apenas XGBoost + LightGBM + Stacking:
#    > python main.py --skip-eda --no-reset --models xgb,lgbm,stacking
#
# 5. Execucao de Manutencao (Sem Limpeza):
#    Nao apaga os arquivos processados.
#    > python main.py --no-reset
#
# 6. Execucao com Simulacao de Predicao:
#    > python main.py --predict
#
# 7. Combinando Flags (Exemplo):
#    > python main.py --skip-eda --no-reset --models xgb,lgbm --predict
# ==============================================================================