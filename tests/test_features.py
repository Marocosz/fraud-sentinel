# ==============================================================================
# ARQUIVO: test_features.py
#
# OBJETIVO:
#   Homologar a pipeline de transformações numéricas e categóricas.
#
# PARTE DO SISTEMA:
#   Testes Manuais/Unitários (Engenharia de features).
#
# RESPONSABILIDADES:
#   - Testar o comportamento esperado do robust scaling e one-hot encoding
#     evitando que colunas sejam perdidas acidentalmente.
#
# COMUNICAÇÃO:
#   - Interage instanciando funções e pacotes presentes no `build_features.py`.
# ==============================================================================

# Verifica se o One-Hot Encoding gerou as colunas certas
