FROM python:3.12-slim

# Evita prompts interativos durante a instalação de pacotes do SO
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências de sistema necessárias para os pacotes Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    libgraphviz-dev \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências Python primeiro (aproveita cache do Docker)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copia o código-fonte
COPY src/ ./src/
COPY main.py .

# Copia os dados brutos necessários para o pipeline
COPY data/raw/ ./data/raw/

# Cria diretórios de saída que serão gerados em runtime
RUN mkdir -p data/processed models reports/figures

# Usuário não-root para segurança
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "main.py"]
CMD []
