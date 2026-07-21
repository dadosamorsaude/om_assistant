FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Instalar dependências mínimas do sistema operacional (se necessário)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de dependências compiladas
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte da aplicação
COPY . .

# Expõe a porta padrão (o Render mapeia e direciona o tráfego usando a variável PORT)
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["python", "api.py"]
