# AI-Trader Crypto Trading Agent
# Multi-model cryptocurrency trading with Alpaca Paper Trading

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for logs and data
RUN mkdir -p /app/logs /app/data/agent_data_crypto

# Environment variables (override with docker run -e or docker-compose)
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command - run scheduled trading
CMD ["python", "main_alpaca.py", "configs/alpaca_crypto_config.json", "--scheduled"]
