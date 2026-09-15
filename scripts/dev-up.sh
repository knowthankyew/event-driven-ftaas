#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

echo "========================================================"
echo "🚀 Starting FTaaS Local Infrastructure (Phase 1)"
echo "========================================================"

# 1. Ensure required data directories exist
echo "📁 Setting up local directory layout..."
mkdir -p data/datasets data/storage mlflow_data/artifacts

# 2. Check Docker daemon availability
if ! docker info >/dev/null 2>&1; then
  echo "⚠️ Docker daemon is not responding. Attempting to start Docker Desktop on macOS..."
  open -a Docker || true
  echo "⏳ Waiting for Docker daemon to become responsive..."
  for i in {1..30}; do
    if docker info >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
fi

if ! docker info >/dev/null 2>&1; then
  echo "❌ Error: Docker daemon is not running. Please start Docker and re-run."
  exit 1
fi

# 3. Bring up containers
echo "🐳 Starting RabbitMQ and MLflow via Docker Compose..."
docker compose up -d

# 4. Wait for RabbitMQ readiness
echo "⏳ Waiting for RabbitMQ broker (port 5672 / 15672)..."
for i in {1..30}; do
  if docker exec ftaas-rabbitmq rabbitmq-diagnostics -q ping >/dev/null 2>&1; then
    echo "✅ RabbitMQ is healthy and ready!"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "❌ Timeout waiting for RabbitMQ."
    docker logs ftaas-rabbitmq --tail 20
    exit 1
  fi
  sleep 2
done

# 5. Wait for MLflow readiness
echo "⏳ Waiting for MLflow tracking server (port 5001)..."
for i in {1..30}; do
  if curl -s -f http://localhost:5001/health >/dev/null 2>&1 || curl -s -f http://localhost:5001/ >/dev/null 2>&1; then
    echo "✅ MLflow server is healthy and ready!"
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "❌ Timeout waiting for MLflow."
    docker logs ftaas-mlflow --tail 20
    exit 1
  fi
  sleep 2
done

# 6. Seed sample domain datasets
echo "🌱 Seeding sample domain dataset..."
python3 scripts/seed_dataset.py

echo ""
echo "========================================================"
echo "🎉 FTaaS Local Infrastructure is READY!"
echo "========================================================"
echo "📊 MLflow UI:       http://localhost:5001"
echo "🐰 RabbitMQ UI:     http://localhost:15672 (guest / guest)"
echo "🐰 RabbitMQ AMQP:   amqp://guest:guest@localhost:5672"
echo "📂 Sample Dataset:  data/datasets/sample-financial-sentiment.jsonl"
echo "========================================================"
