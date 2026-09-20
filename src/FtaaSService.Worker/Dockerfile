# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY ml/src/FtaaSService.Worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker application source
COPY ml/src/FtaaSService.Worker/ /app/

# Create data and artifact directories with non-root permissions
RUN useradd -m -u 1000 appuser && \
    mkdir -p /data/artifacts /app/data && \
    chown -R appuser:appuser /app /data

USER appuser

ENV DATA_ROOT=/data

CMD ["python", "consumer.py"]
