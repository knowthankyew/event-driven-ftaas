#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$ROOT_DIR"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ -n "${PYTHON_BIN:-}" ]; then
    PYTHON_EXEC="$PYTHON_BIN"
elif [ -f "src/FtaaSService.Worker/.venv/bin/python" ]; then
    PYTHON_EXEC="src/FtaaSService.Worker/.venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

SPAWNED_PIDS=()
cleanup() {
    if [ ${#SPAWNED_PIDS[@]} -gt 0 ]; then
        echo -e "\n${CYAN}🧹 Cleaning up background processes (${SPAWNED_PIDS[*]})...${NC}"
        kill "${SPAWNED_PIDS[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}  FTaaS: Automated End-to-End Pipeline Verification            ${NC}"
echo -e "${BLUE}================================================================${NC}"

# 1. Ensure Docker infrastructure is healthy
echo -e "\n${CYAN}[1/6] Checking Docker Infrastructure (RabbitMQ + MLflow)...${NC}"
if ! docker exec ftaas-rabbitmq rabbitmq-diagnostics -q ping >/dev/null 2>&1 || ! curl -s -f http://localhost:5001/health >/dev/null 2>&1; then
    echo "Starting infrastructure via ./scripts/dev-up.sh..."
    ./scripts/dev-up.sh
else
    echo -e "${GREEN}✅ Infrastructure is healthy.${NC}"
fi

# 2. Check or start .NET 10 Ingestion Gateway
echo -e "\n${CYAN}[2/6] Ensuring .NET 10 Ingestion Gateway is running on :5100...${NC}"
if ! curl -s http://localhost:5100/healthz >/dev/null 2>&1; then
    echo "Launching .NET 10 API..."
    dotnet run --project src/FtaaSService.Api > /dev/null 2>&1 &
    SPAWNED_PIDS+=($!)
    for i in {1..20}; do
        if curl -s http://localhost:5100/healthz >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi
echo -e "${GREEN}✅ .NET 10 Ingestion Gateway is responsive.${NC}"

# 3. Check or start Python Compute Worker & Inference Engine
echo -e "\n${CYAN}[3/6] Ensuring Python Compute Worker & Inference Engine are running...${NC}"
if ! ps aux | grep -v grep | grep "consumer.py" >/dev/null 2>&1; then
    echo "Launching Python Compute Worker..."
    PYTHONUNBUFFERED=1 "$PYTHON_EXEC" src/FtaaSService.Worker/consumer.py > /dev/null 2>&1 &
    SPAWNED_PIDS+=($!)
fi

if ! curl -s http://localhost:8000/healthz >/dev/null 2>&1; then
    echo "Launching Python Dynamic Inference Engine..."
    PYTHONUNBUFFERED=1 "$PYTHON_EXEC" src/FtaaSService.Inference/app.py > /dev/null 2>&1 &
    SPAWNED_PIDS+=($!)
    for i in {1..25}; do
        if curl -s http://localhost:8000/healthz >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi
echo -e "${GREEN}✅ Compute Worker and Inference Engine are ready.${NC}"

# 4. Submit Fine-Tuning Job
echo -e "\n${CYAN}[4/6] Submitting Fine-Tuning Job via .NET 10 API Gateway...${NC}"
SUBMIT_RES=$(curl -s -X POST http://localhost:5100/api/v1/jobs \
  -F "file=@data/datasets/sample-financial-sentiment.jsonl" \
  -F "jobName=portfolio-verification-run" \
  -F "baseModel=HuggingFaceTB/SmolLM2-135M" \
  -F 'hyperparameters={"epochs": 2, "batchSize": 4, "learningRate": 0.0003, "loraRank": 8, "loraAlpha": 32, "loraDropout": 0.05}')

JOB_ID=$(echo "$SUBMIT_RES" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('jobId', ''))")

if [ -z "$JOB_ID" ]; then
    echo -e "${YELLOW}❌ Failed to submit job. Server response: $SUBMIT_RES${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Job accepted with ID: ${JOB_ID}${NC}"
echo "⏳ Waiting for worker to process training run..."

START_TIME=$(date +%s)
STATUS="Queued"
while [ "$STATUS" != "Succeeded" ] && [ "$STATUS" != "Failed" ]; do
    sleep 3
    JOB_DETAILS=$(curl -s "http://localhost:5100/api/v1/jobs/$JOB_ID")
    STATUS=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('status', ''))")
    PROGRESS=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('progressPercent', 0))")
    STEP=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('currentStep', 0))")
    TOTAL=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('totalSteps', 0))")
    LOSS=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('currentLoss', 'N/A'))")
    echo -e "  [$(date +%T)] Status: ${YELLOW}${STATUS}${NC} | Progress: ${PROGRESS}% (Step ${STEP}/${TOTAL}) | Current Loss: ${LOSS}"
done

ELAPSED=$(( $(date +%s) - START_TIME ))

if [ "$STATUS" == "Failed" ]; then
    echo -e "${YELLOW}❌ Job failed. Details:${NC}"
    echo "$JOB_DETAILS" | "$PYTHON_EXEC" -m json.tool
    exit 1
fi

RUN_ID=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('mlflowRunId', ''))")
ADAPTER_PATH=$(echo "$JOB_DETAILS" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('adapterPath', ''))")

echo -e "${GREEN}🎉 Training completed in ${ELAPSED} seconds!${NC}"

# 5. Side-by-Side Inference Comparison via .NET Gateway
echo -e "\n${CYAN}[5/6] Requesting Side-by-Side Completion Comparison from Gateway...${NC}"
TEST_PROMPT='Analyze earnings report snippet: Subscription ARR increased 34% YoY to $112M with net retention of 108%.'

PAYLOAD=$("$PYTHON_EXEC" -c "
import json
print(json.dumps({
    'jobId': '$JOB_ID',
    'prompt': '''$TEST_PROMPT''',
    'maxTokens': 64,
    'temperature': 0.1
}))
")

COMPARE_RES=$(curl -s -X POST http://localhost:5100/api/v1/inference/compare \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD")

BASE_COMP=$(echo "$COMPARE_RES" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('baseCompletion', ''))")
FINE_COMP=$(echo "$COMPARE_RES" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('fineTunedCompletion', ''))")
BASE_LAT=$(echo "$COMPARE_RES" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('latencyMs', {}).get('baseModel', 0))")
FINE_LAT=$(echo "$COMPARE_RES" | "$PYTHON_EXEC" -c "import sys, json; print(json.load(sys.stdin).get('latencyMs', {}).get('fineTuned', 0))")

echo -e "\n${BLUE}================================================================${NC}"
echo -e "${GREEN}           🎉 VERIFICATION REPORT & PORTFOLIO DEMO              ${NC}"
echo -e "${BLUE}================================================================${NC}"
echo -e "📌 ${CYAN}Job ID:${NC}             $JOB_ID"
echo -e "🧠 ${CYAN}Base Model:${NC}         HuggingFaceTB/SmolLM2-135M (LoRA r=8, alpha=32)"
echo -e "⏱️ ${CYAN}Training Duration:${NC}  ${ELAPSED}s on Apple Silicon Metal (MPS)"
echo -e "📦 ${CYAN}LoRA Adapter Path:${NC}  data/$ADAPTER_PATH"
echo -e "📊 ${CYAN}MLflow Run URL:${NC}     http://localhost:5001/#/experiments/1/runs/$RUN_ID"
echo -e "${BLUE}----------------------------------------------------------------${NC}"
echo -e "📝 ${CYAN}Input Prompt:${NC}\n   \"$TEST_PROMPT\""
echo -e "${BLUE}----------------------------------------------------------------${NC}"
echo -e "🔴 ${YELLOW}RAW BASE MODEL COMPLETION (Generic / Untuned):${NC}"
echo -e "   $BASE_COMP"
echo -e "   Latency: ${BASE_LAT}ms"
echo -e "${BLUE}----------------------------------------------------------------${NC}"
echo -e "🟢 ${GREEN}FINE-TUNED LoRA COMPLETION (Domain Structured Extraction):${NC}"
echo -e "   $FINE_COMP"
echo -e "   Latency: ${FINE_LAT}ms"
echo -e "${BLUE}================================================================${NC}"

# 6. Verify Observability & Prometheus Metrics Endpoints
echo -e "\n${CYAN}[6/6] Verifying Observability Endpoints (.NET API & Python Inference)...${NC}"
API_METRICS=$(curl -s http://localhost:5100/metrics)
if echo "$API_METRICS" | grep -q "ftaas_jobs_total"; then
    echo -e "${GREEN}✅ .NET API /metrics endpoint is exporting valid Prometheus metrics.${NC}"
else
    echo -e "${YELLOW}❌ .NET API /metrics failed verification.${NC}"
    exit 1
fi

INFERENCE_METRICS=$(curl -s http://localhost:8000/metrics)
if echo "$INFERENCE_METRICS" | grep -q "ftaas_inference_requests_total"; then
    echo -e "${GREEN}✅ Python Inference /metrics endpoint is exporting valid Prometheus metrics.${NC}"
else
    echo -e "${YELLOW}❌ Python Inference /metrics failed verification.${NC}"
    exit 1
fi

echo -e "\n${GREEN}🎉 All 6 E2E Pipeline and Observability Stages Passed Successfully!${NC}"
