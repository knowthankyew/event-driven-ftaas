#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo "  FTaaS Enterprise Studio - Playwright QA Baseline Suite  "
echo "=========================================================="

if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required to execute the Playwright QA baseline."
    exit 1
fi

SPAWNED_PID=""
cleanup() {
    if [ -n "$SPAWNED_PID" ]; then
        echo "Shutting down temporary Studio server (PID $SPAWNED_PID)..."
        kill "$SPAWNED_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Ensure Studio API is running
if ! curl -s http://localhost:5100/healthz >/dev/null 2>&1; then
    echo "Starting Studio Control Plane on http://localhost:5100..."
    dotnet run --project src/FtaaSService.Api &
    SPAWNED_PID=$!
    
    echo "Waiting for Studio to become responsive..."
    READY=0
    for i in {1..30}; do
        if curl -s http://localhost:5100/healthz >/dev/null 2>&1; then
            READY=1
            echo "Studio is ready!"
            break
        fi
        sleep 1
    done

    if [ "$READY" -ne 1 ]; then
        echo "Error: Studio API failed to start on port 5100." >&2
        exit 1
    fi
else
    echo "Studio API is already active on http://localhost:5100."
fi

# Run Playwright QA Baseline Walkthrough
node scripts/qa-walkthrough.js

echo ""
echo "QA Baseline execution completed successfully."
echo "Visual checkpoints saved to: tests/qa-screenshots/"
