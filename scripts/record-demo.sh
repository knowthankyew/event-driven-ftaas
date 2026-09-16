#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================================="
echo "  FTaaS Enterprise Studio - Automated Demo Video Recorder"
echo "=========================================================="

if ! command -v node &> /dev/null; then
    echo "Error: Node.js is required to record video demos."
    exit 1
fi

# Ensure playwright-core is installed locally in scratch/ or global
if [ ! -d "$REPO_ROOT/node_modules/playwright-core" ]; then
    echo "Installing playwright-core..."
    npm install --no-save playwright-core
fi

# Start .NET API server in background
echo "Starting Studio server on http://localhost:5100..."
dotnet run --project src/FtaaSService.Api &
API_PID=$!

# Ensure API is killed on exit
cleanup() {
    echo "Shutting down Studio server (PID $API_PID)..."
    kill $API_PID 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server readiness
echo "Waiting for Studio to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5100 > /dev/null; then
        echo "Studio is ready!"
        break
    fi
    sleep 1
done

# Run Playwright recording script
node scripts/record-demo.js

echo "Demo recording saved to: $REPO_ROOT/demo.mp4"
