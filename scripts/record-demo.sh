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

# Check if a recognized system browser is present
HAS_SYSTEM_BROWSER=0
if [ -f "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser" ] || \
   [ -f "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" ] || \
   command -v google-chrome &> /dev/null || \
   command -v chromium &> /dev/null; then
    HAS_SYSTEM_BROWSER=1
fi

# Ensure Playwright is available locally or via portfolio root
if [ ! -d "$REPO_ROOT/node_modules/playwright-core" ] && [ ! -d "$REPO_ROOT/../node_modules/playwright" ]; then
    echo "Installing playwright-core..."
    npm install --no-save playwright-core
fi

# If no recognized system browser exists, ensure Playwright Chromium is installed
if [ "$HAS_SYSTEM_BROWSER" -eq 0 ]; then
    echo "No system browser detected. Ensuring Playwright Chromium is installed..."
    npx --yes playwright install chromium || true
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
READY=0
for i in {1..120}; do
    if curl -s http://localhost:5100 > /dev/null; then
        echo "Studio is ready!"
        READY=1
        break
    fi
    sleep 1
done
if [ "$READY" -ne 1 ]; then
    echo "Error: Studio did not become ready on http://localhost:5100" >&2
    exit 1
fi

# Run Playwright recording script
node scripts/record-demo.js

echo "Demo recording saved to:"
echo "  - MP4: $REPO_ROOT/demo.mp4"
if [ -f "$REPO_ROOT/demo.gif" ]; then
    echo "  - GIF: $REPO_ROOT/demo.gif"
fi
