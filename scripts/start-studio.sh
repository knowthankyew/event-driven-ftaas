#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "🚀 Starting FTaaS Enterprise Studio (Non-Tech Portal)"
echo "=========================================================="

# 1. Check Pre-requisites
if ! command -v dotnet >/dev/null 2>&1; then
    echo "❌ Missing Prerequisite: .NET 10 SDK is not installed or not in PATH."
    echo ""
    echo "👉 To run FTaaS Studio, please install the free .NET 10 SDK from:"
    echo "   https://dotnet.microsoft.com/download"
    echo ""
    echo "Once installed, re-run: ./scripts/start-studio.sh"
    exit 1
fi

DOTNET_VER=$(dotnet --version 2>/dev/null || echo "Unknown")
echo "✓ .NET SDK detected: v$DOTNET_VER"

# Check port 5100 availability
if lsof -i :5100 >/dev/null 2>&1; then
    echo "⚠️ Warning: Port 5100 is already in use by another process."
    echo "   If another instance of FTaaS is running, open http://localhost:5100"
    echo "   Or terminate that process with: kill \$(lsof -t -i :5100)"
fi

echo "• Web App: http://localhost:5100"
echo "• Mode: Interactive Preview by default (Real GPU Compute when Worker is up)"
echo "=========================================================="

# Build API
echo "Building Control Plane..."
dotnet build src/FtaaSService.Api -c Release --nologo -v q

echo ""
echo "✨ Opening FTaaS Studio in your browser: http://localhost:5100"
echo "Press Ctrl+C to stop the studio anytime."
echo ""

# Try opening browser on macOS or Linux
if command -v open >/dev/null 2>&1; then
    (sleep 2 && open "http://localhost:5100") &
elif command -v xdg-open >/dev/null 2>&1; then
    (sleep 2 && xdg-open "http://localhost:5100") &
fi

cd src/FtaaSService.Api
dotnet run --configuration Release --no-build
