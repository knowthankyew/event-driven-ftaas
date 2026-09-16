#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "🚀 Starting FTaaS Enterprise Studio (Non-Tech Portal)"
echo "=========================================================="
echo "• Control Plane: .NET 10 Minimal API"
echo "• Web App: Hosted directly on http://localhost:5100"
echo "• Dataset Engine: sample-support-compliance & sample-financial"
echo "=========================================================="

# Build API
echo "Building .NET 10 API..."
dotnet build src/FtaaSService.Api -c Release --nologo -v q

# Launch in background or foreground
echo ""
echo "✨ Opening FTaaS Studio in your browser: http://localhost:5100"
echo "Press Ctrl+C to stop the studio anytime."
echo ""

# Try opening browser on macOS
if command -v open >/dev/null 2>&1; then
    (sleep 2 && open "http://localhost:5100") &
fi

cd src/FtaaSService.Api
dotnet run --configuration Release --no-build
