# syntax=docker/dockerfile:1
# ── Stage 1: Build & Publish Control Plane ────────────────────────────
FROM mcr.microsoft.com/dotnet/sdk:10.0-alpine AS build
WORKDIR /workspace

# Copy privacy-telemetry project
COPY privacy-telemetry/packages/KnowThankYew.Privacy.Telemetry /workspace/privacy-telemetry/packages/KnowThankYew.Privacy.Telemetry

# Copy ml CPM definitions and project files
COPY ml/Directory.Build.props ml/Directory.Packages.props /workspace/ml/
COPY ml/src/FtaaSService.Api/FtaaSService.Api.csproj /workspace/ml/src/FtaaSService.Api/

WORKDIR /workspace/ml/src/FtaaSService.Api
RUN dotnet restore

# Copy all source files and publish
COPY ml/src/FtaaSService.Api /workspace/ml/src/FtaaSService.Api
RUN dotnet publish -c Release -o /app/publish

# ── Stage 2: Hardened Runtime ─────────────────────────────────────────
FROM mcr.microsoft.com/dotnet/aspnet:10.0-alpine AS runtime
WORKDIR /app

RUN apk add --no-cache wget

# Create data directory for SQLite jobs database
RUN mkdir -p /app/data && chown -R $APP_UID:$APP_UID /app/data

COPY --from=build /app/publish .

ENV ASPNETCORE_URLS=http://+:5000
ENV ASPNETCORE_ENVIRONMENT=Production
ENV ConnectionStrings__DefaultConnection="Data Source=/app/data/ftaas.db"

USER $APP_UID

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost:5000/health || exit 1

ENTRYPOINT ["dotnet", "FtaaSService.Api.dll"]
