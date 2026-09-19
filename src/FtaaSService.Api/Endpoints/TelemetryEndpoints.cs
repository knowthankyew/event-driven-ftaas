using FtaaSService.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace FtaaSService.Api.Endpoints;

public static class TelemetryEndpoints
{
    public static RouteGroupBuilder MapTelemetryEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/privacy-audit", (
            [FromServices] IFtaasTelemetry telemetry) =>
        {
            return Results.Ok(telemetry.GetPrivacyAuditReport());
        });

        group.MapGet("/spans", (
            [FromServices] IFtaasTelemetry telemetry) =>
        {
            return Results.Ok(new
            {
                mode = telemetry.Mode,
                count = telemetry.GetBufferedSpans().Count,
                spans = telemetry.GetBufferedSpans()
            });
        });

        group.MapPost("/burn", (
            [FromServices] IFtaasTelemetry telemetry) =>
        {
            telemetry.Burn();
            return Results.Ok(new
            {
                message = "Telemetry buffer successfully wiped from memory.",
                remainingSpans = telemetry.GetBufferedSpans().Count
            });
        });

        return group;
    }
}
