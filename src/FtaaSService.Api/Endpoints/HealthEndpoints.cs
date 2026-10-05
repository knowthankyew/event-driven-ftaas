using FtaaSService.Api.Messaging;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Options;
using RabbitMQ.Client;

namespace FtaaSService.Api.Endpoints;

public static class HealthEndpoints
{
    public static void MapHealthEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/healthz", async (
            [FromServices] IJobRepository jobRepository,
            [FromServices] IEventPublisher eventPublisher,
            [FromServices] IOptions<RabbitMqConfig> rabbitOptions,
            CancellationToken cancellationToken) =>
        {
            var checks = new Dictionary<string, string>();
            bool healthy = true;

            // 1. Database Check
            try
            {
                await jobRepository.ListAsync(1, cancellationToken);
                checks["database"] = "Healthy";
            }
            catch (Exception ex)
            {
                checks["database"] = $"Unhealthy: {ex.Message}";
                healthy = false;
            }

            // 2. RabbitMQ Connectivity Check (Reuses publisher connection, avoids TCP churn)
            if (eventPublisher.IsConnected)
            {
                checks["rabbitmq"] = "Healthy";
            }
            else
            {
                try
                {
                    var cfg = rabbitOptions.Value;
                    var factory = new ConnectionFactory
                    {
                        HostName = cfg.HostName,
                        Port = cfg.Port,
                        UserName = cfg.UserName,
                        Password = cfg.Password
                    };
                    using var conn = await factory.CreateConnectionAsync(cancellationToken);
                    checks["rabbitmq"] = "Healthy";
                }
                catch (Exception ex)
                {
                    checks["rabbitmq"] = $"Unhealthy: {ex.Message}";
                    healthy = false;
                }
            }

            // 3. Dead-Letter Queue (DLQ) Monitoring
            if (checks.TryGetValue("rabbitmq", out var rStatus) && rStatus == "Healthy")
            {
                try
                {
                    var dlqCount = await eventPublisher.GetDlqMessageCountAsync(cancellationToken);
                    if (dlqCount > 0)
                    {
                        checks["dlq"] = $"Warning: {dlqCount} message(s) in dead-letter queue";
                    }
                    else
                    {
                        checks["dlq"] = "Healthy (0 messages)";
                    }
                }
                catch (Exception ex)
                {
                    checks["dlq"] = $"Unavailable: {ex.Message}";
                }
            }

            var result = new
            {
                status = healthy ? "Healthy" : "Degraded",
                timestamp = DateTimeOffset.UtcNow,
                checks
            };

            return healthy ? Results.Ok(result) : Results.Json(result, statusCode: StatusCodes.Status503ServiceUnavailable);
        });
    }
}
