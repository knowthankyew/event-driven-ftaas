using System.Diagnostics;
using System.Globalization;
using System.Text;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;

namespace FtaaSService.Api.Endpoints;

public static class MetricsEndpoints
{
    private static readonly DateTimeOffset ProcessStartTime = DateTimeOffset.UtcNow;

    public static void MapMetricsEndpoints(this IEndpointRouteBuilder app)
    {
        app.MapGet("/metrics", GetMetricsAsync);
    }

    public static async Task<IResult> GetMetricsAsync(
        [FromServices] IJobRepository jobRepository,
        [FromServices] IEventPublisher eventPublisher,
        [FromServices] IFtaasTelemetry telemetry,
        CancellationToken cancellationToken)
    {
        var sb = new StringBuilder();

        // 1. Process Uptime
        var uptimeSeconds = (DateTimeOffset.UtcNow - ProcessStartTime).TotalSeconds;
        sb.AppendLine("# HELP ftaas_uptime_seconds Total time the service has been running in seconds.");
        sb.AppendLine("# TYPE ftaas_uptime_seconds gauge");
        sb.AppendLine(string.Create(CultureInfo.InvariantCulture, $"ftaas_uptime_seconds {uptimeSeconds:F2}"));
        sb.AppendLine();

        // 2. Data Sources Operational Status
        bool dbSuccess = false;
        IReadOnlyDictionary<string, int>? jobCounts = null;
        try
        {
            jobCounts = await jobRepository.GetJobCountsByStatusAsync(cancellationToken);
            dbSuccess = true;
        }
        catch
        {
            dbSuccess = false;
        }

        bool dlqSuccess = false;
        uint dlqCount = 0;
        if (eventPublisher.IsConnected)
        {
            try
            {
                dlqCount = await eventPublisher.GetDlqMessageCountAsync(cancellationToken);
                dlqSuccess = true;
            }
            catch
            {
                dlqSuccess = false;
            }
        }

        sb.AppendLine("# HELP ftaas_metrics_source_up Indicates whether a metrics data source is operational (1) or failed (0).");
        sb.AppendLine("# TYPE ftaas_metrics_source_up gauge");
        sb.AppendLine($"ftaas_metrics_source_up{{source=\"database\"}} {(dbSuccess ? 1 : 0)}");
        sb.AppendLine($"ftaas_metrics_source_up{{source=\"dlq\"}} {(dlqSuccess ? 1 : 0)}");
        sb.AppendLine();

        // 3. Job Counts by Status (omitted on database failure)
        if (dbSuccess && jobCounts != null)
        {
            var statuses = new[] { "Pending", "Queued", "Training", "Succeeded", "Failed" };
            sb.AppendLine("# HELP ftaas_jobs_total Current number of fine-tuning jobs recorded by status.");
            sb.AppendLine("# TYPE ftaas_jobs_total gauge");
            foreach (var status in statuses)
            {
                jobCounts.TryGetValue(status, out var count);
                sb.AppendLine($"ftaas_jobs_total{{status=\"{status}\"}} {count}");
            }
            sb.AppendLine();

            // Active Jobs (Pending + Queued + Training)
            jobCounts.TryGetValue("Pending", out var pendingCount);
            jobCounts.TryGetValue("Queued", out var queuedCount);
            jobCounts.TryGetValue("Training", out var trainingCount);
            var activeJobs = pendingCount + queuedCount + trainingCount;
            sb.AppendLine("# HELP ftaas_active_jobs Current number of active fine-tuning jobs (pending, queued, or training).");
            sb.AppendLine("# TYPE ftaas_active_jobs gauge");
            sb.AppendLine($"ftaas_active_jobs {activeJobs}");
            sb.AppendLine();
        }

        // 4. DLQ Depth (omitted on DLQ query failure)
        if (dlqSuccess)
        {
            sb.AppendLine("# HELP ftaas_dlq_message_count Number of dead-lettered messages in the broker DLQ.");
            sb.AppendLine("# TYPE ftaas_dlq_message_count gauge");
            sb.AppendLine($"ftaas_dlq_message_count {dlqCount}");
            sb.AppendLine();
        }

        // 5. Telemetry Buffered Spans
        var spanCount = telemetry.GetBufferedSpans().Count;
        sb.AppendLine("# HELP ftaas_telemetry_buffered_spans Current in-memory buffered privacy spans before purge.");
        sb.AppendLine("# TYPE ftaas_telemetry_buffered_spans gauge");
        sb.AppendLine($"ftaas_telemetry_buffered_spans {spanCount}");
        sb.AppendLine();

        return Results.Text(sb.ToString(), "text/plain; version=0.0.4; charset=utf-8", Encoding.UTF8);
    }
}
