using FtaaSService.Api.Services;
using Microsoft.Extensions.Configuration;
using Xunit;

namespace FtaaSService.Api.Tests;

public sealed class FtaasTelemetryTests
{
    private readonly IConfiguration _config;

    public FtaasTelemetryTests()
    {
        var inMemorySettings = new Dictionary<string, string?>
        {
            ["Telemetry:Mode"] = "memory_only",
            ["Privacy:BurnEnabled"] = "true"
        };

        _config = new ConfigurationBuilder()
            .AddInMemoryCollection(inMemorySettings)
            .Build();
    }

    [Fact]
    public void RecordSpan_SanitizesProhibitedDatasetContent()
    {
        var telemetry = new FtaasTelemetry(_config);

        var attributes = new Dictionary<string, object>
        {
            ["dataset_hash"] = "abc123def456",
            ["dataset_text"] = "RAW USER PROMPT AND SECRET TRAINING EXAMPLE",
            ["prompt"] = "Confidential prompt content",
            ["device"] = "mps",
            ["adapter_size_bytes"] = 1048576
        };

        telemetry.RecordSpan("job.training.started", "job-test-1", "Training", 120, attributes);

        var spans = telemetry.GetBufferedSpans();
        Assert.Single(spans);
        var span = spans[0];

        Assert.Equal("job.training.started", span.Name);
        Assert.Equal("job-test-1", span.Attributes["job_id"]);
        Assert.Equal("Training", span.Attributes["status"]);
        Assert.Equal("mps", span.Attributes["device"]);
        Assert.Equal(1048576, span.Attributes["adapter_size_bytes"]);

        // Prohibited/unknown keys must be redacted by allowlist
        Assert.Equal("[REDACTED_NOT_IN_ALLOWLIST]", span.Attributes["dataset_text"]);
        Assert.Equal("[REDACTED_NOT_IN_ALLOWLIST]", span.Attributes["prompt"]);
    }

    [Fact]
    public void Burn_ClearsBufferedSpansImmediately()
    {
        var telemetry = new FtaasTelemetry(_config);

        telemetry.RecordSpan("job.accepted", "job-1", "Queued");
        telemetry.RecordSpan("job.published", "job-1", "Queued");

        Assert.Equal(2, telemetry.GetBufferedSpans().Count);

        telemetry.Burn();

        Assert.Empty(telemetry.GetBufferedSpans());
    }

    [Fact]
    public void GetPrivacyAuditReport_ReportsHonestObservability()
    {
        var telemetry = new FtaasTelemetry(_config);

        var report = telemetry.GetPrivacyAuditReport();
        Assert.NotNull(report);

        // Reflection or dynamic property check
        var modeProp = report.GetType().GetProperty("telemetryMode");
        var isHonestProp = report.GetType().GetProperty("isLocalOnlyHonest");

        Assert.Equal("memory_only", modeProp?.GetValue(report));
        Assert.Equal(true, isHonestProp?.GetValue(report));
    }
}
