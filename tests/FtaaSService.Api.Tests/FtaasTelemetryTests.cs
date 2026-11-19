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

        telemetry.RecordSpan("job.training.started", "job-101", "Training", 120, attributes);

        var spans = telemetry.GetBufferedSpans();
        Assert.Single(spans);
        var span = spans[0];

        Assert.Equal("job.training.started", span.Name);
        Assert.Equal("job-101", span.Attributes["job_id"]);
        Assert.Equal("Training", span.Attributes["status"]);
        Assert.Equal("mps", span.Attributes["device"]);
        Assert.Equal(1048576, span.Attributes["adapter_size_bytes"]);

        // Prohibited/unknown keys must be stripped and fail-closed redacted under [REDACTED_KEY]
        Assert.False(span.Attributes.ContainsKey("dataset_text"));
        Assert.False(span.Attributes.ContainsKey("prompt"));
        Assert.Equal("[REDACTED_NOT_IN_ALLOWLIST]", span.Attributes["[REDACTED_KEY]"]);
    }

    [Fact]
    public void RecordSpan_SanitizesPathValuedAttributesToBasenameOrKeyedHmac()
    {
        var keyedConfig = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Telemetry:Mode"] = "memory_only",
                ["Privacy:BurnEnabled"] = "true",
                ["Telemetry:HmacKey"] = "unit-test-hmac-secret"
            })
            .Build();
        var telemetry = new FtaasTelemetry(keyedConfig);

        const string rawDatasetPath = "/Users/alice/private/datasets/job-999.jsonl";
        const string rawAdapterPath = "/Users/alice/private/artifacts/job-999/model_adapters";
        var attributes = new Dictionary<string, object>
        {
            ["dataset_relative_path"] = rawDatasetPath,
            ["adapter_path"] = rawAdapterPath
        };

        telemetry.RecordSpan("job.completed", "job-999", "Succeeded", 50, attributes);

        var span = Assert.Single(telemetry.GetBufferedSpans());
        var dsPath = Assert.IsType<string>(span.Attributes["dataset_relative_path"]);
        var adapterPath = Assert.IsType<string>(span.Attributes["adapter_path"]);

        Assert.Equal("job-999.jsonl", dsPath);
        Assert.DoesNotContain("/", dsPath);
        Assert.DoesNotContain("alice", dsPath);

        Assert.Matches("^hmac:[a-f0-9]{16}$", adapterPath);
        Assert.DoesNotContain("/", adapterPath);
        Assert.DoesNotContain("alice", adapterPath);

        // Must match independent single-pass keyed HMAC (not unsalted SHA-256 and not double-hashed)
        var independentMac = System.Security.Cryptography.HMACSHA256.HashData(
            System.Text.Encoding.UTF8.GetBytes("unit-test-hmac-secret"),
            System.Text.Encoding.UTF8.GetBytes(rawAdapterPath));
        var expectedSinglePass = $"hmac:{Convert.ToHexString(independentMac).ToLowerInvariant()[..16]}";
        Assert.Equal(expectedSinglePass, adapterPath);

        var unsaltedSha256 = $"sha256:{Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(rawAdapterPath))).ToLowerInvariant()[..16]}";
        Assert.NotEqual(unsaltedSha256.Replace("sha256:", "hmac:"), adapterPath);
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
