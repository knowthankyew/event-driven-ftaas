using System.Collections.Concurrent;
using System.Diagnostics;

namespace FtaaSService.Api.Services;

public sealed class FtaasSpanRecord
{
    public required string Id { get; init; }
    public required string Name { get; init; }
    public required DateTimeOffset Timestamp { get; init; }
    public long DurationMs { get; set; }
    public required string Status { get; set; }
    public Dictionary<string, object> Attributes { get; init; } = new();
}

public interface IFtaasTelemetry
{
    string Mode { get; }
    bool IsBurnEnabled { get; }
    string? OtlpEndpoint { get; }
    void RecordSpan(string name, string jobId, string status, long durationMs = 0, IDictionary<string, object>? attributes = null);
    IReadOnlyList<FtaasSpanRecord> GetBufferedSpans();
    void Burn();
    object GetPrivacyAuditReport();
}

public sealed class FtaasTelemetry : IFtaasTelemetry
{
    public static readonly ActivitySource ActivitySource = new("KnowThankYew.FtaaSService", "1.0.0");

    private readonly ConcurrentQueue<FtaasSpanRecord> _spanBuffer = new();
    private const int MaxBufferSize = 500;
    private readonly string _mode;
    private readonly string? _otlpEndpoint;
    private readonly bool _burnEnabled;

    private static readonly HashSet<string> ProhibitedAttributes = new(StringComparer.OrdinalIgnoreCase)
    {
        "text", "body", "dataset_text", "prompt", "completion", "data", "payload", "raw_content", "dataset_content"
    };

    public string Mode => _mode;
    public bool IsBurnEnabled => _burnEnabled;
    public string? OtlpEndpoint => _otlpEndpoint;

    public FtaasTelemetry(IConfiguration configuration)
    {
        _otlpEndpoint = Environment.GetEnvironmentVariable("OTEL_EXPORTER_OTLP_ENDPOINT")
                        ?? configuration["Telemetry:OtlpEndpoint"];

        _mode = !string.IsNullOrEmpty(_otlpEndpoint)
            ? "otlp"
            : (configuration["Telemetry:Mode"] ?? "memory_only");

        _burnEnabled = bool.TryParse(configuration["Privacy:BurnEnabled"], out var b) ? b : true;
    }

    public void RecordSpan(string name, string jobId, string status, long durationMs = 0, IDictionary<string, object>? attributes = null)
    {
        if (_mode == "disabled") return;

        using var activity = ActivitySource.StartActivity(name);
        activity?.SetTag("job_id", jobId);
        activity?.SetTag("status", status);
        if (durationMs > 0) activity?.SetTag("duration_ms", durationMs);

        var sanitizedAttrs = new Dictionary<string, object>
        {
            ["job_id"] = jobId,
            ["status"] = status
        };

        if (attributes != null)
        {
            foreach (var kv in attributes)
            {
                if (ProhibitedAttributes.Contains(kv.Key))
                {
                    sanitizedAttrs[kv.Key] = "[REDACTED_BY_PRIVACY_POLICY]";
                    continue;
                }

                // Protect against raw large text in values
                if (kv.Value is string s && s.Length > 256)
                {
                    sanitizedAttrs[kv.Key] = $"[TRUNCATED_HASH_{s[..8]}...]";
                }
                else
                {
                    sanitizedAttrs[kv.Key] = kv.Value;
                }

                activity?.SetTag(kv.Key, sanitizedAttrs[kv.Key]);
            }
        }

        var record = new FtaasSpanRecord
        {
            Id = activity?.Id ?? Guid.NewGuid().ToString("N"),
            Name = name,
            Timestamp = DateTimeOffset.UtcNow,
            DurationMs = durationMs,
            Status = status,
            Attributes = sanitizedAttrs
        };

        _spanBuffer.Enqueue(record);

        while (_spanBuffer.Count > MaxBufferSize && _spanBuffer.TryDequeue(out _))
        {
        }
    }

    public IReadOnlyList<FtaasSpanRecord> GetBufferedSpans() => _spanBuffer.ToArray();

    public void Burn()
    {
        if (_burnEnabled)
        {
            _spanBuffer.Clear();
        }
    }

    public object GetPrivacyAuditReport()
    {
        return new
        {
            telemetryMode = _mode,
            networkEgress = _mode == "otlp" ? "allow_otlp" : "deny",
            burnEnabled = _burnEnabled,
            otlpEndpoint = _otlpEndpoint,
            allowRawPayloads = false,
            activeSpanCount = _spanBuffer.Count,
            isLocalOnlyHonest = _mode != "otlp" && string.IsNullOrEmpty(_otlpEndpoint)
        };
    }
}
