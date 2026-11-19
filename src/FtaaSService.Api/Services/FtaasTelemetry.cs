using System.Diagnostics;
using KnowThankYew.Privacy.Telemetry.Abstractions;
using KnowThankYew.Privacy.Telemetry.Core;
using Microsoft.Extensions.Configuration;

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
    private static readonly byte[] EphemeralFallbackHmacKey =
        System.Security.Cryptography.RandomNumberGenerator.GetBytes(32);

    private readonly IPrivacyTelemetry _inner;
    private readonly byte[] _hmacKey;

    public string Mode => _inner.Mode;
    public bool IsBurnEnabled => _inner.IsBurnEnabled;
    public string? OtlpEndpoint => _inner.OtlpEndpoint;

    public FtaasTelemetry(IConfiguration configuration)
    {
        var configuredHmacKey = configuration["FTAAS_TELEMETRY_HMAC_KEY"]
            ?? configuration["Telemetry:HmacKey"]
            ?? Environment.GetEnvironmentVariable("FTAAS_TELEMETRY_HMAC_KEY");
        _hmacKey = string.IsNullOrWhiteSpace(configuredHmacKey)
            ? EphemeralFallbackHmacKey
            : System.Text.Encoding.UTF8.GetBytes(configuredHmacKey);

        var allowlist = new SafeAllowlist(new[]
        {
            "job_id", "status", "duration_ms", "duration_sec", "base_model", "dataset_hash",
            "dataset_relative_path", "current_step", "total_steps", "progress_pct", "loss",
            "device", "adapter_path", "adapter_size_bytes", "exchange", "routing_key",
            "surprisal.preflight.enabled", "surprisal.preflight.status", "surprisal.preflight.duration_ms",
            "surprisal.cases.original_count", "surprisal.cases.ablated_count", "surprisal.cases.anomalous_count",
            "howso.preflight.enabled", "howso.preflight.status", "howso.preflight.duration_ms",
            "howso.cases.original_count", "howso.cases.ablated_count", "howso.cases.anomalous_count"
        });

        _inner = new PrivacyTelemetryService(
            configuration: configuration,
            allowlist: allowlist);
    }

    public static string SanitizePathAttribute(string key, string? value, byte[]? hmacKey = null)
    {
        if (string.IsNullOrWhiteSpace(value)) return string.Empty;
        if (string.Equals(key, "adapter_path", StringComparison.OrdinalIgnoreCase))
        {
            var keyBytes = (hmacKey is { Length: > 0 }) ? hmacKey : EphemeralFallbackHmacKey;
            var mac = System.Security.Cryptography.HMACSHA256.HashData(
                keyBytes,
                System.Text.Encoding.UTF8.GetBytes(value));
            return $"hmac:{Convert.ToHexString(mac).ToLowerInvariant()[..16]}";
        }
        var segments = value.Replace('\\', '/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        return segments.Length > 0 ? segments[^1] : string.Empty;
    }

    public void RecordSpan(string name, string jobId, string status, long durationMs = 0, IDictionary<string, object>? attributes = null)
    {
        IDictionary<string, object>? sanitized = attributes;
        if (attributes is not null)
        {
            Dictionary<string, object>? copied = null;
            foreach (var kv in attributes)
            {
                if (string.Equals(kv.Key, "adapter_path", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(kv.Key, "dataset_relative_path", StringComparison.OrdinalIgnoreCase))
                {
                    copied ??= new Dictionary<string, object>(attributes);
                    copied[kv.Key] = SanitizePathAttribute(kv.Key, kv.Value?.ToString(), _hmacKey);
                }
            }
            if (copied is not null)
            {
                sanitized = copied;
            }
        }
        _inner.RecordSpan(name, jobId, status, durationMs, sanitized);
    }

    public IReadOnlyList<FtaasSpanRecord> GetBufferedSpans()
    {
        return _inner.GetBufferedSpans().Select(s => new FtaasSpanRecord
        {
            Id = s.Id,
            Name = s.Name,
            Timestamp = s.Timestamp,
            DurationMs = s.DurationMs,
            Status = s.Status,
            Attributes = s.Attributes
        }).ToList();
    }

    public void Burn()
    {
        _inner.Burn();
    }

    public object GetPrivacyAuditReport()
    {
        var report = _inner.GetPrivacyAuditReport();
        return new
        {
            telemetryMode = report.TelemetryMode,
            networkEgress = report.NetworkEgress,
            burnEnabled = report.BurnEnabled,
            otlpEndpoint = report.OtlpEndpoint,
            allowRawPayloads = report.AllowRawPayloads,
            activeSpanCount = report.ActiveSpanCount,
            isLocalOnlyHonest = report.IsLocalOnlyHonest
        };
    }
}
