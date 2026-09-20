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
    private readonly IPrivacyTelemetry _inner;

    public string Mode => _inner.Mode;
    public bool IsBurnEnabled => _inner.IsBurnEnabled;
    public string? OtlpEndpoint => _inner.OtlpEndpoint;

    public FtaasTelemetry(IConfiguration configuration)
    {
        var allowlist = new SafeAllowlist(new[]
        {
            "job_id", "status", "duration_ms", "duration_sec", "base_model", "dataset_hash",
            "dataset_relative_path", "current_step", "total_steps", "progress_pct", "loss",
            "device", "adapter_path", "adapter_size_bytes", "exchange", "routing_key"
        });

        _inner = new PrivacyTelemetryService(
            configuration: configuration,
            allowlist: allowlist);
    }

    public void RecordSpan(string name, string jobId, string status, long durationMs = 0, IDictionary<string, object>? attributes = null)
    {
        _inner.RecordSpan(name, jobId, status, durationMs, attributes);
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
