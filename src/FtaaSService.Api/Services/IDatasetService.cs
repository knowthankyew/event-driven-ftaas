using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;
using Microsoft.AspNetCore.Http;

namespace FtaaSService.Api.Services;

public sealed record DatasetResult(
    bool IsValid,
    string? RelativePath,
    string? Sha256Hash,
    int RecordCount,
    string? ErrorMessage,
    SurprisalAuditSummary? SurprisalAudit = null,
    string? SurprisalPreflightStatus = null,
    double? PreflightDurationMs = null
)
{
    // Backward-compatibility getters
    [Obsolete("Use SurprisalAudit instead.")]
    public HowsoAuditSummary? HowsoAudit => SurprisalAudit is null ? null : new HowsoAuditSummary(
        SurprisalAudit.OriginalCount,
        SurprisalAudit.CleanedCount,
        SurprisalAudit.AblatedCount,
        SurprisalAudit.AnomalousCount,
        SurprisalAudit.MeasuredDeviations,
        SurprisalAudit.AnomalousRecords,
        SurprisalAudit.AblatedRecords
    );

    [Obsolete("Use SurprisalPreflightStatus instead.")]
    public string? HowsoPreflightStatus => SurprisalPreflightStatus;
}

public interface IDatasetService
{
    Task<DatasetResult> NormalizeAndStoreAsync(
        string jobId,
        IFormFile? uploadedFile,
        string? existingPath,
        bool enableSurprisalPreflight = false,
        CancellationToken cancellationToken = default
    );

    Task<(bool IsValid, string? ErrorMessage, List<SurprisalDatasetRecord>? Records)> ValidateAndExtractRecordsAsync(
        IFormFile? uploadedFile,
        string? existingPath,
        CancellationToken cancellationToken = default
    );
}
