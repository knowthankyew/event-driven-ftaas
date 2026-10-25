using FtaaSService.Api.Services.Howso;
using Microsoft.AspNetCore.Http;

namespace FtaaSService.Api.Services;

public sealed record DatasetResult(
    bool IsValid,
    string? RelativePath,
    string? Sha256Hash,
    int RecordCount,
    string? ErrorMessage,
    HowsoAuditSummary? HowsoAudit = null,
    string? HowsoPreflightStatus = null,
    double? PreflightDurationMs = null
);

public interface IDatasetService
{
    Task<DatasetResult> NormalizeAndStoreAsync(
        string jobId,
        IFormFile? uploadedFile,
        string? existingPath,
        bool enableHowsoPreflight = false,
        CancellationToken cancellationToken = default
    );

    Task<(bool IsValid, string? ErrorMessage, List<HowsoDatasetRecord>? Records)> ValidateAndExtractRecordsAsync(
        IFormFile? uploadedFile,
        string? existingPath,
        CancellationToken cancellationToken = default
    );
}
