namespace FtaaSService.Api.Services;

public sealed record DatasetResult(
    bool IsValid,
    string? RelativePath,
    string? Sha256Hash,
    int RecordCount,
    string? ErrorMessage
);

public interface IDatasetService
{
    Task<DatasetResult> NormalizeAndStoreAsync(
        string jobId,
        IFormFile? uploadedFile,
        string? existingPath,
        CancellationToken cancellationToken = default
    );
}
