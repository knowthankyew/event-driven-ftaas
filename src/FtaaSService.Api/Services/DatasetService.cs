using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace FtaaSService.Api.Services;

public sealed class DatasetService : IDatasetService
{
    private readonly string _dataRoot;
    private readonly ILogger<DatasetService> _logger;

    public DatasetService(IConfiguration configuration, ILogger<DatasetService> logger)
    {
        _logger = logger;
        _dataRoot = configuration.GetValue<string>("DATA_ROOT") 
                    ?? Path.Combine(Directory.GetCurrentDirectory(), "data");

        Directory.CreateDirectory(Path.Combine(_dataRoot, "datasets"));
    }

    public async Task<DatasetResult> NormalizeAndStoreAsync(
        string jobId,
        IFormFile? uploadedFile,
        string? existingPath,
        CancellationToken cancellationToken = default)
    {
        try
        {
            Stream inputStream;
            if (uploadedFile is not null && uploadedFile.Length > 0)
            {
                inputStream = uploadedFile.OpenReadStream();
            }
            else if (!string.IsNullOrWhiteSpace(existingPath))
            {
                // Resolve path relative to dataRoot if not rooted
                var resolved = Path.IsPathRooted(existingPath)
                    ? existingPath
                    : Path.Combine(_dataRoot, existingPath);

                if (!File.Exists(resolved))
                {
                    return new DatasetResult(false, null, null, 0, $"Dataset file not found at '{existingPath}'");
                }
                inputStream = File.OpenRead(resolved);
            }
            else
            {
                return new DatasetResult(false, null, null, 0, "Neither a file upload nor a datasetPath was provided.");
            }

            using (inputStream)
            {
                var relativePath = Path.Combine("datasets", $"{jobId}.jsonl");
                var targetDiskPath = Path.Combine(_dataRoot, relativePath);

                using var reader = new StreamReader(inputStream, Encoding.UTF8);
                using var outStream = File.Create(targetDiskPath);
                using var writer = new StreamWriter(outStream, new UTF8Encoding(false));
                using var sha256 = SHA256.Create();

                int recordCount = 0;
                string? line;

                while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    JsonDocument doc;
                    try
                    {
                        doc = JsonDocument.Parse(line);
                    }
                    catch (Exception ex)
                    {
                        File.Delete(targetDiskPath);
                        return new DatasetResult(false, null, null, 0, $"Invalid JSON at line {recordCount + 1}: {ex.Message}");
                    }

                    using (doc)
                    {
                        var root = doc.RootElement;
                        if (!root.TryGetProperty("prompt", out var promptElem) || promptElem.ValueKind != JsonValueKind.String)
                        {
                            File.Delete(targetDiskPath);
                            return new DatasetResult(false, null, null, 0, $"Line {recordCount + 1} is missing required string 'prompt' property.");
                        }

                        if (!root.TryGetProperty("completion", out var compElem) || compElem.ValueKind != JsonValueKind.String)
                        {
                            File.Delete(targetDiskPath);
                            return new DatasetResult(false, null, null, 0, $"Line {recordCount + 1} is missing required string 'completion' property.");
                        }

                        var prompt = promptElem.GetString()?.Trim() ?? "";
                        var completion = compElem.GetString()?.Trim() ?? "";

                        if (prompt.Length < 3 || completion.Length < 1)
                        {
                            File.Delete(targetDiskPath);
                            return new DatasetResult(false, null, null, 0, $"Line {recordCount + 1} has empty or insufficiently short prompt/completion tokens.");
                        }

                        // Write normalized json string
                        var normalizedJson = JsonSerializer.Serialize(new { prompt, completion });
                        await writer.WriteLineAsync(normalizedJson);
                        recordCount++;
                    }
                }

                await writer.FlushAsync(cancellationToken);

                if (recordCount == 0)
                {
                    File.Delete(targetDiskPath);
                    return new DatasetResult(false, null, null, 0, "Dataset contains no valid records.");
                }

                // Compute hash over written normalized file
                outStream.Position = 0;
                var hashBytes = await sha256.ComputeHashAsync(outStream, cancellationToken);
                var hashHex = Convert.ToHexStringLower(hashBytes);

                _logger.LogInformation("Job {JobId}: Successfully normalized {RecordCount} records (SHA: {Hash}) to {Path}",
                    jobId, recordCount, hashHex, relativePath);

                return new DatasetResult(true, relativePath, hashHex, recordCount, null);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error normalizing dataset for Job {JobId}", jobId);
            return new DatasetResult(false, null, null, 0, $"Error processing dataset: {ex.Message}");
        }
    }
}
