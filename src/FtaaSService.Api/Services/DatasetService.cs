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

                        // Server-Side PII Compliance Check
                        var piiDetected = ScanForPii(prompt) ?? ScanForPii(completion);
                        if (piiDetected is not null)
                        {
                            File.Delete(targetDiskPath);
                            _logger.LogWarning("Job {JobId}: Ingestion blocked due to detected customer {PiiType} at record {RecordIndex}",
                                jobId, piiDetected, recordCount + 1);
                            return new DatasetResult(false, null, null, 0,
                                $"Security/PII compliance violation at record {recordCount + 1}: Detected potential real customer {piiDetected}. Training dataset rejected by server ingestion gateway.");
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

    // SSN format: 3-2-4 with delimiters, respecting SSA rules (area != 000, 666, 900-999; group != 00; serial != 0000)
    private static readonly System.Text.RegularExpressions.Regex SsnDelimitedRegex = new(
        @"\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    // Contextual SSN: preceded by 'ssn' or 'social security' to prevent blocking bare 9-digit order/invoice IDs
    private static readonly System.Text.RegularExpressions.Regex SsnContextualRegex = new(
        @"(?i)\b(?:ssn|social\s*security(?:\s*number)?)[^\d]{1,10}((?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4})\b",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    // Credit Card candidates: 13-19 digits formatted or contiguous
    private static readonly System.Text.RegularExpressions.Regex CreditCardCandidateRegex = new(
        @"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{15,16}\b",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    private static string? ScanForPii(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return null;

        if (SsnDelimitedRegex.IsMatch(text) || SsnContextualRegex.IsMatch(text))
        {
            return "Social Security Number (SSN)";
        }

        var ccMatches = CreditCardCandidateRegex.Matches(text);
        foreach (System.Text.RegularExpressions.Match match in ccMatches)
        {
            var digitsOnly = match.Value.Replace("-", "").Replace(" ", "");
            if (digitsOnly.Length >= 13 && digitsOnly.Length <= 19 && PassesLuhn(digitsOnly))
            {
                return "Credit/Debit Card Number";
            }
        }

        return null;
    }

    private static bool PassesLuhn(string number)
    {
        int sum = 0;
        bool alternate = false;
        for (int i = number.Length - 1; i >= 0; i--)
        {
            char c = number[i];
            if (!char.IsDigit(c)) continue;
            int n = c - '0';
            if (alternate)
            {
                n *= 2;
                if (n > 9) n -= 9;
            }
            sum += n;
            alternate = !alternate;
        }
        return sum % 10 == 0;
    }
}
