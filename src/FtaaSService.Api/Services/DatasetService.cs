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

    private const long MaxFileSizeBytes = 25 * 1024 * 1024; // 25 MB safety ceiling
    private const int MaxRecordCount = 50_000;

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
                if (uploadedFile.Length > MaxFileSizeBytes)
                {
                    return new DatasetResult(false, null, null, 0,
                        $"Upload size ({uploadedFile.Length / (1024 * 1024)} MB) exceeds maximum permitted dataset limit of {MaxFileSizeBytes / (1024 * 1024)} MB.");
                }
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

                var fileInfo = new FileInfo(resolved);
                if (fileInfo.Length > MaxFileSizeBytes)
                {
                    return new DatasetResult(false, null, null, 0,
                        $"Dataset file size ({fileInfo.Length / (1024 * 1024)} MB) exceeds maximum permitted limit of {MaxFileSizeBytes / (1024 * 1024)} MB.");
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
                var validatedRecords = new List<(string Prompt, string Completion)>();
                int recordCount = 0;
                string? line;
                bool isFirstLine = true;
                bool isCsv = uploadedFile?.FileName.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) == true
                             || existingPath?.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) == true;
                int promptColIdx = 0;
                int compColIdx = 1;

                // Pass 1: Stream-scan input entirely in-memory — zero files created on disk prior to 100% compliance pass
                while ((line = await reader.ReadLineAsync(cancellationToken)) != null)
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;

                    // Full-Row PII Check across all columns/properties (including unmapped metadata)
                    var piiDetected = ScanForPii(line);
                    if (piiDetected is not null)
                    {
                        // Redacted logging: record only Job ID and PII category name, NEVER raw payload or token values
                        _logger.LogWarning("Job {JobId}: Ingestion blocked due to detected customer {PiiType} in payload at record {RecordIndex}",
                            jobId, piiDetected, recordCount + 1);
                        return new DatasetResult(false, null, null, 0,
                            $"Security/PII compliance violation at record {recordCount + 1}: Detected potential real customer {piiDetected} in payload (including unmapped metadata columns). Training dataset rejected by server ingestion gateway.");
                    }

                    if (isFirstLine && !line.TrimStart().StartsWith("{"))
                    {
                        isCsv = true;
                    }

                    string prompt = "";
                    string completion = "";

                    if (isCsv)
                    {
                        var cols = line.Split(',');
                        if (isFirstLine)
                        {
                            isFirstLine = false;
                            var headerLower = cols.Select(c => c.Trim().ToLowerInvariant().Trim('"', '\'')).ToList();
                            var pIdx = headerLower.FindIndex(h => h is "prompt" or "question" or "inquiry");
                            var cIdx = headerLower.FindIndex(h => h is "completion" or "response" or "answer");
                            if (pIdx != -1 && cIdx != -1)
                            {
                                promptColIdx = pIdx;
                                compColIdx = cIdx;
                                continue;
                            }
                        }

                        if (cols.Length > Math.Max(promptColIdx, compColIdx))
                        {
                            prompt = cols[promptColIdx].Trim().Trim('"', '\'');
                            completion = (compColIdx == cols.Length - 1)
                                ? string.Join(",", cols.Skip(compColIdx)).Trim().Trim('"', '\'')
                                : cols[compColIdx].Trim().Trim('"', '\'');
                        }
                        else if (cols.Length >= 2)
                        {
                            prompt = cols[0].Trim().Trim('"', '\'');
                            completion = string.Join(",", cols.Skip(1)).Trim().Trim('"', '\'');
                        }
                        else
                        {
                            return new DatasetResult(false, null, null, 0, $"CSV line {recordCount + 1} has insufficient columns (expected at least prompt and completion).");
                        }
                    }
                    else
                    {
                        isFirstLine = false;
                        JsonDocument doc;
                        try
                        {
                            doc = JsonDocument.Parse(line);
                        }
                        catch (Exception ex)
                        {
                            return new DatasetResult(false, null, null, 0, $"Invalid JSON at line {recordCount + 1}: {ex.Message}");
                        }

                        using (doc)
                        {
                            var root = doc.RootElement;
                            if (!root.TryGetProperty("prompt", out var promptElem) || promptElem.ValueKind != JsonValueKind.String)
                            {
                                return new DatasetResult(false, null, null, 0, $"Line {recordCount + 1} is missing required string 'prompt' property.");
                            }

                            if (!root.TryGetProperty("completion", out var compElem) || compElem.ValueKind != JsonValueKind.String)
                            {
                                return new DatasetResult(false, null, null, 0, $"Line {recordCount + 1} is missing required string 'completion' property.");
                            }

                            prompt = promptElem.GetString()?.Trim() ?? "";
                            completion = compElem.GetString()?.Trim() ?? "";
                        }
                    }

                    if (prompt.Length < 3 || completion.Length < 1)
                    {
                        return new DatasetResult(false, null, null, 0, $"Line {recordCount + 1} has empty or insufficiently short prompt/completion tokens.");
                    }

                    validatedRecords.Add((prompt, completion));
                    recordCount++;

                    if (recordCount >= MaxRecordCount)
                    {
                        return new DatasetResult(false, null, null, 0,
                            $"Dataset exceeds maximum permitted ceiling of {MaxRecordCount:N0} records.");
                    }
                }

                if (recordCount == 0)
                {
                    return new DatasetResult(false, null, null, 0, "Dataset contains no valid records.");
                }

                // Pass 2: Write strictly sanitized normalized { prompt, completion } records to disk
                using var outStream = File.Create(targetDiskPath);
                using var writer = new StreamWriter(outStream, new UTF8Encoding(false));
                using var sha256 = SHA256.Create();

                foreach (var (prompt, completion) in validatedRecords)
                {
                    var normalizedJson = JsonSerializer.Serialize(new { prompt, completion });
                    await writer.WriteLineAsync(normalizedJson);
                }

                await writer.FlushAsync(cancellationToken);

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
