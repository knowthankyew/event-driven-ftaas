using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace FtaaSService.Api.Services;

public sealed class DatasetService : IDatasetService
{
    private readonly string _dataRoot;
    private readonly ILogger<DatasetService> _logger;
    private readonly ISurprisalClient? _surprisalClient;

    public DatasetService(
        IConfiguration configuration,
        ILogger<DatasetService> logger,
        ISurprisalClient? surprisalClient = null)
    {
        _logger = logger;
        _surprisalClient = surprisalClient;
        _dataRoot = configuration.GetValue<string>("DATA_ROOT") 
                    ?? Path.Combine(Directory.GetCurrentDirectory(), "data");

        Directory.CreateDirectory(Path.Combine(_dataRoot, "datasets"));
    }

    private const long MaxFileSizeBytes = 25 * 1024 * 1024; // 25 MB safety ceiling
    public const int MaxRecordCount = 50_000;
    public const int MaxPreviewAuditRecords = 2_500;
    private const int MaxBatchAuditSize = 1_000;
    public const int MaxSurprisalPreflightCeiling = 5_000;
    [Obsolete("Use MaxSurprisalPreflightCeiling instead.")]
    public const int MaxHowsoPreflightCeiling = MaxSurprisalPreflightCeiling;
    private static readonly TimeSpan PreflightTimeout = TimeSpan.FromSeconds(90);

    public async Task<DatasetResult> NormalizeAndStoreAsync(
        string jobId,
        IFormFile? uploadedFile,
        string? existingPath,
        bool enableSurprisalPreflight = false,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var (isValid, errorMessage, sourceResolvedPath, validatedRecords) = 
                await ExtractRecordsInternalAsync(jobId, uploadedFile, existingPath, cancellationToken);

            if (!isValid || validatedRecords is null)
            {
                return new DatasetResult(false, null, null, 0, errorMessage);
            }

            SurprisalAuditSummary? surprisalAuditSummary = null;
            string preflightStatus = enableSurprisalPreflight ? "Skipped" : "Disabled";
            double? preflightDurationMs = null;
            var recordsToWrite = validatedRecords;

            if (enableSurprisalPreflight && _surprisalClient != null)
            {
                var preflightStart = Stopwatch.GetTimestamp();
                using var preflightCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                preflightCts.CancelAfter(PreflightTimeout);

                try
                {
                    if (validatedRecords.Count > MaxSurprisalPreflightCeiling)
                    {
                        preflightStatus = "Skipped";
                        _logger.LogWarning("Job {JobId}: Dataset record count ({Count}) exceeds Surprisal preflight ceiling ({Ceiling}). Skipping preflight audit to preserve ingestion latency SLA.",
                            jobId, validatedRecords.Count, MaxSurprisalPreflightCeiling);
                    }
                    else if (validatedRecords.Count <= MaxBatchAuditSize)
                    {
                        _logger.LogInformation("Job {JobId}: Executing Surprisal information-theoretic dataset preflight audit on {RecordCount} records", jobId, validatedRecords.Count);
                        var surprisalRecords = validatedRecords.Select(r => new SurprisalDatasetRecord(r.Prompt, r.Completion, r.Weight)).ToList();
                        var auditResult = await _surprisalClient.AuditDatasetAsync(surprisalRecords, cancellationToken: preflightCts.Token);

                        if (auditResult != null)
                        {
                            surprisalAuditSummary = auditResult.Summary;
                            preflightStatus = "Completed";
                            _logger.LogInformation("Job {JobId}: Surprisal preflight audit complete. Original: {OriginalCount}, Cleaned: {CleanedCount}, Ablated: {AblatedCount}, Anomalous: {AnomalousCount}",
                                jobId, surprisalAuditSummary.OriginalCount, surprisalAuditSummary.CleanedCount, surprisalAuditSummary.AblatedCount, surprisalAuditSummary.AnomalousCount);

                            if (auditResult.CleanedRecords.Count > 0)
                            {
                                recordsToWrite = auditResult.CleanedRecords.Select(r => (r.Prompt, r.Completion, r.Weight)).ToList();
                            }
                            else
                            {
                                surprisalAuditSummary = null;
                                preflightStatus = "FellBack";
                                _logger.LogWarning("Job {JobId}: Surprisal audit cleaned record count is 0. Retaining original validated records.", jobId);
                            }
                        }
                        else
                        {
                            preflightStatus = "FellBack";
                            _logger.LogWarning("Job {JobId}: Surprisal preflight audit returned null or failed. Falling back to standard validated records.", jobId);
                        }
                    }
                    else
                    {
                        // Chunk larger datasets across MaxBatchAuditSize batches to respect service bounds while preventing memory exhaustion
                        var chunks = validatedRecords.Chunk(MaxBatchAuditSize).ToList();
                        _logger.LogInformation("Job {JobId}: Executing chunked Surprisal preflight audit across {ChunkCount} batches ({TotalCount} records total)",
                            jobId, chunks.Count, validatedRecords.Count);

                        var cleanedChunkRecords = new List<SurprisalDatasetRecord>();
                        int totalOriginal = 0;
                        int totalCleaned = 0;
                        int totalAblated = 0;
                        int totalAnomalous = 0;
                        var allAblated = new List<Dictionary<string, object>>();
                        var allAnomalous = new List<Dictionary<string, object>>();
                        var allDeviations = new Dictionary<string, float>();
                        bool allSucceeded = true;

                        for (int chunkIdx = 0; chunkIdx < chunks.Count; chunkIdx++)
                        {
                            var chunk = chunks[chunkIdx];
                            int indexOffset = chunkIdx * MaxBatchAuditSize;
                            var surprisalChunk = chunk.Select(r => new SurprisalDatasetRecord(r.Prompt, r.Completion, r.Weight)).ToList();
                            var chunkResult = await _surprisalClient.AuditDatasetAsync(surprisalChunk, cancellationToken: preflightCts.Token);
                            if (chunkResult is null)
                            {
                                preflightStatus = "FellBack";
                                _logger.LogWarning("Job {JobId}: Surprisal preflight batch {ChunkIndex} returned null or failed. Aborting chunked audit and falling back to standard records.",
                                    jobId, chunkIdx + 1);
                                allSucceeded = false;
                                break;
                            }

                            cleanedChunkRecords.AddRange(chunkResult.CleanedRecords);
                            totalOriginal += chunkResult.Summary.OriginalCount;
                            totalCleaned += chunkResult.Summary.CleanedCount;
                            totalAblated += chunkResult.Summary.AblatedCount;
                            totalAnomalous += chunkResult.Summary.AnomalousCount;

                            foreach (var r in chunkResult.Summary.AblatedRecords)
                            {
                                var cloned = new Dictionary<string, object>(r) { ["batch_index"] = chunkIdx };
                                if (cloned.TryGetValue("ablated_index", out var ai) && int.TryParse(ai.ToString(), out var parsedAi))
                                    cloned["ablated_index"] = parsedAi + indexOffset;
                                if (cloned.TryGetValue("merged_into_index", out var mi) && int.TryParse(mi.ToString(), out var parsedMi))
                                    cloned["merged_into_index"] = parsedMi + indexOffset;
                                allAblated.Add(cloned);
                            }

                            foreach (var r in chunkResult.Summary.AnomalousRecords)
                            {
                                var cloned = new Dictionary<string, object>(r) { ["batch_index"] = chunkIdx };
                                if (cloned.TryGetValue("record_index", out var ri) && int.TryParse(ri.ToString(), out var parsedRi))
                                    cloned["record_index"] = parsedRi + indexOffset;
                                allAnomalous.Add(cloned);
                            }

                            foreach (var (k, v) in chunkResult.Summary.MeasuredDeviations)
                            {
                                allDeviations[k] = v;
                            }
                        }

                        if (allSucceeded && cleanedChunkRecords.Count > 0)
                        {
                            surprisalAuditSummary = new SurprisalAuditSummary(
                                OriginalCount: totalOriginal,
                                CleanedCount: totalCleaned,
                                AblatedCount: totalAblated,
                                AnomalousCount: totalAnomalous,
                                MeasuredDeviations: allDeviations,
                                AnomalousRecords: allAnomalous,
                                AblatedRecords: allAblated
                            );
                            preflightStatus = "Completed";
                            recordsToWrite = cleanedChunkRecords.Select(r => (r.Prompt, r.Completion, r.Weight)).ToList();
                            _logger.LogInformation("Job {JobId}: Chunked Surprisal preflight audit complete. Original: {OriginalCount}, Cleaned: {CleanedCount}, Ablated: {AblatedCount}, Anomalous: {AnomalousCount}",
                                jobId, totalOriginal, totalCleaned, totalAblated, totalAnomalous);
                        }
                        else
                        {
                            preflightStatus = "FellBack";
                        }
                    }
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    throw;
                }
                catch (OperationCanceledException)
                {
                    preflightStatus = "FellBack";
                    _logger.LogWarning("Job {JobId}: Surprisal preflight audit timed out after {TimeoutSeconds}s. Falling back to standard validated records.", jobId, (int)PreflightTimeout.TotalSeconds);
                }
                catch (Exception ex)
                {
                    preflightStatus = "FellBack";
                    _logger.LogWarning(ex, "Job {JobId}: Error during Surprisal preflight audit: {Message}. Falling back to standard validated records.", jobId, ex.Message);
                }
                finally
                {
                    preflightDurationMs = Stopwatch.GetElapsedTime(preflightStart).TotalMilliseconds;
                }
            }

            var relativePath = Path.Combine("datasets", $"{jobId}.jsonl");
            var targetDiskPath = Path.Combine(_dataRoot, relativePath);

            // Pass 2: Write strictly sanitized normalized { prompt, completion[, weight] } records to disk
            using var outStream = File.Create(targetDiskPath);
            using var writer = new StreamWriter(outStream, new UTF8Encoding(false));
            using var sha256 = SHA256.Create();

            foreach (var (prompt, completion, weight) in recordsToWrite)
            {
                var normalizedJson = Math.Abs(weight - 1.0f) > 1e-6f
                    ? JsonSerializer.Serialize(new { prompt, completion, weight })
                    : JsonSerializer.Serialize(new { prompt, completion });
                await writer.WriteLineAsync(normalizedJson);
            }

            await writer.FlushAsync(cancellationToken);

            // Compute hash over written normalized file
            outStream.Position = 0;
            var hashBytes = await sha256.ComputeHashAsync(outStream, cancellationToken);
            var hashHex = Convert.ToHexStringLower(hashBytes);

            if (!string.IsNullOrWhiteSpace(sourceResolvedPath))
            {
                var existingDir = Path.GetDirectoryName(sourceResolvedPath) ?? "";
                var existingFileNameWithoutExt = Path.GetFileNameWithoutExtension(sourceResolvedPath);
                var existingExt = Path.GetExtension(sourceResolvedPath);
                var companionValCandidate = Path.Combine(existingDir, $"{existingFileNameWithoutExt}-val{existingExt}");
                if (File.Exists(companionValCandidate))
                {
                    var valTargetDiskPath = Path.Combine(_dataRoot, "datasets", $"{jobId}-val.jsonl");
                    try
                    {
                        File.Copy(companionValCandidate, valTargetDiskPath, overwrite: true);
                        _logger.LogInformation("Job {JobId}: Copied companion validation dataset from {Source} to {Dest}",
                            jobId, companionValCandidate, valTargetDiskPath);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "Job {JobId}: Failed copying companion validation dataset: {Message}", jobId, ex.Message);
                    }
                }
            }

            _logger.LogInformation("Job {JobId}: Successfully normalized {RecordCount} records (SHA: {Hash}) to {Path}",
                jobId, recordsToWrite.Count, hashHex, relativePath);

            return new DatasetResult(true, relativePath, hashHex, recordsToWrite.Count, null, surprisalAuditSummary, preflightStatus, preflightDurationMs);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error normalizing dataset for Job {JobId}", jobId);
            return new DatasetResult(false, null, null, 0, $"Error processing dataset: {ex.Message}");
        }
    }

    public async Task<(bool IsValid, string? ErrorMessage, List<SurprisalDatasetRecord>? Records)> ValidateAndExtractRecordsAsync(
        IFormFile? uploadedFile,
        string? existingPath,
        CancellationToken cancellationToken = default)
    {
        var (isValid, errorMessage, _, records) = await ExtractRecordsInternalAsync(null, uploadedFile, existingPath, cancellationToken);
        if (!isValid || records is null)
        {
            return (false, errorMessage, null);
        }

        var surprisalRecords = records.Select(r => new SurprisalDatasetRecord(r.Prompt, r.Completion, r.Weight)).ToList();
        return (true, null, surprisalRecords);
    }

    private async Task<(bool IsValid, string? ErrorMessage, string? SourceResolvedPath, List<(string Prompt, string Completion, float Weight)>? Records)> ExtractRecordsInternalAsync(
        string? jobId,
        IFormFile? uploadedFile,
        string? existingPath,
        CancellationToken cancellationToken)
    {
        Stream inputStream;
        string? sourceResolvedPath = null;
        if (uploadedFile is not null && uploadedFile.Length > 0)
        {
            if (uploadedFile.Length > MaxFileSizeBytes)
            {
                return (false,
                    $"Upload size ({uploadedFile.Length / (1024 * 1024)} MB) exceeds maximum permitted dataset limit of {MaxFileSizeBytes / (1024 * 1024)} MB.",
                    null, null);
            }
            inputStream = uploadedFile.OpenReadStream();
        }
        else if (!string.IsNullOrWhiteSpace(existingPath))
        {
            var datasetsRoot = Path.GetFullPath(Path.Combine(_dataRoot, "datasets")) + Path.DirectorySeparatorChar;
            var resolved = Path.GetFullPath(Path.Combine(_dataRoot, existingPath));
            sourceResolvedPath = resolved;
            if (Path.IsPathRooted(existingPath) ||
                !resolved.StartsWith(datasetsRoot, StringComparison.Ordinal))
            {
                return (false, "datasetPath must reference a file under 'datasets/'.", null, null);
            }

            if (!File.Exists(resolved))
            {
                return (false, $"Dataset file not found at '{existingPath}'", null, null);
            }

            var fileInfo = new FileInfo(resolved);
            if (fileInfo.Length > MaxFileSizeBytes)
            {
                return (false,
                    $"Dataset file size ({fileInfo.Length / (1024 * 1024)} MB) exceeds maximum permitted limit of {MaxFileSizeBytes / (1024 * 1024)} MB.",
                    null, null);
            }

            inputStream = File.OpenRead(resolved);
        }
        else
        {
            return (false, "Neither a file upload nor a datasetPath was provided.", null, null);
        }

        using (inputStream)
        {
            using var reader = new StreamReader(inputStream, Encoding.UTF8);
            var validatedRecords = new List<(string Prompt, string Completion, float Weight)>();
            int recordCount = 0;
            string? line;
            bool isFirstLine = true;
            bool isCsv = uploadedFile?.FileName.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) == true
                         || existingPath?.EndsWith(".csv", StringComparison.OrdinalIgnoreCase) == true;
            int promptColIdx = 0;
            int compColIdx = 1;
            int weightColIdx = -1;

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
                        jobId ?? "Preflight", piiDetected, recordCount + 1);
                    return (false,
                        $"Security/PII compliance violation at record {recordCount + 1}: Detected potential real customer {piiDetected} in payload (including unmapped metadata columns). Training dataset rejected by server ingestion gateway.",
                        null, null);
                }

                if (isFirstLine && !line.TrimStart().StartsWith("{"))
                {
                    isCsv = true;
                }

                string prompt = "";
                string completion = "";
                float weight = 1.0f;

                if (isCsv)
                {
                    var cols = ParseCsvRow(line);
                    if (cols is null)
                    {
                        return (false,
                            $"CSV line {recordCount + 1} has unbalanced quotes (multiline fields are unsupported).",
                            null, null);
                    }

                    if (isFirstLine)
                    {
                        isFirstLine = false;
                        var headerLower = cols.Select(c => c.Trim().ToLowerInvariant().Trim('"', '\'')).ToList();
                        var pIdx = headerLower.FindIndex(h => h is "prompt" or "question" or "inquiry");
                        var cIdx = headerLower.FindIndex(h => h is "completion" or "response" or "answer");
                        var wIdx = headerLower.FindIndex(h => h is "weight" or "sample_weight");
                        if (pIdx != -1 && cIdx != -1)
                        {
                            promptColIdx = pIdx;
                            compColIdx = cIdx;
                            weightColIdx = wIdx;
                            continue;
                        }
                    }

                    if (cols.Count > Math.Max(promptColIdx, compColIdx))
                    {
                        prompt = cols[promptColIdx].Trim();
                        completion = cols[compColIdx].Trim();
                        if (weightColIdx != -1 && cols.Count > weightColIdx)
                        {
                            var rawWeightCell = cols[weightColIdx].Trim();
                            if (!string.IsNullOrEmpty(rawWeightCell))
                            {
                                if (!float.TryParse(rawWeightCell, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var csvWeight) ||
                                    float.IsNaN(csvWeight) || float.IsInfinity(csvWeight) || csvWeight <= 0f)
                                {
                                    return (false,
                                        $"CSV line {recordCount + 1} has an invalid or non-positive 'weight' value '{rawWeightCell}'.",
                                        null, null);
                                }
                                weight = csvWeight;
                            }
                        }
                    }
                    else if (cols.Count >= 2)
                    {
                        prompt = cols[0].Trim();
                        completion = cols[1].Trim();
                    }
                    else
                    {
                        return (false,
                            $"CSV line {recordCount + 1} has insufficient columns (expected at least prompt and completion).",
                            null, null);
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
                        return (false, $"Invalid JSON at line {recordCount + 1}: {ex.Message}", null, null);
                    }

                    using (doc)
                    {
                        var root = doc.RootElement;
                        if (!root.TryGetProperty("prompt", out var promptElem) || promptElem.ValueKind != JsonValueKind.String)
                        {
                            return (false, $"Line {recordCount + 1} is missing required string 'prompt' property.", null, null);
                        }

                        if (!root.TryGetProperty("completion", out var compElem) || compElem.ValueKind != JsonValueKind.String)
                        {
                            return (false, $"Line {recordCount + 1} is missing required string 'completion' property.", null, null);
                        }

                        if (root.TryGetProperty("weight", out var weightElem) && weightElem.ValueKind != JsonValueKind.Null)
                        {
                            if (weightElem.ValueKind != JsonValueKind.Number ||
                                !weightElem.TryGetSingle(out var parsedWeight) ||
                                float.IsNaN(parsedWeight) || float.IsInfinity(parsedWeight) || parsedWeight <= 0f)
                            {
                                return (false, $"Line {recordCount + 1} has an invalid or non-positive 'weight' property.", null, null);
                            }
                            weight = parsedWeight;
                        }

                        prompt = promptElem.GetString()?.Trim() ?? "";
                        completion = compElem.GetString()?.Trim() ?? "";
                    }
                }

                if (prompt.Length < 3 || completion.Length < 1)
                {
                    return (false, $"Line {recordCount + 1} has empty or insufficiently short prompt/completion tokens.", null, null);
                }

                if (prompt.Length > 10_000 || completion.Length > 10_000)
                {
                    return (false, $"Line {recordCount + 1} exceeds maximum permitted length of 10,000 characters per field.", null, null);
                }

                validatedRecords.Add((prompt, completion, weight));
                recordCount++;

                if (recordCount > MaxRecordCount)
                {
                    return (false, $"Dataset exceeds maximum permitted ceiling of {MaxRecordCount:N0} records.", null, null);
                }
            }

            if (recordCount == 0)
            {
                return (false, "Dataset contains no valid records.", null, null);
            }

            return (true, null, sourceResolvedPath, validatedRecords);
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

    public static string? ScanForPii(string text)
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

    private static List<string>? ParseCsvRow(string line)
    {
        var fields = new List<string>();
        var sb = new StringBuilder();
        bool inQuotes = false;
        int i = 0;

        while (i < line.Length)
        {
            char c = line[i];

            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        sb.Append('"');
                        i += 2;
                        continue;
                    }
                    else
                    {
                        inQuotes = false;
                        i++;
                        continue;
                    }
                }
                else
                {
                    sb.Append(c);
                    i++;
                }
            }
            else
            {
                if (c == '"')
                {
                    inQuotes = true;
                    i++;
                }
                else if (c == ',')
                {
                    fields.Add(sb.ToString().Trim());
                    sb.Clear();
                    i++;
                }
                else
                {
                    sb.Append(c);
                    i++;
                }
            }
        }

        if (inQuotes)
        {
            return null;
        }

        fields.Add(sb.ToString().Trim());
        return fields;
    }

    public static bool ValidateDirectRecords(IReadOnlyList<SurprisalDatasetRecord>? records, out string? errorMessage)
    {
        if (records is null || records.Count == 0)
        {
            errorMessage = "Records list cannot be null or empty.";
            return false;
        }

        for (int i = 0; i < records.Count; i++)
        {
            var r = records[i];
            if (r is null)
            {
                errorMessage = $"Direct records validation error: Record {i + 1} is null.";
                return false;
            }

            if (string.IsNullOrWhiteSpace(r.Prompt) || r.Prompt.Trim().Length < 3 ||
                string.IsNullOrWhiteSpace(r.Completion) || r.Completion.Trim().Length < 1)
            {
                errorMessage = $"Direct records validation error: Record {i + 1} has empty or insufficiently short prompt/completion tokens.";
                return false;
            }

            if (r.Prompt.Length > 10_000 || r.Completion.Length > 10_000)
            {
                errorMessage = $"Direct records validation error: Record {i + 1} exceeds maximum token character limit of 10,000 characters.";
                return false;
            }

            if (float.IsNaN(r.Weight) || float.IsInfinity(r.Weight) || r.Weight < 0)
            {
                errorMessage = $"Direct records validation error: Record {i + 1} has an invalid or negative weight.";
                return false;
            }

            var piiPrompt = ScanForPii(r.Prompt);
            if (piiPrompt is not null)
            {
                errorMessage = $"Security/PII compliance violation at record {i + 1}: Detected potential real customer {piiPrompt} in prompt payload.";
                return false;
            }

            var piiComp = ScanForPii(r.Completion);
            if (piiComp is not null)
            {
                errorMessage = $"Security/PII compliance violation at record {i + 1}: Detected potential real customer {piiComp} in completion payload.";
                return false;
            }
        }

        errorMessage = null;
        return true;
    }
}
