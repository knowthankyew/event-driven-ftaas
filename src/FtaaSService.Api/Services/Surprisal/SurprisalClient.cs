using System.Net.Http.Json;
using System.Text.Json;

namespace FtaaSService.Api.Services.Surprisal;

public class SurprisalClient : ISurprisalClient, FtaaSService.Api.Services.Howso.IHowsoClient
{
    /// <summary>
    /// Maximum reference records accepted by the containerized Surprisal verification endpoint.
    /// Matches the max_length=2000 constraint on VerifyRequest.reference_records in surprisal-service.
    /// </summary>
    public const int MaxVerifyReferenceRecords = 2000;

    private readonly IHttpClientFactory _httpClientFactory;
    private readonly string _baseUrl;
    private readonly ILogger<SurprisalClient> _logger;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    public SurprisalClient(
        IHttpClientFactory httpClientFactory,
        IConfiguration configuration,
        ILogger<SurprisalClient> logger)
    {
        _httpClientFactory = httpClientFactory;
        _logger = logger;
        _baseUrl = (configuration.GetValue<string>("SURPRISAL_SERVICE_URL")
                    ?? configuration["Surprisal:BaseUrl"]
                    ?? configuration.GetValue<string>("HOWSO_SERVICE_URL")
                    ?? configuration["Howso:BaseUrl"]
                    ?? "http://localhost:5200").TrimEnd('/');
    }

    public async Task<SurprisalHealthResult> CheckHealthAsync(CancellationToken cancellationToken = default)
    {
        var client = _httpClientFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(3);

        try
        {
            var response = await client.GetAsync($"{_baseUrl}/health", cancellationToken);
            if (response.IsSuccessStatusCode)
            {
                using var doc = await response.Content.ReadFromJsonAsync<JsonDocument>(cancellationToken: cancellationToken);
                if (doc is not null)
                {
                    var root = doc.RootElement;
                    string status = root.TryGetProperty("status", out var s) ? s.GetString() ?? "unknown" : "unknown";
                    string engine = root.TryGetProperty("engine", out var e) ? e.GetString() ?? "unknown" : "unknown";
                    string license = root.TryGetProperty("license", out var l) ? l.GetString() ?? "unknown" : "unknown";

                    return new SurprisalHealthResult(
                        IsLive: true,
                        Status: status,
                        Engine: engine,
                        License: license,
                        Message: "Surprisal containerized data arbiter is active"
                    );
                }
            }

            return new SurprisalHealthResult(
                IsLive: false,
                Status: "Unhealthy",
                Engine: "None",
                License: "Unknown",
                Message: $"Surprisal service returned HTTP {(int)response.StatusCode}"
            );
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Surprisal health check failed: {Message}", ex.Message);
            return new SurprisalHealthResult(
                IsLive: false,
                Status: "Offline",
                Engine: "None",
                License: "Unknown",
                Message: "Surprisal containerized service is unreachable"
            );
        }
    }

    public async Task<SurprisalAuditResult?> AuditDatasetAsync(
        IEnumerable<SurprisalDatasetRecord> records,
        float ablationThreshold = 0.15f,
        float anomalyThreshold = 5.0f,
        CancellationToken cancellationToken = default)
    {
        var recordList = records.ToList();
        if (recordList.Count == 0)
        {
            return null;
        }

        var client = _httpClientFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(15);

        var payload = new
        {
            records = recordList.Select(r => new { prompt = r.Prompt, completion = r.Completion, weight = r.Weight }),
            ablation_threshold = ablationThreshold,
            anomaly_threshold = anomalyThreshold
        };

        try
        {
            var response = await client.PostAsJsonAsync($"{_baseUrl}/v1/audit/dataset", payload, JsonOptions, cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Surprisal dataset audit rejected with HTTP {StatusCode}", (int)response.StatusCode);
                return null;
            }

            var result = await response.Content.ReadFromJsonAsync<SurprisalAuditResultDto>(JsonOptions, cancellationToken);
            if (result?.Summary is null)
            {
                return null;
            }

            var cleaned = result.CleanedRecords.Select(r => new SurprisalDatasetRecord(r.Prompt, r.Completion, r.Weight)).ToList();
            var summary = new SurprisalAuditSummary(
                OriginalCount: result.Summary.OriginalCount,
                CleanedCount: result.Summary.CleanedCount,
                AblatedCount: result.Summary.AblatedCount,
                AnomalousCount: result.Summary.AnomalousCount,
                MeasuredDeviations: result.Summary.MeasuredDeviations ?? new Dictionary<string, float>(),
                AnomalousRecords: result.Summary.AnomalousRecords ?? new List<Dictionary<string, object>>(),
                AblatedRecords: result.Summary.AblatedRecords ?? new List<Dictionary<string, object>>()
            );

            return new SurprisalAuditResult(summary, cleaned);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to communicate with Surprisal data arbiter at {BaseUrl}: {Message}. Falling back to standard dataset.", _baseUrl, ex.Message);
            return null;
        }
    }

    public async Task<SurprisalVerifyResult?> VerifyCompletionAsync(
        IEnumerable<SurprisalDatasetRecord> referenceRecords,
        string prompt,
        string candidateCompletion,
        float maxTolerableSurprisal = 6.0f,
        CancellationToken cancellationToken = default)
    {
        var refList = referenceRecords.Take(MaxVerifyReferenceRecords).ToList();
        if (refList.Count == 0)
        {
            return null;
        }

        var client = _httpClientFactory.CreateClient();
        client.Timeout = TimeSpan.FromSeconds(10);

        var payload = new
        {
            reference_records = refList.Select(r => new { prompt = r.Prompt, completion = r.Completion, weight = r.Weight }),
            prompt,
            candidate_completion = candidateCompletion,
            max_tolerable_surprisal = maxTolerableSurprisal
        };

        try
        {
            var response = await client.PostAsJsonAsync($"{_baseUrl}/v1/verify/completion", payload, JsonOptions, cancellationToken);
            if (!response.IsSuccessStatusCode)
            {
                _logger.LogWarning("Surprisal completion verification rejected with HTTP {StatusCode}", (int)response.StatusCode);
                return null;
            }

            var result = await response.Content.ReadFromJsonAsync<SurprisalVerifyResultDto>(JsonOptions, cancellationToken);
            if (result is null)
            {
                return null;
            }

            var cases = result.InfluentialCases.Select(c => new SurprisalInfluentialCase(
                CaseIndex: c.CaseIndex,
                Tag: c.Tag ?? "UNKNOWN",
                SurprisalNats: c.SurprisalNats,
                InfluenceWeightPercent: c.InfluenceWeightPercent,
                SourcePrompt: c.SourcePrompt ?? ""
            )).ToList();

            return new SurprisalVerifyResult(
                IsConforming: result.IsConforming,
                MeasuredSurprisalNats: result.MeasuredSurprisalNats,
                PredictedTag: result.PredictedTag ?? "UNKNOWN",
                CandidateTag: result.CandidateTag ?? "UNKNOWN",
                InfluentialCases: cases,
                Explanation: result.Explanation ?? ""
            );
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Surprisal verify completion failed: {Message}", ex.Message);
            return null;
        }
    }

    private sealed record SurprisalAuditSummaryDto(
        int OriginalCount,
        int CleanedCount,
        int AblatedCount,
        int AnomalousCount,
        Dictionary<string, float>? MeasuredDeviations,
        List<Dictionary<string, object>>? AnomalousRecords,
        List<Dictionary<string, object>>? AblatedRecords
    );

    private sealed record SurprisalRecordDto(
        string Prompt,
        string Completion,
        float Weight
    );

    private sealed record SurprisalAuditResultDto(
        SurprisalAuditSummaryDto Summary,
        List<SurprisalRecordDto> CleanedRecords
    );

    private sealed record SurprisalInfluentialCaseDto(
        int CaseIndex,
        string? Tag,
        float SurprisalNats,
        float InfluenceWeightPercent,
        string? SourcePrompt
    );

    private sealed record SurprisalVerifyResultDto(
        bool IsConforming,
        float MeasuredSurprisalNats,
        string? PredictedTag,
        string? CandidateTag,
        List<SurprisalInfluentialCaseDto> InfluentialCases,
        string? Explanation
    );
}
