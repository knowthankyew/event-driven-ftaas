using System.Net.Http.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Services;
using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Caching.Memory;

namespace FtaaSService.Api.Endpoints;

public static class InferenceEndpoints
{
    public static RouteGroupBuilder MapInferenceEndpoints(this RouteGroupBuilder group)
    {
        group.MapPost("/compare", CompareInferenceAsync);
        group.MapPost("/verify", VerifyProvenanceAsync);
        return group;
    }

    internal static async Task<IResult> CompareInferenceAsync(
        [FromBody] InferenceCompareRequest request,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IDatasetService datasetService,
        [FromServices] ISurprisalClient surprisalClient,
        [FromServices] IHttpClientFactory httpClientFactory,
        [FromServices] IConfiguration configuration,
        [FromServices] IMemoryCache memoryCache,
        [FromServices] ILoggerFactory loggerFactory,
        CancellationToken cancellationToken)
    {
        var logger = loggerFactory.CreateLogger("InferenceEndpoints");
        string baseModel = request.BaseModel ?? "HuggingFaceTB/SmolLM2-135M";
        string? adapterPath = request.AdapterPath;
        FinetuneJob? job = null;

        if (!string.IsNullOrWhiteSpace(request.JobId))
        {
            job = await jobRepository.GetByIdAsync(request.JobId, cancellationToken);
            if (job is null)
            {
                return Results.NotFound(new { error = $"Job '{request.JobId}' not found." });
            }

            if (job.Status != JobStatus.Succeeded)
            {
                return Results.BadRequest(new
                {
                    error = $"Cannot run inference on job '{request.JobId}'. Status is '{job.Status}' (must be 'Succeeded')."
                });
            }

            baseModel = job.BaseModel;
            adapterPath = job.AdapterPath;
        }

        var inferenceUrl = configuration.GetValue<string>("INFERENCE_SERVICE_URL") ?? "http://localhost:8000";
        var client = httpClientFactory.CreateClient();
        var proxyTimeoutSeconds = configuration.GetValue<int?>("INFERENCE_PROXY_TIMEOUT_SECONDS") ?? 120;
        client.Timeout = TimeSpan.FromSeconds(Math.Max(30, proxyTimeoutSeconds));

        var apiKey = configuration.GetValue<string>("FTAAS_API_KEY");
        if (!string.IsNullOrWhiteSpace(apiKey))
        {
            client.DefaultRequestHeaders.Add("X-API-Key", apiKey);
        }

        var payload = new
        {
            jobId = request.JobId,
            baseModel,
            adapterPath,
            prompt = request.Prompt,
            maxTokens = request.MaxTokens,
            temperature = request.Temperature
        };

        try
        {
            var response = await client.PostAsJsonAsync($"{inferenceUrl}/api/v1/inference/compare", payload, cancellationToken);
            if (response.IsSuccessStatusCode)
            {
                var data = await response.Content.ReadFromJsonAsync<InferenceCompareResponse>(cancellationToken: cancellationToken);
                if (data is not null)
                {
                    var provenance = await VerifyProvenanceInternalAsync(surprisalClient, datasetService, memoryCache, job, request.Prompt, data.FineTunedCompletion, logger, cancellationToken);
                    return Results.Ok(data with { SurprisalProvenance = provenance });
                }
                return Results.Ok(data);
            }

            var errText = await response.Content.ReadAsStringAsync(cancellationToken);
            return Results.Problem(
                statusCode: (int)response.StatusCode,
                title: "Inference Engine Error",
                detail: string.IsNullOrWhiteSpace(errText) ? "Inference engine returned an error without a message body." : errText
            );
        }
        catch (HttpRequestException)
        {
            if (configuration.GetValue<bool>("STUDIO_PREVIEW_FALLBACK"))
            {
                var fallback = GenerateStudioPreview(request.Prompt, request.JobId, baseModel, adapterPath);
                var provenance = await VerifyProvenanceInternalAsync(surprisalClient, datasetService, memoryCache, job, request.Prompt, fallback.FineTunedCompletion, logger, cancellationToken);
                return Results.Ok(fallback with { SurprisalProvenance = provenance });
            }
            return Results.Problem(
                statusCode: StatusCodes.Status502BadGateway,
                title: "Inference Engine Unreachable",
                detail: "Inference engine service is offline or unreachable."
            );
        }
        catch (TaskCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return Results.Problem(
                statusCode: StatusCodes.Status504GatewayTimeout,
                title: "Inference Engine Timeout",
                detail: "Inference engine timed out while processing completion."
            );
        }
    }

    private static async Task<List<SurprisalDatasetRecord>?> GetDatasetRecordsAsync(
        IDatasetService datasetService,
        IMemoryCache memoryCache,
        FinetuneJob job,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(job.DatasetPath)) return null;

        string cacheKey = $"surprisal_dataset_{(!string.IsNullOrWhiteSpace(job.DatasetHash) ? job.DatasetHash : job.DatasetPath)}";
        if (memoryCache.TryGetValue(cacheKey, out List<SurprisalDatasetRecord>? cached) && cached is not null)
        {
            return cached;
        }

        var (isValid, _, records) = await datasetService.ValidateAndExtractRecordsAsync(null, job.DatasetPath, cancellationToken);
        if (isValid && records is not null && records.Count > 0)
        {
            var cacheEntryOptions = new MemoryCacheEntryOptions()
                .SetSize(records.Count)
                .SetSlidingExpiration(TimeSpan.FromMinutes(30));
            memoryCache.Set(cacheKey, records, cacheEntryOptions);
            return records;
        }

        return null;
    }

    private static List<SurprisalDatasetRecord> SelectRelevantRecords(
        List<SurprisalDatasetRecord> records,
        string prompt,
        int maxRecords = 2000)
    {
        if (records.Count <= maxRecords)
        {
            return records;
        }

        var promptTokens = prompt
            .ToLowerInvariant()
            .Split(new[] { ' ', '\t', '\r', '\n', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .ToHashSet();

        if (promptTokens.Count == 0)
        {
            return records.Take(maxRecords).ToList();
        }

        return records
            .Select(r =>
            {
                int score = 0;
                var words = r.Prompt
                    .ToLowerInvariant()
                    .Split(new[] { ' ', '\t', '\r', '\n', '.', ',', ';', ':', '!', '?' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                foreach (var w in words)
                {
                    if (promptTokens.Contains(w)) score++;
                }
                return (Record: r, Score: score);
            })
            .OrderByDescending(x => x.Score)
            .Take(maxRecords)
            .Select(x => x.Record)
            .ToList();
    }

    private static async Task<SurprisalVerifyResult?> VerifyProvenanceInternalAsync(
        ISurprisalClient surprisalClient,
        IDatasetService datasetService,
        IMemoryCache memoryCache,
        FinetuneJob? job,
        string prompt,
        string fineTunedCompletion,
        ILogger logger,
        CancellationToken cancellationToken)
    {
        try
        {
            List<SurprisalDatasetRecord>? refRecords = null;
            if (job is not null)
            {
                refRecords = await GetDatasetRecordsAsync(datasetService, memoryCache, job, cancellationToken);
                if (refRecords is null || refRecords.Count == 0)
                {
                    return null;
                }
            }
            else
            {
                refRecords = StudioEndpoints.GetDefaultReferenceRecords(prompt);
            }

            var selectedRecords = SelectRelevantRecords(refRecords, prompt);
            return await surprisalClient.VerifyCompletionAsync(selectedRecords, prompt, fineTunedCompletion, cancellationToken: cancellationToken);
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "Failed to compute Surprisal completion provenance; continuing with graceful fallback.");
            return null;
        }
    }

    internal static async Task<IResult> VerifyProvenanceAsync(
        [FromBody] InferenceVerifyRequest request,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IDatasetService datasetService,
        [FromServices] ISurprisalClient surprisalClient,
        [FromServices] IMemoryCache memoryCache,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request.Prompt) || string.IsNullOrWhiteSpace(request.CandidateCompletion))
        {
            return Results.BadRequest(new { error = "Prompt and CandidateCompletion cannot be empty." });
        }

        if (request.Prompt.Length > 10_000 || request.CandidateCompletion.Length > 10_000)
        {
            return Results.BadRequest(new { error = "Prompt and CandidateCompletion cannot exceed 10,000 characters." });
        }

        if (request.ReferenceRecords is not null && request.ReferenceRecords.Count > 0)
        {
            if (request.ReferenceRecords.Count > DatasetService.MaxPreviewAuditRecords)
            {
                return Results.BadRequest(new
                {
                    error = $"Reference records payload exceeds maximum limit of {DatasetService.MaxPreviewAuditRecords:N0} records."
                });
            }

            if (!DatasetService.ValidateDirectRecords(request.ReferenceRecords, out var directError))
            {
                return Results.BadRequest(new { error = directError });
            }
        }

        List<SurprisalDatasetRecord>? refRecords = request.ReferenceRecords;

        if (refRecords is null || refRecords.Count == 0)
        {
            if (!string.IsNullOrWhiteSpace(request.JobId))
            {
                var job = await jobRepository.GetByIdAsync(request.JobId, cancellationToken);
                if (job is null)
                {
                    return Results.NotFound(new { error = $"Job '{request.JobId}' not found." });
                }

                refRecords = await GetDatasetRecordsAsync(datasetService, memoryCache, job, cancellationToken);
                if (refRecords is null || refRecords.Count == 0)
                {
                    return Results.UnprocessableEntity(new { error = $"Dataset for job '{request.JobId}' is empty, missing, or invalid." });
                }
            }
            else
            {
                refRecords = StudioEndpoints.GetDefaultReferenceRecords(request.Prompt);
            }
        }

        var selectedRecords = SelectRelevantRecords(refRecords, request.Prompt);

        var result = await surprisalClient.VerifyCompletionAsync(
            selectedRecords,
            request.Prompt,
            request.CandidateCompletion,
            maxTolerableSurprisal: request.MaxTolerableSurprisal,
            cancellationToken: cancellationToken);

        if (result is null)
        {
            return Results.Problem(
                statusCode: StatusCodes.Status503ServiceUnavailable,
                title: "Surprisal Arbiter Unavailable",
                detail: "Surprisal containerized provenance arbiter failed to respond or is offline."
            );
        }

        return Results.Ok(new InferenceVerifyResponse
        {
            IsConforming = result.IsConforming,
            Provenance = result,
            Message = result.Explanation
        });
    }

    private static InferenceCompareResponse GenerateStudioPreview(string prompt, string? jobId, string baseModel, string? adapterPath)
    {
        string pLower = prompt.ToLowerInvariant();
        string baseCompletion;
        string fineTunedCompletion;

        if (pLower.Contains("transfer") || pLower.Contains("15,000") || pLower.Contains("ach") || pLower.Contains("delay"))
        {
            baseCompletion = "We apologize for the delay with your transfer. In general, bank transfers can take several business days to arrive depending on your financial institution. Please wait another 24 to 48 hours or check with your bank.";
            fineTunedCompletion = "Advise the customer: Under Reg CC and internal ACH policy, external transfers exceeding $10,000 are subject to standard 3-5 business day secondary verification. Reference Ticket Tag: [ACH-HELD-VERIFY]. Mandatory compliance notice: 'Funds are held in accordance with Federal Reserve Regulation CC and FinCEN transaction monitoring guidelines. FDIC insurance coverage applies once funds are credited to your deposit account.'";
        }
        else if (pLower.Contains("cancel") || pLower.Contains("45 days") || pLower.Contains("contract") || pLower.Contains("adoption"))
        {
            baseCompletion = "I understand you would like to cancel your subscription due to low team usage. We are sorry to hear that. I can process your request, but please review our cancellation page for possible refunds.";
            fineTunedCompletion = "Internal Policy Response: Under Section 8.2 of Enterprise SaaS Master Services Agreement, the standard cancellation window is strictly 30 calendar days from provision date. Contracts beyond 30 days are non-refundable for the remaining annual term. Tag: [MSA-SEC8-NONREF]. Recommended escalation: Offer dedicated Customer Success review [CS-REENGAGE] or contract seat reallocation under Addendum B.";
        }
        else if (pLower.Contains("bitcoin") || pLower.Contains("ethereum") || pLower.Contains("crypto") || pLower.Contains("yield") || pLower.Contains("advice") || pLower.Contains("apy"))
        {
            baseCompletion = "Both Bitcoin and Ethereum have shown strong historical growth. Depending on your personal risk appetite and investment goals, allocating a diversified portion to either could be beneficial.";
            fineTunedCompletion = "STRICT REGULATORY DISCLAIMER: We are an execution platform and cannot provide investment, legal, or tax advice. Response: 'We do not offer financial or investment advice. Cryptocurrencies involve substantial market risk and volatility. Please consult a licensed financial advisor before making trading decisions.' Tag: [SEC-NO-ADVISORY].";
        }
        else if (pLower.Contains("gross margin") || pLower.Contains("operating income") || pLower.Contains("revenue") || pLower.Contains("earnings"))
        {
            baseCompletion = "The company reported good improvements in their gross margin and operating income due to lower costs and better sales.";
            fineTunedCompletion = "SENTIMENT: Positive | METRICS: Gross Margin +340bps (43.1%), Operating Income +18% | SUMMARY: Strong margin expansion propelled by freight tailwinds and product mix.";
        }
        else if (pLower.Contains("sso") || pLower.Contains("outage") || pLower.Contains("sev-1"))
        {
            baseCompletion = "We have received your report of a login issue with European SSO. Our team is investigating the problem and will reply as soon as possible.";
            fineTunedCompletion = "Incident Response: Sev-1 SLA requires engineering response within 15 minutes and updates every 30 minutes. Immediately escalate to On-Call Identity Team via PagerDuty [#incident-sev1-idp]. Post public status notice to trust.domain.com under Incident Classification P0-SSO-EMEA. Tag: [INCIDENT-SEV1-ESCALATE].";
        }
        else
        {
            baseCompletion = $"Thank you for contacting support regarding: \"{prompt}\". We are reviewing your inquiry and will follow up shortly.";
            fineTunedCompletion = $"[COMPLIANCE-TAG: TEAM-ADAPTER-ACTIVE] Response according to company policy for inquiry: \"{prompt}\". Disclaimers and enterprise audit logging verified under standard protocol [SLO-P2].";
        }

        return new InferenceCompareResponse
        {
            JobId = jobId,
            BaseModel = baseModel,
            AdapterPath = adapterPath,
            Prompt = prompt,
            BaseCompletion = baseCompletion,
            FineTunedCompletion = fineTunedCompletion,
            LatencyMs = new LatencyBreakdown
            {
                BaseModel = 42.5,
                FineTuned = 45.1
            },
            IsSimulated = true,
            Note = "Inference service (FastAPI :8000) is offline. Displaying high-fidelity studio preview."
        };
    }
}
