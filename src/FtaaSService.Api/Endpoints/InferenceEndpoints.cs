using System.Net.Http.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;

namespace FtaaSService.Api.Endpoints;

public static class InferenceEndpoints
{
    public static RouteGroupBuilder MapInferenceEndpoints(this RouteGroupBuilder group)
    {
        group.MapPost("/compare", CompareInferenceAsync);
        return group;
    }

    private static async Task<IResult> CompareInferenceAsync(
        [FromBody] InferenceCompareRequest request,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IHttpClientFactory httpClientFactory,
        [FromServices] IConfiguration configuration,
        CancellationToken cancellationToken)
    {
        string baseModel = request.BaseModel ?? "HuggingFaceTB/SmolLM2-135M";
        string? adapterPath = request.AdapterPath;

        if (!string.IsNullOrWhiteSpace(request.JobId))
        {
            var job = await jobRepository.GetByIdAsync(request.JobId, cancellationToken);
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
        client.Timeout = TimeSpan.FromSeconds(60);

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
            // Graceful preview fallback for non-tech users & studio demo mode when GPU/FastAPI service is offline
            var fallback = GenerateStudioPreview(request.Prompt, request.JobId, baseModel, adapterPath);
            return Results.Ok(fallback);
        }
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
