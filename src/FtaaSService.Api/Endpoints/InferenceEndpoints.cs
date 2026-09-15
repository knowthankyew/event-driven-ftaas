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
        catch (HttpRequestException ex)
        {
            return Results.Problem(
                statusCode: StatusCodes.Status503ServiceUnavailable,
                title: "Inference Engine Unavailable",
                detail: $"Unable to reach inference engine at '{inferenceUrl}'. Ensure the inference service is running (Phase 4). Inner: {ex.Message}"
            );
        }
    }
}
