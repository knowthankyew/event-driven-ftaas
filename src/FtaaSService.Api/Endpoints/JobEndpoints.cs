using System.Text.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;

namespace FtaaSService.Api.Endpoints;

public static class JobEndpoints
{
    public static RouteGroupBuilder MapJobEndpoints(this RouteGroupBuilder group)
    {
        group.MapPost("/", SubmitJobAsync)
            .DisableAntiforgery();

        group.MapGet("/", ListJobsAsync);
        group.MapGet("/{id}", GetJobByIdAsync);

        return group;
    }

    private static async Task<IResult> SubmitJobAsync(
        HttpRequest request,
        [FromServices] IDatasetService datasetService,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IEventPublisher eventPublisher,
        CancellationToken cancellationToken)
    {
        string jobId = Guid.NewGuid().ToString("N");
        string jobName = "default-finetune";
        string baseModel = "HuggingFaceTB/SmolLM2-135M";
        string? datasetPath = null;
        IFormFile? uploadedFile = null;
        Hyperparameters hyperparameters = new();

        if (request.HasFormContentType)
        {
            var form = await request.ReadFormAsync(cancellationToken);
            if (form.Files.Count > 0)
            {
                uploadedFile = form.Files[0];
            }

            if (form.TryGetValue("jobName", out var formJobName) && !string.IsNullOrWhiteSpace(formJobName))
                jobName = formJobName.ToString();

            if (form.TryGetValue("baseModel", out var formBaseModel) && !string.IsNullOrWhiteSpace(formBaseModel))
                baseModel = formBaseModel.ToString();

            if (form.TryGetValue("datasetPath", out var formPath) && !string.IsNullOrWhiteSpace(formPath))
                datasetPath = formPath.ToString();

            if (form.TryGetValue("hyperparameters", out var formHp) && !string.IsNullOrWhiteSpace(formHp))
            {
                try
                {
                    hyperparameters = JsonSerializer.Deserialize<Hyperparameters>(formHp.ToString()) ?? new();
                }
                catch { /* fallback to defaults */ }
            }
        }
        else
        {
            SubmitJobRequest? body;
            try
            {
                body = await request.ReadFromJsonAsync<SubmitJobRequest>(cancellationToken: cancellationToken);
            }
            catch (Exception ex)
            {
                return Results.BadRequest(new { error = $"Invalid JSON payload: {ex.Message}" });
            }

            if (body is not null)
            {
                jobName = body.JobName;
                baseModel = body.BaseModel;
                datasetPath = body.DatasetPath;
                hyperparameters = body.Hyperparameters;
            }
        }

        // 1. Validate and normalize dataset
        var normResult = await datasetService.NormalizeAndStoreAsync(jobId, uploadedFile, datasetPath, cancellationToken);
        if (!normResult.IsValid)
        {
            return Results.UnprocessableEntity(new
            {
                error = "Dataset validation failed",
                details = normResult.ErrorMessage
            });
        }

        // 2. Persist Initial Queued Job
        var hpJson = JsonSerializer.Serialize(hyperparameters);
        var job = new FinetuneJob
        {
            Id = jobId,
            JobName = jobName,
            Status = JobStatus.Queued,
            BaseModel = baseModel,
            DatasetPath = normResult.RelativePath!,
            DatasetHash = normResult.Sha256Hash!,
            HyperparametersJson = hpJson,
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };

        await jobRepository.CreateAsync(job, cancellationToken);

        // 3. Publish AMQP Event
        var requestedEvent = new JobRequestedEvent
        {
            JobId = jobId,
            JobName = jobName,
            BaseModel = baseModel,
            DatasetPath = normResult.RelativePath!,
            DatasetHash = normResult.Sha256Hash!,
            Hyperparameters = hyperparameters,
            SubmittedAt = job.CreatedAt
        };

        await eventPublisher.PublishJobRequestedAsync(requestedEvent, cancellationToken);

        var response = new SubmitJobResponse
        {
            JobId = jobId,
            JobName = jobName,
            Status = JobStatus.Queued.ToString(),
            BaseModel = baseModel,
            DatasetPath = normResult.RelativePath!,
            DatasetHash = normResult.Sha256Hash!,
            CreatedAt = job.CreatedAt
        };

        return Results.Accepted($"/api/v1/jobs/{jobId}", response);
    }

    private static async Task<IResult> ListJobsAsync(
        [FromServices] IJobRepository jobRepository,
        [FromQuery] int? limit,
        CancellationToken cancellationToken)
    {
        var jobs = await jobRepository.ListAsync(limit ?? 50, cancellationToken);
        var response = jobs.Select(MapToDetailResponse);
        return Results.Ok(response);
    }

    private static async Task<IResult> GetJobByIdAsync(
        string id,
        [FromServices] IJobRepository jobRepository,
        CancellationToken cancellationToken)
    {
        var job = await jobRepository.GetByIdAsync(id, cancellationToken);
        if (job is null)
        {
            return Results.NotFound(new { error = $"Job '{id}' was not found." });
        }

        return Results.Ok(MapToDetailResponse(job));
    }

    private static JobDetailResponse MapToDetailResponse(FinetuneJob job)
    {
        Hyperparameters hp;
        try
        {
            hp = JsonSerializer.Deserialize<Hyperparameters>(job.HyperparametersJson) ?? new();
        }
        catch
        {
            hp = new();
        }

        return new JobDetailResponse
        {
            JobId = job.Id,
            JobName = job.JobName,
            Status = job.Status.ToString(),
            BaseModel = job.BaseModel,
            DatasetPath = job.DatasetPath,
            DatasetHash = job.DatasetHash,
            Hyperparameters = hp,
            ProgressPercent = job.ProgressPercent,
            CurrentStep = job.CurrentStep,
            TotalSteps = job.TotalSteps,
            CurrentLoss = job.CurrentLoss,
            MlflowExperimentId = job.MlflowExperimentId,
            MlflowRunId = job.MlflowRunId,
            AdapterPath = job.AdapterPath,
            ErrorMessage = job.ErrorMessage,
            SequenceNumber = job.SequenceNumber,
            CreatedAt = job.CreatedAt,
            StartedAt = job.StartedAt,
            FinishedAt = job.FinishedAt,
            UpdatedAt = job.UpdatedAt
        };
    }
}
