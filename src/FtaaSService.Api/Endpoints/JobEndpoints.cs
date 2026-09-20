using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
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
        group.MapGet("/{id}/export/edge", GetEdgeExportStatusAsync);
        group.MapPost("/{id}/export/edge", TriggerEdgeExportAsync);

        return group;
    }

    private static async Task<IResult> SubmitJobAsync(
        HttpRequest request,
        [FromServices] IDatasetService datasetService,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IEventPublisher eventPublisher,
        [FromServices] IFtaasTelemetry telemetry,
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
                    hyperparameters = JsonSerializer.Deserialize<Hyperparameters>(formHp.ToString()) 
                                      ?? throw new JsonException("Hyperparameters payload cannot be null.");
                }
                catch (Exception ex)
                {
                    return Results.BadRequest(new { error = "Invalid hyperparameters JSON.", details = ex.Message });
                }
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

        telemetry.RecordSpan("job.accepted", jobId, JobStatus.Queued.ToString(), 0, new Dictionary<string, object>
        {
            ["base_model"] = baseModel,
            ["dataset_hash"] = normResult.Sha256Hash!,
            ["dataset_relative_path"] = normResult.RelativePath!
        });

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

        telemetry.RecordSpan("job.published", jobId, JobStatus.Queued.ToString(), 0, new Dictionary<string, object>
        {
            ["exchange"] = "ftaas.direct",
            ["routing_key"] = "job.requested"
        });

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

    private static bool IsValidJobId(string id) =>
        !string.IsNullOrWhiteSpace(id) && Regex.IsMatch(id, "^[a-zA-Z0-9_-]+$");

    private static async Task<IResult> GetEdgeExportStatusAsync(
        string id,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IConfiguration configuration,
        CancellationToken cancellationToken)
    {
        if (!IsValidJobId(id))
        {
            return Results.BadRequest(new { error = $"Invalid job ID format: '{id}'." });
        }

        var job = await jobRepository.GetByIdAsync(id, cancellationToken);
        if (job is null)
        {
            return Results.NotFound(new { error = $"Job '{id}' was not found." });
        }

        var dataRoot = configuration.GetValue<string>("DATA_ROOT") ?? Path.Combine(Directory.GetCurrentDirectory(), "data");
        var manifestPath = Path.Combine(dataRoot, "artifacts", id, "edge_onnx", "edge_model_manifest.json");

        if (!File.Exists(manifestPath))
        {
            return Results.NotFound(new 
            { 
                jobId = id, 
                status = "NotExported", 
                message = "Edge ONNX package has not been exported yet. Send POST to export." 
            });
        }

        var json = await File.ReadAllTextAsync(manifestPath, cancellationToken);
        var manifest = JsonSerializer.Deserialize<JsonElement>(json);
        return Results.Ok(new 
        { 
            jobId = id, 
            status = "Exported", 
            manifest = manifest 
        });
    }

    private static async Task<IResult> TriggerEdgeExportAsync(
        string id,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IConfiguration configuration,
        [FromServices] IFtaasTelemetry telemetry,
        CancellationToken cancellationToken)
    {
        if (!IsValidJobId(id))
        {
            return Results.BadRequest(new { error = $"Invalid job ID format: '{id}'." });
        }

        var job = await jobRepository.GetByIdAsync(id, cancellationToken);
        if (job is null)
        {
            return Results.NotFound(new { error = $"Job '{id}' was not found." });
        }

        if (job.Status != JobStatus.Succeeded)
        {
            return Results.BadRequest(new { error = $"Cannot export edge model. Job status is '{job.Status}', but must be 'Succeeded'." });
        }

        var dataRoot = configuration.GetValue<string>("DATA_ROOT") ?? Path.Combine(Directory.GetCurrentDirectory(), "data");
        var edgeDir = Path.Combine(dataRoot, "artifacts", id, "edge_onnx");
        var manifestPath = Path.Combine(edgeDir, "edge_model_manifest.json");
        var modelOnnxPath = Path.Combine(edgeDir, "model.onnx");

        // 1. If real ONNX export already exists, return manifest directly
        if (File.Exists(manifestPath) && File.Exists(modelOnnxPath) && new FileInfo(modelOnnxPath).Length > 1024)
        {
            var existingContent = await File.ReadAllTextAsync(manifestPath, cancellationToken);
            var existingManifest = JsonSerializer.Deserialize<JsonElement>(existingContent);
            return Results.Ok(new
            {
                jobId = id,
                status = "Exported",
                manifest = existingManifest
            });
        }

        // 2. Attempt real ONNX export using worker script
        var (success, output) = await TryExecuteEdgeExportScriptAsync(id, configuration, cancellationToken);

        if (!success || !File.Exists(manifestPath))
        {
            return Results.Problem(
                statusCode: 500,
                title: "Edge Export Failed",
                detail: string.IsNullOrWhiteSpace(output) 
                    ? $"Failed to generate ONNX edge export for job '{id}'." 
                    : output
            );
        }

        telemetry.RecordSpan("job.edge_export_requested", id, job.Status.ToString(), 0, new Dictionary<string, object>
        {
            ["base_model"] = job.BaseModel,
            ["action"] = "edge_export"
        });

        var content = await File.ReadAllTextAsync(manifestPath, cancellationToken);
        var manifest = JsonSerializer.Deserialize<JsonElement>(content);
        return Results.Ok(new
        {
            jobId = id,
            status = "Exported",
            manifest = manifest
        });
    }

    private static async Task<(bool success, string output)> TryExecuteEdgeExportScriptAsync(
        string id,
        IConfiguration configuration,
        CancellationToken cancellationToken)
    {
        var projectRoot = Directory.GetCurrentDirectory();
        var pythonExe = configuration.GetValue<string>("PYTHON_EXE");
        if (string.IsNullOrEmpty(pythonExe))
        {
            var candidates = new[]
            {
                Path.Combine(projectRoot, "src", "FtaaSService.Worker", ".venv", "bin", "python"),
                Path.Combine(projectRoot, "..", "src", "FtaaSService.Worker", ".venv", "bin", "python"),
                Path.Combine(projectRoot, "..", "..", "src", "FtaaSService.Worker", ".venv", "bin", "python")
            };
            pythonExe = candidates.FirstOrDefault(File.Exists) ?? "python3";
        }

        var scriptCandidates = new[]
        {
            Path.Combine(projectRoot, "scripts", "export_edge_adapter.py"),
            Path.Combine(projectRoot, "..", "scripts", "export_edge_adapter.py"),
            Path.Combine(projectRoot, "..", "..", "scripts", "export_edge_adapter.py")
        };
        var scriptPath = scriptCandidates.FirstOrDefault(File.Exists);
        if (scriptPath is null)
        {
            return (false, "export_edge_adapter.py script not found on disk.");
        }

        try
        {
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" --job-id \"{id}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            using var process = Process.Start(psi);
            if (process is null) return (false, "Could not start Python export process.");

            var stdoutTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
            var stderrTask = process.StandardError.ReadToEndAsync(cancellationToken);

            await process.WaitForExitAsync(cancellationToken);
            var stdout = await stdoutTask;
            var stderr = await stderrTask;

            return process.ExitCode == 0
                ? (true, stdout)
                : (false, string.IsNullOrWhiteSpace(stderr) ? stdout : stderr);
        }
        catch (Exception ex)
        {
            return (false, ex.Message);
        }
    }
}
