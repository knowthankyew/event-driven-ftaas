using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;

namespace FtaaSService.Api.Endpoints;

public static class JobEndpoints
{
    public static RouteGroupBuilder MapJobEndpoints(this RouteGroupBuilder group)
    {
        group.MapPost("/", SubmitJobAsync)
            .DisableAntiforgery();

        group.MapPost("/audit", AuditDatasetPreviewAsync)
            .DisableAntiforgery();

        group.MapGet("/", ListJobsAsync);
        group.MapGet("/{id}", GetJobByIdAsync);
        group.MapGet("/{id}/export/edge", GetEdgeExportStatusAsync);
        group.MapPost("/{id}/export/edge", TriggerEdgeExportAsync);

        return group;
    }

    internal static async Task<IResult> SubmitJobAsync(
        HttpRequest request,
        [FromServices] IDatasetService datasetService,
        [FromServices] IJobRepository jobRepository,
        [FromServices] IEventPublisher eventPublisher,
        [FromServices] IFtaasTelemetry telemetry,
        CancellationToken cancellationToken)
    {
        string jobId = Guid.NewGuid().ToString("N");
        string jobName = "default-finetune";
        string baseModel = SupportedModels.Gemma2_2B_IT;
        string? datasetPath = null;
        IFormFile? uploadedFile = null;
        Hyperparameters hyperparameters = new();
        bool enableSurprisalPreflight = false;

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

            if (form.TryGetValue("enableSurprisalPreflight", out var formSurprisalPreflight) && !string.IsNullOrWhiteSpace(formSurprisalPreflight))
            {
                if (bool.TryParse(formSurprisalPreflight.ToString(), out var parsedPreflight))
                {
                    enableSurprisalPreflight = parsedPreflight;
                }
            }
            else if (form.TryGetValue("enableHowsoPreflight", out var formHowsoPreflight) && !string.IsNullOrWhiteSpace(formHowsoPreflight))
            {
                if (bool.TryParse(formHowsoPreflight.ToString(), out var parsedPreflight))
                {
                    enableSurprisalPreflight = parsedPreflight;
                }
            }

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
                if (!string.IsNullOrWhiteSpace(body.JobName))
                    jobName = body.JobName;
                if (!string.IsNullOrWhiteSpace(body.BaseModel))
                    baseModel = body.BaseModel;
                datasetPath = body.DatasetPath;
                if (body.Hyperparameters is not null)
                    hyperparameters = body.Hyperparameters;
                enableSurprisalPreflight = body.EnableSurprisalPreflight;
            }
        }

        if (!enableSurprisalPreflight && request.Query.TryGetValue("enableSurprisalPreflight", out var querySurprisal) && bool.TryParse(querySurprisal, out var qs))
        {
            enableSurprisalPreflight = qs;
        }
        else if (!enableSurprisalPreflight && request.Query.TryGetValue("enableHowsoPreflight", out var queryHowso) && bool.TryParse(queryHowso, out var qh))
        {
            enableSurprisalPreflight = qh;
        }

        // 1. Validate hyperparameters range
        if (hyperparameters is null
            || hyperparameters.Epochs is < 1 or > 20
            || hyperparameters.BatchSize is < 1 or > 64
            || hyperparameters.LoraRank is < 1 or > 128
            || hyperparameters.LoraAlpha is < 1 or > 512
            || hyperparameters.LearningRate is <= 0 or > 0.1
            || hyperparameters.LoraDropout is < 0 or >= 1)
        {
            return Results.BadRequest(new
            {
                error = "invalid_hyperparameters",
                message = "Hyperparameters are outside safe operational bounds (epochs 1-20, batchSize 1-64, loraRank 1-128, loraAlpha 1-512, lr (0, 0.1], loraDropout [0, 1))."
            });
        }

        // 2. Validate base model against the supported catalog
        if (!SupportedModels.All.Contains(baseModel))
        {
            return Results.BadRequest(new
            {
                error = "unsupported_base_model",
                message = $"Base model '{baseModel}' is not supported.",
                supported = SupportedModels.All
            });
        }

        // 2. Validate and normalize dataset
        var normResult = await datasetService.NormalizeAndStoreAsync(jobId, uploadedFile, datasetPath, enableSurprisalPreflight, cancellationToken);
        if (!normResult.IsValid)
        {
            return Results.UnprocessableEntity(new
            {
                error = "Dataset validation failed",
                details = normResult.ErrorMessage
            });
        }

        // 3. Persist Initial Queued Job
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

        var spanAttrs = new Dictionary<string, object>
        {
            ["base_model"] = baseModel,
            ["dataset_hash"] = normResult.Sha256Hash!,
            ["dataset_relative_path"] = normResult.RelativePath!
        };

        if (enableSurprisalPreflight)
        {
            spanAttrs["surprisal.preflight.enabled"] = true;
            spanAttrs["surprisal.preflight.status"] = normResult.SurprisalPreflightStatus ?? "Disabled";
            spanAttrs["howso.preflight.enabled"] = true;
            spanAttrs["howso.preflight.status"] = normResult.SurprisalPreflightStatus ?? "Disabled";
            if (normResult.PreflightDurationMs.HasValue)
            {
                spanAttrs["surprisal.preflight.duration_ms"] = normResult.PreflightDurationMs.Value;
                spanAttrs["howso.preflight.duration_ms"] = normResult.PreflightDurationMs.Value;
            }
            if (normResult.SurprisalAudit is not null)
            {
                spanAttrs["surprisal.cases.original_count"] = normResult.SurprisalAudit.OriginalCount;
                spanAttrs["surprisal.cases.ablated_count"] = normResult.SurprisalAudit.AblatedCount;
                spanAttrs["surprisal.cases.anomalous_count"] = normResult.SurprisalAudit.AnomalousCount;
                spanAttrs["howso.cases.original_count"] = normResult.SurprisalAudit.OriginalCount;
                spanAttrs["howso.cases.ablated_count"] = normResult.SurprisalAudit.AblatedCount;
                spanAttrs["howso.cases.anomalous_count"] = normResult.SurprisalAudit.AnomalousCount;
            }
        }

        telemetry.RecordSpan("job.accepted", jobId, JobStatus.Queued.ToString(), 0, spanAttrs);

        // 4. Publish AMQP Event
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

        try
        {
            await eventPublisher.PublishJobRequestedAsync(requestedEvent, cancellationToken);
        }
        catch (Exception ex)
        {
            await jobRepository.UpdateStatusIdempotentAsync(new JobUpdatedEvent
            {
                JobId = jobId,
                Status = JobStatus.Failed.ToString(),
                SequenceNumber = 1,
                ErrorMessage = $"Failed to dispatch job to message broker: {ex.Message}",
                UpdatedAt = DateTimeOffset.UtcNow
            }, CancellationToken.None);

            telemetry.RecordSpan("job.publish_failed", jobId, JobStatus.Failed.ToString(), 0, new Dictionary<string, object>
            {
                ["error"] = ex.Message
            });

            return Results.Problem(
                statusCode: StatusCodes.Status503ServiceUnavailable,
                title: "Message Broker Unavailable",
                detail: "Job submission failed while publishing to the message broker. The job has been marked as Failed."
            );
        }

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
            CreatedAt = job.CreatedAt,
            SurprisalAudit = normResult.SurprisalAudit,
            SurprisalPreflightStatus = normResult.SurprisalPreflightStatus
        };

        return Results.Accepted($"/api/v1/jobs/{jobId}", response);
    }

    private static async Task<IResult> ListJobsAsync(
        [FromServices] IJobRepository jobRepository,
        [FromQuery] int? limit,
        CancellationToken cancellationToken)
    {
        var clampedLimit = Math.Clamp(limit ?? 50, 1, 200);
        var jobs = await jobRepository.ListAsync(clampedLimit, cancellationToken);
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

    internal static bool IsValidJobId(string id) =>
        !string.IsNullOrWhiteSpace(id) && Regex.IsMatch(id, "^[a-zA-Z0-9_-]+$");

    internal static async Task<IResult> GetEdgeExportStatusAsync(
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

    internal static async Task<IResult> TriggerEdgeExportAsync(
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

        var jobLock = _jobExportLocks.GetOrAdd(id, _ => new SemaphoreSlim(1, 1));
        if (!await jobLock.WaitAsync(TimeSpan.FromSeconds(60), cancellationToken))
        {
            return Results.Problem(
                statusCode: StatusCodes.Status503ServiceUnavailable,
                title: "Export Service Busy",
                detail: $"An export operation is already in progress for job '{id}'."
            );
        }

        try
        {
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
        finally
        {
            jobLock.Release();
        }
    }

    private static readonly SemaphoreSlim _exportLock = new(2, 2);
    private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, SemaphoreSlim> _jobExportLocks = new();

    private static async Task<(bool success, string output)> TryExecuteEdgeExportScriptAsync(
        string id,
        IConfiguration configuration,
        CancellationToken cancellationToken)
    {
        if (!await _exportLock.WaitAsync(TimeSpan.FromSeconds(5), cancellationToken))
        {
            return (false, "Export service is currently busy. Please try again shortly.");
        }

        try
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

            using var cts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            cts.CancelAfter(TimeSpan.FromMinutes(15));

            var dataRoot = configuration.GetValue<string>("DATA_ROOT") ?? Path.Combine(projectRoot, "data");
            var psi = new ProcessStartInfo
            {
                FileName = pythonExe,
                Arguments = $"\"{scriptPath}\" --job-id \"{id}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            psi.Environment["DATA_ROOT"] = Path.GetFullPath(dataRoot);

            using var process = Process.Start(psi);
            if (process is null) return (false, "Could not start Python export process.");

            try
            {
                var stdoutTask = process.StandardOutput.ReadToEndAsync(cts.Token);
                var stderrTask = process.StandardError.ReadToEndAsync(cts.Token);

                await process.WaitForExitAsync(cts.Token);
                var stdout = await stdoutTask;
                var stderr = await stderrTask;

                return process.ExitCode == 0
                    ? (true, stdout)
                    : (false, "Edge export process exited with a non-zero exit code.");
            }
            catch (OperationCanceledException)
            {
                try { process.Kill(entireProcessTree: true); } catch { }
                return (false, "Edge export timed out or was cancelled.");
            }
        }
        catch (Exception ex)
        {
            return (false, ex.Message);
        }
        finally
        {
            _exportLock.Release();
        }
    }

    internal static async Task<IResult> AuditDatasetPreviewAsync(
        HttpRequest request,
        [FromServices] IDatasetService datasetService,
        [FromServices] ISurprisalClient surprisalClient,
        CancellationToken cancellationToken)
    {
        IFormFile? uploadedFile = null;
        string? datasetPath = null;
        List<SurprisalDatasetRecord>? directRecords = null;
        float ablationThreshold = 0.15f;
        float anomalyThreshold = 5.0f;

        if (request.HasFormContentType)
        {
            var form = await request.ReadFormAsync(cancellationToken);
            if (form.Files.Count > 0)
            {
                uploadedFile = form.Files[0];
            }
            if (form.TryGetValue("datasetPath", out var pathVal) && !string.IsNullOrWhiteSpace(pathVal))
            {
                datasetPath = pathVal.ToString();
            }
            if (form.TryGetValue("ablationThreshold", out var ablVal) && !string.IsNullOrWhiteSpace(ablVal))
            {
                if (!float.TryParse(ablVal, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var at) || float.IsNaN(at) || float.IsInfinity(at) || at < 0)
                {
                    return Results.BadRequest(new DatasetAuditResponse
                    {
                        IsValid = false,
                        ErrorMessage = "ablationThreshold must be a non-negative finite number."
                    });
                }
                ablationThreshold = at;
            }
            if (form.TryGetValue("anomalyThreshold", out var anomVal) && !string.IsNullOrWhiteSpace(anomVal))
            {
                if (!float.TryParse(anomVal, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var ant) || float.IsNaN(ant) || float.IsInfinity(ant) || ant < 0)
                {
                    return Results.BadRequest(new DatasetAuditResponse
                    {
                        IsValid = false,
                        ErrorMessage = "anomalyThreshold must be a non-negative finite number."
                    });
                }
                anomalyThreshold = ant;
            }
        }
        else
        {
            DatasetAuditRequest? body = null;
            try
            {
                body = await request.ReadFromJsonAsync<DatasetAuditRequest>(cancellationToken: cancellationToken);
            }
            catch (Exception ex)
            {
                return Results.BadRequest(new DatasetAuditResponse
                {
                    IsValid = false,
                    ErrorMessage = $"Invalid JSON payload: {ex.Message}"
                });
            }

            if (body is not null)
            {
                datasetPath = body.DatasetPath;
                directRecords = body.Records;

                if (float.IsNaN(body.AblationThreshold) || float.IsInfinity(body.AblationThreshold) || body.AblationThreshold < 0)
                {
                    return Results.BadRequest(new DatasetAuditResponse
                    {
                        IsValid = false,
                        ErrorMessage = "ablationThreshold must be a non-negative finite number."
                    });
                }
                ablationThreshold = body.AblationThreshold;

                if (float.IsNaN(body.AnomalyThreshold) || float.IsInfinity(body.AnomalyThreshold) || body.AnomalyThreshold < 0)
                {
                    return Results.BadRequest(new DatasetAuditResponse
                    {
                        IsValid = false,
                        ErrorMessage = "anomalyThreshold must be a non-negative finite number."
                    });
                }
                anomalyThreshold = body.AnomalyThreshold;
            }
        }

        List<SurprisalDatasetRecord> recordsToAudit;

        if (directRecords is not null && directRecords.Count > 0)
        {
            if (directRecords.Count > DatasetService.MaxPreviewAuditRecords)
            {
                return Results.UnprocessableEntity(new DatasetAuditResponse
                {
                    IsValid = false,
                    ErrorMessage = $"Direct records payload exceeds maximum preview audit limit of {DatasetService.MaxPreviewAuditRecords:N0} records. For larger datasets, submit a job with enableSurprisalPreflight enabled."
                });
            }

            if (!DatasetService.ValidateDirectRecords(directRecords, out var directError))
            {
                return Results.UnprocessableEntity(new DatasetAuditResponse
                {
                    IsValid = false,
                    ErrorMessage = directError
                });
            }

            recordsToAudit = directRecords;
        }
        else
        {
            var (isValid, errorMessage, extractedRecords) = await datasetService.ValidateAndExtractRecordsAsync(
                uploadedFile, datasetPath, cancellationToken);

            if (!isValid || extractedRecords is null)
            {
                return Results.UnprocessableEntity(new DatasetAuditResponse
                {
                    IsValid = false,
                    ErrorMessage = errorMessage ?? "Dataset validation failed"
                });
            }

            if (extractedRecords.Count > DatasetService.MaxPreviewAuditRecords)
            {
                return Results.UnprocessableEntity(new DatasetAuditResponse
                {
                    IsValid = false,
                    ErrorMessage = $"Preview audit dataset exceeds maximum preview limit of {DatasetService.MaxPreviewAuditRecords:N0} records (contained {extractedRecords.Count:N0} records). For larger datasets, submit a training job with enableSurprisalPreflight enabled."
                });
            }

            recordsToAudit = extractedRecords;
        }

        var auditResult = await surprisalClient.AuditDatasetAsync(
            recordsToAudit,
            ablationThreshold: ablationThreshold,
            anomalyThreshold: anomalyThreshold,
            cancellationToken: cancellationToken);

        if (auditResult is null)
        {
            return Results.Problem(
                statusCode: StatusCodes.Status503ServiceUnavailable,
                title: "Surprisal Arbiter Unavailable",
                detail: "Surprisal containerized audit service failed to respond or is offline."
            );
        }

        return Results.Ok(new DatasetAuditResponse
        {
            IsValid = true,
            Summary = auditResult.Summary,
            CleanedRecordCount = auditResult.CleanedRecords.Count
        });
    }
}
