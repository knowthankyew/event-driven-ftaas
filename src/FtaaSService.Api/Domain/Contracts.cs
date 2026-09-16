using System.Text.Json.Serialization;

namespace FtaaSService.Api.Domain;

public sealed record SubmitJobRequest
{
    [JsonPropertyName("jobName")]
    public string JobName { get; init; } = "default-finetune";

    [JsonPropertyName("baseModel")]
    public string BaseModel { get; init; } = "HuggingFaceTB/SmolLM2-135M";

    [JsonPropertyName("datasetPath")]
    public string? DatasetPath { get; init; }

    [JsonPropertyName("hyperparameters")]
    public Hyperparameters Hyperparameters { get; init; } = new();
}

public sealed record SubmitJobResponse
{
    [JsonPropertyName("jobId")]
    public required string JobId { get; init; }

    [JsonPropertyName("jobName")]
    public required string JobName { get; init; }

    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("baseModel")]
    public required string BaseModel { get; init; }

    [JsonPropertyName("datasetPath")]
    public required string DatasetPath { get; init; }

    [JsonPropertyName("datasetHash")]
    public required string DatasetHash { get; init; }

    [JsonPropertyName("createdAt")]
    public required DateTimeOffset CreatedAt { get; init; }
}

public sealed record JobDetailResponse
{
    [JsonPropertyName("jobId")]
    public required string JobId { get; init; }

    [JsonPropertyName("jobName")]
    public required string JobName { get; init; }

    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("baseModel")]
    public required string BaseModel { get; init; }

    [JsonPropertyName("datasetPath")]
    public required string DatasetPath { get; init; }

    [JsonPropertyName("datasetHash")]
    public required string DatasetHash { get; init; }

    [JsonPropertyName("hyperparameters")]
    public required Hyperparameters Hyperparameters { get; init; }

    [JsonPropertyName("progressPercent")]
    public double ProgressPercent { get; init; }

    [JsonPropertyName("currentStep")]
    public int CurrentStep { get; init; }

    [JsonPropertyName("totalSteps")]
    public int TotalSteps { get; init; }

    [JsonPropertyName("currentLoss")]
    public double? CurrentLoss { get; init; }

    [JsonPropertyName("mlflowExperimentId")]
    public string? MlflowExperimentId { get; init; }

    [JsonPropertyName("mlflowRunId")]
    public string? MlflowRunId { get; init; }

    [JsonPropertyName("adapterPath")]
    public string? AdapterPath { get; init; }

    [JsonPropertyName("errorMessage")]
    public string? ErrorMessage { get; init; }

    [JsonPropertyName("sequenceNumber")]
    public long SequenceNumber { get; init; }

    [JsonPropertyName("createdAt")]
    public DateTimeOffset CreatedAt { get; init; }

    [JsonPropertyName("startedAt")]
    public DateTimeOffset? StartedAt { get; init; }

    [JsonPropertyName("finishedAt")]
    public DateTimeOffset? FinishedAt { get; init; }

    [JsonPropertyName("updatedAt")]
    public DateTimeOffset UpdatedAt { get; init; }
}

public sealed record JobRequestedEvent
{
    [JsonPropertyName("jobId")]
    public required string JobId { get; init; }

    [JsonPropertyName("jobName")]
    public required string JobName { get; init; }

    [JsonPropertyName("baseModel")]
    public required string BaseModel { get; init; }

    [JsonPropertyName("datasetPath")]
    public required string DatasetPath { get; init; }

    [JsonPropertyName("datasetHash")]
    public required string DatasetHash { get; init; }

    [JsonPropertyName("hyperparameters")]
    public required Hyperparameters Hyperparameters { get; init; }

    [JsonPropertyName("submittedAt")]
    public DateTimeOffset SubmittedAt { get; init; } = DateTimeOffset.UtcNow;
}

public sealed record JobUpdatedEvent
{
    [JsonPropertyName("jobId")]
    public required string JobId { get; init; }

    [JsonPropertyName("status")]
    public required string Status { get; init; }

    [JsonPropertyName("sequenceNumber")]
    public long SequenceNumber { get; init; }

    [JsonPropertyName("progressPercent")]
    public double ProgressPercent { get; init; }

    [JsonPropertyName("currentStep")]
    public int CurrentStep { get; init; }

    [JsonPropertyName("totalSteps")]
    public int TotalSteps { get; init; }

    [JsonPropertyName("currentLoss")]
    public double? CurrentLoss { get; init; }

    [JsonPropertyName("mlflowExperimentId")]
    public string? MlflowExperimentId { get; init; }

    [JsonPropertyName("mlflowRunId")]
    public string? MlflowRunId { get; init; }

    [JsonPropertyName("adapterPath")]
    public string? AdapterPath { get; init; }

    [JsonPropertyName("errorMessage")]
    public string? ErrorMessage { get; init; }

    [JsonPropertyName("startedAt")]
    public DateTimeOffset? StartedAt { get; init; }

    [JsonPropertyName("finishedAt")]
    public DateTimeOffset? FinishedAt { get; init; }

    [JsonPropertyName("updatedAt")]
    public DateTimeOffset UpdatedAt { get; init; }
}

public sealed record InferenceCompareRequest
{
    [JsonPropertyName("jobId")]
    public string? JobId { get; init; }

    [JsonPropertyName("baseModel")]
    public string? BaseModel { get; init; }

    [JsonPropertyName("adapterPath")]
    public string? AdapterPath { get; init; }

    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    [JsonPropertyName("maxTokens")]
    public int MaxTokens { get; init; } = 64;

    [JsonPropertyName("temperature")]
    public double Temperature { get; init; } = 0.2;
}

public sealed record InferenceCompareResponse
{
    [JsonPropertyName("jobId")]
    public string? JobId { get; init; }

    [JsonPropertyName("baseModel")]
    public required string BaseModel { get; init; }

    [JsonPropertyName("adapterPath")]
    public string? AdapterPath { get; init; }

    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    [JsonPropertyName("baseCompletion")]
    public required string BaseCompletion { get; init; }

    [JsonPropertyName("fineTunedCompletion")]
    public required string FineTunedCompletion { get; init; }

    [JsonPropertyName("latencyMs")]
    public required LatencyBreakdown LatencyMs { get; init; }

    [JsonPropertyName("isSimulated")]
    public bool IsSimulated { get; init; } = false;

    [JsonPropertyName("note")]
    public string? Note { get; init; }
}

public sealed record LatencyBreakdown
{
    [JsonPropertyName("baseModel")]
    public double BaseModel { get; init; }

    [JsonPropertyName("fineTuned")]
    public double FineTuned { get; init; }
}
