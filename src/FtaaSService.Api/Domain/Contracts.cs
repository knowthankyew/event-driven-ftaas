using System.Text.Json.Serialization;
using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;

namespace FtaaSService.Api.Domain;

public sealed record SubmitJobRequest
{
    private bool _enableSurprisalPreflight;

    [JsonPropertyName("jobName")]
    public string JobName { get; init; } = "default-finetune";

    [JsonPropertyName("baseModel")]
    public string BaseModel { get; init; } = SupportedModels.Gemma2_2B_IT;

    [JsonPropertyName("datasetPath")]
    public string? DatasetPath { get; init; }

    [JsonPropertyName("hyperparameters")]
    public Hyperparameters Hyperparameters { get; init; } = new();

    [JsonPropertyName("enableSurprisalPreflight")]
    public bool EnableSurprisalPreflight
    {
        get => _enableSurprisalPreflight;
        init => _enableSurprisalPreflight = value;
    }

    [JsonPropertyName("enableHowsoPreflight")]
    [Obsolete("Use EnableSurprisalPreflight instead.")]
    public bool EnableHowsoPreflight
    {
        get => _enableSurprisalPreflight;
        init => _enableSurprisalPreflight = value;
    }
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

    [JsonPropertyName("surprisalAudit")]
    public SurprisalAuditSummary? SurprisalAudit { get; init; }

    [JsonPropertyName("howsoAudit")]
    [Obsolete("Use SurprisalAudit instead.")]
    public HowsoAuditSummary? HowsoAudit => SurprisalAudit is null ? null : new HowsoAuditSummary(
        SurprisalAudit.OriginalCount,
        SurprisalAudit.CleanedCount,
        SurprisalAudit.AblatedCount,
        SurprisalAudit.AnomalousCount,
        SurprisalAudit.MeasuredDeviations,
        SurprisalAudit.AnomalousRecords,
        SurprisalAudit.AblatedRecords
    );

    [JsonPropertyName("surprisalPreflightStatus")]
    public string? SurprisalPreflightStatus { get; init; }

    [JsonPropertyName("howsoPreflightStatus")]
    [Obsolete("Use SurprisalPreflightStatus instead.")]
    public string? HowsoPreflightStatus => SurprisalPreflightStatus;
}

public sealed record DatasetAuditRequest
{
    [JsonPropertyName("datasetPath")]
    public string? DatasetPath { get; init; }

    [JsonPropertyName("records")]
    public List<SurprisalDatasetRecord>? Records { get; init; }

    [JsonPropertyName("ablationThreshold")]
    public float AblationThreshold { get; init; } = 0.15f;

    [JsonPropertyName("anomalyThreshold")]
    public float AnomalyThreshold { get; init; } = 5.0f;
}

public sealed record DatasetAuditResponse
{
    [JsonPropertyName("isValid")]
    public bool IsValid { get; init; }

    [JsonPropertyName("errorMessage")]
    public string? ErrorMessage { get; init; }

    [JsonPropertyName("summary")]
    public SurprisalAuditSummary? Summary { get; init; }

    [JsonPropertyName("cleanedRecordCount")]
    public int CleanedRecordCount { get; init; }
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

    [JsonPropertyName("surprisalProvenance")]
    public SurprisalVerifyResult? SurprisalProvenance { get; init; }

    [JsonPropertyName("howsoProvenance")]
    [Obsolete("Use SurprisalProvenance instead.")]
    public HowsoVerifyResult? HowsoProvenance => SurprisalProvenance is null ? null : new HowsoVerifyResult(
        SurprisalProvenance.IsConforming,
        SurprisalProvenance.MeasuredSurprisalNats,
        SurprisalProvenance.PredictedTag,
        SurprisalProvenance.CandidateTag,
        SurprisalProvenance.InfluentialCases,
        SurprisalProvenance.Explanation
    );
}

public sealed record InferenceVerifyRequest
{
    [JsonPropertyName("jobId")]
    public string? JobId { get; init; }

    [JsonPropertyName("prompt")]
    public required string Prompt { get; init; }

    [JsonPropertyName("candidateCompletion")]
    public required string CandidateCompletion { get; init; }

    [JsonPropertyName("referenceRecords")]
    public List<SurprisalDatasetRecord>? ReferenceRecords { get; init; }

    [JsonPropertyName("maxTolerableSurprisal")]
    public float MaxTolerableSurprisal { get; init; } = 6.0f;
}

public sealed record InferenceVerifyResponse
{
    [JsonPropertyName("isConforming")]
    public bool IsConforming { get; init; }

    [JsonPropertyName("provenance")]
    public SurprisalVerifyResult? Provenance { get; init; }

    [JsonPropertyName("message")]
    public string? Message { get; init; }
}

public sealed record LatencyBreakdown
{
    [JsonPropertyName("baseModel")]
    public double BaseModel { get; init; }

    [JsonPropertyName("fineTuned")]
    public double FineTuned { get; init; }
}

/// <summary>
/// Validated catalog of supported base model IDs for LoRA fine-tuning.
/// Single source of truth on the .NET side — mirrors model_registry.py.
/// </summary>
public static class SupportedModels
{
    public const string SmolLM2 = "HuggingFaceTB/SmolLM2-135M";
    public const string Gemma2_2B_IT = "google/gemma-2-2b-it";
    public const string BitNet2B4T = "microsoft/BitNet-b1.58-2B-4T";

    public static readonly IReadOnlySet<string> All = new HashSet<string>(StringComparer.Ordinal)
    {
        SmolLM2,
        Gemma2_2B_IT,
        BitNet2B4T,
    };

    public static readonly IReadOnlyList<ModelCatalogEntry> Catalog = new List<ModelCatalogEntry>
    {
        new(
            ModelId: Gemma2_2B_IT,
            DisplayName: "Gemma 2 2B IT",
            ParameterCount: "2B",
            ContextLength: 8192,
            MinGpuVramGb: 8.0,
            RequiresHfAuth: true,
            HardwareDisclaimer:
                "Requires ~8 GB GPU VRAM for training and ~5 GB for inference. " +
                "Accept the Gemma license at huggingface.co/google/gemma-2-2b-it and " +
                "set the HF_TOKEN environment variable before starting the worker."
        ),
        new(
            ModelId: SmolLM2,
            DisplayName: "SmolLM2-135M",
            ParameterCount: "135M",
            ContextLength: 2048,
            MinGpuVramGb: 0.5,
            RequiresHfAuth: false,
            HardwareDisclaimer: null
        ),
        new(
            ModelId: BitNet2B4T,
            DisplayName: "BitNet b1.58 2B-4T (Experimental)",
            ParameterCount: "2.4B",
            ContextLength: 4096,
            MinGpuVramGb: 0.0,
            RequiresHfAuth: false,
            HardwareDisclaimer:
                "1.58-bit ternary weight model. Executes natively on CPU using AVX2 SIMD " +
                "integer operations (zero GPU VRAM required). " +
                "Inference requires the bitnet.cpp C++ runtime.",
            IsExperimental: true,
            Tooltip:
                "Experimental Research Architecture: 1.58-bit ternary CPU execution. " +
                "Embedding context bounded to <= 240 tokens (upstream AVX2 batch ceiling); " +
                "generation bounded to <= 16 prompt tokens (context breakdown boundary). " +
                "Production statutory retrieval operates in Retrieval-Only mode."
        ),
    };
}

public sealed record ModelCatalogEntry(
    string ModelId,
    string DisplayName,
    string ParameterCount,
    int ContextLength,
    double MinGpuVramGb,
    bool RequiresHfAuth,
    string? HardwareDisclaimer,
    bool IsExperimental = false,
    string? Tooltip = null
);
