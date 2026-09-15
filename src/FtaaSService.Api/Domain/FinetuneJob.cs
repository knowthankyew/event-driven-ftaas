namespace FtaaSService.Api.Domain;

public sealed class FinetuneJob
{
    public required string Id { get; init; }
    public required string JobName { get; init; }
    public required JobStatus Status { get; set; }
    public required string BaseModel { get; init; }
    public required string DatasetPath { get; init; }
    public required string DatasetHash { get; init; }
    public required string HyperparametersJson { get; init; }
    public double ProgressPercent { get; set; }
    public int CurrentStep { get; set; }
    public int TotalSteps { get; set; }
    public double? CurrentLoss { get; set; }
    public string? MlflowExperimentId { get; set; }
    public string? MlflowRunId { get; set; }
    public string? AdapterPath { get; set; }
    public string? ErrorMessage { get; set; }
    public long SequenceNumber { get; set; }
    public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
    public DateTimeOffset? StartedAt { get; set; }
    public DateTimeOffset? FinishedAt { get; set; }
    public DateTimeOffset UpdatedAt { get; set; } = DateTimeOffset.UtcNow;
}
