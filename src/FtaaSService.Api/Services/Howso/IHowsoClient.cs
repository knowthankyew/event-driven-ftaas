namespace FtaaSService.Api.Services.Howso;

public record HowsoDatasetRecord(
    string Prompt,
    string Completion,
    float Weight = 1.0f
);

public record HowsoAuditSummary(
    int OriginalCount,
    int CleanedCount,
    int AblatedCount,
    int AnomalousCount,
    Dictionary<string, float> MeasuredDeviations,
    List<Dictionary<string, object>> AnomalousRecords,
    List<Dictionary<string, object>> AblatedRecords
);

public record HowsoAuditResult(
    HowsoAuditSummary Summary,
    List<HowsoDatasetRecord> CleanedRecords
);

public record HowsoInfluentialCase(
    int CaseIndex,
    string Tag,
    float SurprisalNats,
    float InfluenceWeightPercent,
    string SourcePrompt
);

public record HowsoVerifyResult(
    bool IsConforming,
    float MeasuredSurprisalNats,
    string PredictedTag,
    string CandidateTag,
    List<HowsoInfluentialCase> InfluentialCases,
    string Explanation
);

public record HowsoHealthResult(
    bool IsLive,
    string Status,
    string Engine,
    string License,
    string Message
);

public interface IHowsoClient
{
    Task<HowsoHealthResult> CheckHealthAsync(CancellationToken cancellationToken = default);

    Task<HowsoAuditResult?> AuditDatasetAsync(
        IEnumerable<HowsoDatasetRecord> records,
        float ablationThreshold = 0.15f,
        float anomalyThreshold = 5.0f,
        CancellationToken cancellationToken = default);

    Task<HowsoVerifyResult?> VerifyCompletionAsync(
        IEnumerable<HowsoDatasetRecord> referenceRecords,
        string prompt,
        string candidateCompletion,
        float maxTolerableSurprisal = 6.0f,
        CancellationToken cancellationToken = default);
}
