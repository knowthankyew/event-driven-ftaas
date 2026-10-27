namespace FtaaSService.Api.Services.Surprisal;

public record SurprisalDatasetRecord(
    string Prompt,
    string Completion,
    float Weight = 1.0f
);

public record SurprisalAuditSummary(
    int OriginalCount,
    int CleanedCount,
    int AblatedCount,
    int AnomalousCount,
    Dictionary<string, float> MeasuredDeviations,
    List<Dictionary<string, object>> AnomalousRecords,
    List<Dictionary<string, object>> AblatedRecords
);

public record SurprisalAuditResult(
    SurprisalAuditSummary Summary,
    List<SurprisalDatasetRecord> CleanedRecords
);

public record SurprisalInfluentialCase(
    int CaseIndex,
    string Tag,
    float SurprisalNats,
    float InfluenceWeightPercent,
    string SourcePrompt
);

public record SurprisalVerifyResult(
    bool IsConforming,
    float MeasuredSurprisalNats,
    string PredictedTag,
    string CandidateTag,
    List<SurprisalInfluentialCase> InfluentialCases,
    string Explanation
);

public record SurprisalHealthResult(
    bool IsLive,
    string Status,
    string Engine,
    string License,
    string Message
);

public interface ISurprisalClient
{
    Task<SurprisalHealthResult> CheckHealthAsync(CancellationToken cancellationToken = default);

    Task<SurprisalAuditResult?> AuditDatasetAsync(
        IEnumerable<SurprisalDatasetRecord> records,
        float ablationThreshold = 0.15f,
        float anomalyThreshold = 5.0f,
        CancellationToken cancellationToken = default);

    Task<SurprisalVerifyResult?> VerifyCompletionAsync(
        IEnumerable<SurprisalDatasetRecord> referenceRecords,
        string prompt,
        string candidateCompletion,
        float maxTolerableSurprisal = 6.0f,
        CancellationToken cancellationToken = default);
}
