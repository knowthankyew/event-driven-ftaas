namespace FtaaSService.Api.Services.Howso;

using FtaaSService.Api.Services.Surprisal;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

[Obsolete("Use SurprisalDatasetRecord instead.")]
public record HowsoDatasetRecord(string Prompt, string Completion, float Weight = 1.0f)
    : SurprisalDatasetRecord(Prompt, Completion, Weight);

[Obsolete("Use SurprisalAuditSummary instead.")]
public record HowsoAuditSummary(
    int OriginalCount,
    int CleanedCount,
    int AblatedCount,
    int AnomalousCount,
    Dictionary<string, float> MeasuredDeviations,
    List<Dictionary<string, object>> AnomalousRecords,
    List<Dictionary<string, object>> AblatedRecords
) : SurprisalAuditSummary(OriginalCount, CleanedCount, AblatedCount, AnomalousCount, MeasuredDeviations, AnomalousRecords, AblatedRecords);

[Obsolete("Use SurprisalAuditResult instead.")]
public record HowsoAuditResult(
    SurprisalAuditSummary Summary,
    List<SurprisalDatasetRecord> CleanedRecords
) : SurprisalAuditResult(Summary, CleanedRecords);

[Obsolete("Use SurprisalInfluentialCase instead.")]
public record HowsoInfluentialCase(
    int CaseIndex,
    string Tag,
    float SurprisalNats,
    float InfluenceWeightPercent,
    string SourcePrompt
) : SurprisalInfluentialCase(CaseIndex, Tag, SurprisalNats, InfluenceWeightPercent, SourcePrompt);

[Obsolete("Use SurprisalVerifyResult instead.")]
public record HowsoVerifyResult(
    bool IsConforming,
    float MeasuredSurprisalNats,
    string PredictedTag,
    string CandidateTag,
    List<SurprisalInfluentialCase> InfluentialCases,
    string Explanation
) : SurprisalVerifyResult(IsConforming, MeasuredSurprisalNats, PredictedTag, CandidateTag, InfluentialCases, Explanation);

[Obsolete("Use SurprisalHealthResult instead.")]
public record HowsoHealthResult(
    bool IsLive,
    string Status,
    string Engine,
    string License,
    string Message
) : SurprisalHealthResult(IsLive, Status, Engine, License, Message);

[Obsolete("Use ISurprisalClient instead.")]
public interface IHowsoClient : ISurprisalClient
{
}

[Obsolete("Use SurprisalClient instead.")]
public class HowsoClient : SurprisalClient, IHowsoClient
{
    public HowsoClient(
        IHttpClientFactory httpClientFactory,
        IConfiguration configuration,
        ILogger<SurprisalClient> logger)
        : base(httpClientFactory, configuration, logger)
    {
    }
}
