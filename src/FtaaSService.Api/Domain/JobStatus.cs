namespace FtaaSService.Api.Domain;

public enum JobStatus
{
    Pending,
    Queued,
    Training,
    Succeeded,
    Failed
}
