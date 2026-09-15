using FtaaSService.Api.Domain;

namespace FtaaSService.Api.Storage;

public interface IJobRepository
{
    Task InitializeAsync(CancellationToken cancellationToken = default);
    Task<FinetuneJob> CreateAsync(FinetuneJob job, CancellationToken cancellationToken = default);
    Task<FinetuneJob?> GetByIdAsync(string id, CancellationToken cancellationToken = default);
    Task<IReadOnlyList<FinetuneJob>> ListAsync(int limit = 50, CancellationToken cancellationToken = default);
    Task<bool> UpdateStatusIdempotentAsync(JobUpdatedEvent update, CancellationToken cancellationToken = default);
}
