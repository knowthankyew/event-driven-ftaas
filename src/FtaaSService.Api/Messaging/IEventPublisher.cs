using FtaaSService.Api.Domain;

namespace FtaaSService.Api.Messaging;

public interface IEventPublisher
{
    Task PublishJobRequestedAsync(JobRequestedEvent @event, CancellationToken cancellationToken = default);
}
