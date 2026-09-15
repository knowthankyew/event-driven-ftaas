using FtaaSService.Api.Domain;

namespace FtaaSService.Api.Messaging;

public interface IEventPublisher
{
    bool IsConnected { get; }
    Task PublishJobRequestedAsync(JobRequestedEvent @event, CancellationToken cancellationToken = default);
}
