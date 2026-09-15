using System.Text;
using System.Text.Json;
using FtaaSService.Api.Domain;
using Microsoft.Extensions.Options;
using RabbitMQ.Client;

namespace FtaaSService.Api.Messaging;

public sealed class RabbitMqEventPublisher : IEventPublisher, IAsyncDisposable
{
    private readonly RabbitMqConfig _config;
    private readonly ILogger<RabbitMqEventPublisher> _logger;
    private readonly ConnectionFactory _factory;
    private IConnection? _connection;
    private IChannel? _channel;
    private readonly SemaphoreSlim _lock = new(1, 1);

    public RabbitMqEventPublisher(IOptions<RabbitMqConfig> options, ILogger<RabbitMqEventPublisher> logger)
    {
        _config = options.Value;
        _logger = logger;
        _factory = new ConnectionFactory
        {
            HostName = _config.HostName,
            Port = _config.Port,
            UserName = _config.UserName,
            Password = _config.Password
        };
    }

    private async Task EnsureChannelAsync(CancellationToken cancellationToken)
    {
        if (_channel is not null && _channel.IsOpen) return;

        await _lock.WaitAsync(cancellationToken);
        try
        {
            if (_channel is not null && _channel.IsOpen) return;

            _connection ??= await _factory.CreateConnectionAsync(cancellationToken);
            _channel = await _connection.CreateChannelAsync(cancellationToken: cancellationToken);

            // Declare DLX Exchange and DLQ
            await _channel.ExchangeDeclareAsync(_config.DlxExchange, ExchangeType.Direct, durable: true, cancellationToken: cancellationToken);
            await _channel.QueueDeclareAsync(_config.DlxQueue, durable: true, exclusive: false, autoDelete: false, cancellationToken: cancellationToken);
            await _channel.QueueBindAsync(_config.DlxQueue, _config.DlxExchange, _config.JobRequestedRoutingKey, cancellationToken: cancellationToken);

            // Declare Main Exchange
            await _channel.ExchangeDeclareAsync(_config.Exchange, ExchangeType.Direct, durable: true, cancellationToken: cancellationToken);

            // Declare Job Requested Queue with DLX arguments
            var jobQueueArgs = new Dictionary<string, object?>
            {
                { "x-dead-letter-exchange", _config.DlxExchange },
                { "x-dead-letter-routing-key", _config.JobRequestedRoutingKey }
            };
            await _channel.QueueDeclareAsync(_config.JobRequestedQueue, durable: true, exclusive: false, autoDelete: false, arguments: jobQueueArgs, cancellationToken: cancellationToken);
            await _channel.QueueBindAsync(_config.JobRequestedQueue, _config.Exchange, _config.JobRequestedRoutingKey, cancellationToken: cancellationToken);

            // Declare Job Updated Queue
            await _channel.QueueDeclareAsync(_config.JobUpdatedQueue, durable: true, exclusive: false, autoDelete: false, cancellationToken: cancellationToken);
            await _channel.QueueBindAsync(_config.JobUpdatedQueue, _config.Exchange, _config.JobUpdatedRoutingKey, cancellationToken: cancellationToken);

            _logger.LogInformation("RabbitMQ exchanges and queues successfully declared.");
        }
        finally
        {
            _lock.Release();
        }
    }

    public async Task PublishJobRequestedAsync(JobRequestedEvent @event, CancellationToken cancellationToken = default)
    {
        await EnsureChannelAsync(cancellationToken);

        var payload = JsonSerializer.Serialize(@event);
        var body = Encoding.UTF8.GetBytes(payload);

        var props = new BasicProperties
        {
            CorrelationId = @event.JobId,
            ContentType = "application/json",
            DeliveryMode = DeliveryModes.Persistent
        };

        await _channel!.BasicPublishAsync(
            exchange: _config.Exchange,
            routingKey: _config.JobRequestedRoutingKey,
            mandatory: false,
            basicProperties: props,
            body: body,
            cancellationToken: cancellationToken
        );

        _logger.LogInformation("Published JobRequestedEvent for Job {JobId} to routing key {Key}",
            @event.JobId, _config.JobRequestedRoutingKey);
    }

    public async ValueTask DisposeAsync()
    {
        if (_channel is not null)
        {
            await _channel.CloseAsync();
            _channel.Dispose();
        }
        if (_connection is not null)
        {
            await _connection.CloseAsync();
            _connection.Dispose();
        }
    }
}
