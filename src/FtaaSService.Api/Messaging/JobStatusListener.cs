using System.Text;
using System.Text.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Storage;
using Microsoft.Extensions.Options;
using RabbitMQ.Client;
using RabbitMQ.Client.Events;

namespace FtaaSService.Api.Messaging;

public sealed class JobStatusListener : BackgroundService
{
    private readonly RabbitMqConfig _config;
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<JobStatusListener> _logger;
    private readonly ConnectionFactory _factory;

    public JobStatusListener(
        IOptions<RabbitMqConfig> options,
        IServiceProvider serviceProvider,
        ILogger<JobStatusListener> logger)
    {
        _config = options.Value;
        _serviceProvider = serviceProvider;
        _logger = logger;
        _factory = new ConnectionFactory
        {
            HostName = _config.HostName,
            Port = _config.Port,
            UserName = _config.UserName,
            Password = _config.Password
        };
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("JobStatusListener starting background message consumption...");

        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                using var connection = await _factory.CreateConnectionAsync(stoppingToken);
                using var channel = await connection.CreateChannelAsync(cancellationToken: stoppingToken);

                await channel.QueueDeclareAsync(
                    queue: _config.JobUpdatedQueue,
                    durable: true,
                    exclusive: false,
                    autoDelete: false,
                    cancellationToken: stoppingToken
                );

                var consumer = new AsyncEventingBasicConsumer(channel);
                consumer.ReceivedAsync += async (sender, ea) =>
                {
                    try
                    {
                        var body = ea.Body.ToArray();
                        var json = Encoding.UTF8.GetString(body);
                        var update = JsonSerializer.Deserialize<JobUpdatedEvent>(json);

                        if (update is not null)
                        {
                            using var scope = _serviceProvider.CreateScope();
                            var repo = scope.ServiceProvider.GetRequiredService<IJobRepository>();
                            var telemetry = scope.ServiceProvider.GetRequiredService<FtaaSService.Api.Services.IFtaasTelemetry>();
                            
                            var updated = await repo.UpdateStatusIdempotentAsync(update, stoppingToken);
                            if (updated)
                            {
                                _logger.LogInformation("Job {JobId} status transitioned to {Status} (Step {Step}/{Total}, Loss: {Loss})",
                                    update.JobId, update.Status, update.CurrentStep, update.TotalSteps, update.CurrentLoss);

                                var spanAttrs = new Dictionary<string, object>
                                {
                                    ["current_step"] = update.CurrentStep,
                                    ["total_steps"] = update.TotalSteps,
                                    ["progress_pct"] = update.ProgressPercent
                                };
                                if (update.CurrentLoss.HasValue) spanAttrs["loss"] = update.CurrentLoss.Value;
                                if (!string.IsNullOrEmpty(update.AdapterPath)) spanAttrs["adapter_path"] = update.AdapterPath;

                                telemetry.RecordSpan($"job.transition.{update.Status.ToLowerInvariant()}", update.JobId, update.Status, 0, spanAttrs);
                            }
                        }

                        await channel.BasicAckAsync(ea.DeliveryTag, multiple: false);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "Failed processing JobUpdated event.");
                        // nack without requeue if message is malformed
                        await channel.BasicNackAsync(ea.DeliveryTag, multiple: false, requeue: false);
                    }
                };

                await channel.BasicConsumeAsync(
                    queue: _config.JobUpdatedQueue,
                    autoAck: false,
                    consumer: consumer,
                    cancellationToken: stoppingToken
                );

                _logger.LogInformation("JobStatusListener listening on queue: {Queue}", _config.JobUpdatedQueue);

                // Keep open until cancelled
                await Task.Delay(Timeout.Infinite, stoppingToken);
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "RabbitMQ listener encountered an error. Retrying in 5 seconds...");
                await Task.Delay(5000, stoppingToken);
            }
        }
    }
}
