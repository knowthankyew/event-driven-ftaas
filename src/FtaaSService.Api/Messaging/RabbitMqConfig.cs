namespace FtaaSService.Api.Messaging;

public sealed class RabbitMqConfig
{
    public string HostName { get; set; } = "localhost";
    public int Port { get; set; } = 5672;
    public string UserName { get; set; } = "guest";
    public string Password { get; set; } = "guest";
    public string Exchange { get; set; } = "ftaas.direct";
    public string DlxExchange { get; set; } = "ftaas.dlx";
    public string JobRequestedQueue { get; set; } = "ftaas.jobs.requested";
    public string JobUpdatedQueue { get; set; } = "ftaas.jobs.updated";
    public string DlxQueue { get; set; } = "ftaas.jobs.dlq";
    public string JobRequestedRoutingKey { get; set; } = "job.requested";
    public string JobUpdatedRoutingKey { get; set; } = "job.updated";
}
