using System.Text;
using System.Text.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Endpoints;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace FtaaSService.Api.Tests;

public class OperationalMetricsAndDlqTests : IDisposable
{
    private readonly string _testDir;
    private readonly SqliteJobRepository _repository;
    private readonly FtaasTelemetry _telemetry;

    public OperationalMetricsAndDlqTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "ftaas_metrics_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testDir);

        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["DATA_ROOT"] = _testDir,
                ["PrivacyTelemetry:TelemetryMode"] = "StrictLocal",
                ["PrivacyTelemetry:BurnEnabled"] = "true"
            })
            .Build();

        _repository = new SqliteJobRepository(config, NullLogger<SqliteJobRepository>.Instance);
        _telemetry = new FtaasTelemetry(config);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, true); } catch { }
        }
    }

    private class FakeEventPublisher : IEventPublisher
    {
        public bool IsConnected { get; set; } = true;
        public uint DlqCountToReturn { get; set; } = 0;
        public List<JobRequestedEvent> PublishedEvents { get; } = new();

        public Task PublishJobRequestedAsync(JobRequestedEvent @event, CancellationToken cancellationToken = default)
        {
            PublishedEvents.Add(@event);
            return Task.CompletedTask;
        }

        public Task<uint> GetDlqMessageCountAsync(CancellationToken cancellationToken = default)
        {
            return Task.FromResult(DlqCountToReturn);
        }
    }

    [Fact]
    public async Task MetricsEndpoint_GeneratesValidPrometheusExpositionFormat()
    {
        await _repository.InitializeAsync();

        // Seed 1 Queued, 1 Training, 1 Succeeded, 1 Failed
        var job1 = new FinetuneJob
        {
            Id = "job-m-1", JobName = "m1", Status = JobStatus.Queued,
            BaseModel = "SmolLM2", DatasetPath = "p.jsonl", DatasetHash = "h",
            HyperparametersJson = "{}", CreatedAt = DateTimeOffset.UtcNow, UpdatedAt = DateTimeOffset.UtcNow
        };
        var job2 = new FinetuneJob
        {
            Id = "job-m-2", JobName = "m2", Status = JobStatus.Training,
            BaseModel = "SmolLM2", DatasetPath = "p.jsonl", DatasetHash = "h",
            HyperparametersJson = "{}", CreatedAt = DateTimeOffset.UtcNow, UpdatedAt = DateTimeOffset.UtcNow
        };
        var job3 = new FinetuneJob
        {
            Id = "job-m-3", JobName = "m3", Status = JobStatus.Succeeded,
            BaseModel = "SmolLM2", DatasetPath = "p.jsonl", DatasetHash = "h",
            HyperparametersJson = "{}", CreatedAt = DateTimeOffset.UtcNow, UpdatedAt = DateTimeOffset.UtcNow
        };
        var job4 = new FinetuneJob
        {
            Id = "job-m-4", JobName = "m4", Status = JobStatus.Failed,
            BaseModel = "SmolLM2", DatasetPath = "p.jsonl", DatasetHash = "h",
            HyperparametersJson = "{}", CreatedAt = DateTimeOffset.UtcNow, UpdatedAt = DateTimeOffset.UtcNow
        };

        await _repository.CreateAsync(job1);
        await _repository.CreateAsync(job2);
        await _repository.CreateAsync(job3);
        await _repository.CreateAsync(job4);

        // Record a telemetry span
        _telemetry.RecordSpan("test.span", "job-m-1", "Success", 120);

        var publisher = new FakeEventPublisher { IsConnected = true, DlqCountToReturn = 0 };

        var services = new ServiceCollection().AddLogging().BuildServiceProvider();

        var result = await MetricsEndpoints.GetMetricsAsync(_repository, publisher, _telemetry, CancellationToken.None);
        var context = new DefaultHttpContext { RequestServices = services };
        context.Response.Body = new MemoryStream();
        await result.ExecuteAsync(context);
        context.Response.Body.Seek(0, SeekOrigin.Begin);
        using var reader = new StreamReader(context.Response.Body, Encoding.UTF8);
        var metricsText = await reader.ReadToEndAsync();

        Assert.Equal(StatusCodes.Status200OK, context.Response.StatusCode);
        Assert.Contains("ftaas_uptime_seconds", metricsText);
        Assert.Contains("ftaas_metrics_source_up{source=\"database\"} 1", metricsText);
        Assert.Contains("ftaas_metrics_source_up{source=\"dlq\"} 1", metricsText);
        Assert.Contains("ftaas_jobs_total{status=\"Queued\"} 1", metricsText);
        Assert.Contains("ftaas_jobs_total{status=\"Training\"} 1", metricsText);
        Assert.Contains("ftaas_jobs_total{status=\"Succeeded\"} 1", metricsText);
        Assert.Contains("ftaas_jobs_total{status=\"Failed\"} 1", metricsText);
        Assert.Contains("ftaas_active_jobs 2", metricsText);
        Assert.Contains("ftaas_dlq_message_count 0", metricsText);
        Assert.Contains("ftaas_telemetry_buffered_spans 1", metricsText);
    }

    [Fact]
    public async Task DlqMonitoring_FlagsDeadLetteredMessages()
    {
        await _repository.InitializeAsync();
        var publisherWithDlq = new FakeEventPublisher { IsConnected = true, DlqCountToReturn = 7 };
        var rabbitOptions = Options.Create(new RabbitMqConfig { HostName = "localhost", Port = 5672 });
        var services = new ServiceCollection().AddLogging().BuildServiceProvider();

        // Exercise /healthz handler
        var healthResult = await HealthEndpoints.GetHealthAsync(_repository, publisherWithDlq, rabbitOptions, CancellationToken.None);
        var healthContext = new DefaultHttpContext { RequestServices = services };
        healthContext.Response.Body = new MemoryStream();
        await healthResult.ExecuteAsync(healthContext);
        healthContext.Response.Body.Seek(0, SeekOrigin.Begin);
        using var healthReader = new StreamReader(healthContext.Response.Body, Encoding.UTF8);
        var healthJson = await healthReader.ReadToEndAsync();

        Assert.Equal(StatusCodes.Status200OK, healthContext.Response.StatusCode);
        Assert.Contains("Warning: 7 message(s) in dead-letter queue", healthJson);

        // Exercise /metrics handler
        var metricsResult = await MetricsEndpoints.GetMetricsAsync(_repository, publisherWithDlq, _telemetry, CancellationToken.None);
        var metricsContext = new DefaultHttpContext { RequestServices = services };
        metricsContext.Response.Body = new MemoryStream();
        await metricsResult.ExecuteAsync(metricsContext);
        metricsContext.Response.Body.Seek(0, SeekOrigin.Begin);
        using var metricsReader = new StreamReader(metricsContext.Response.Body, Encoding.UTF8);
        var metricsOutput = await metricsReader.ReadToEndAsync();

        Assert.Equal(StatusCodes.Status200OK, metricsContext.Response.StatusCode);
        Assert.Contains("ftaas_dlq_message_count 7", metricsOutput);
    }

    private class FakeFailingEventPublisher : IEventPublisher
    {
        public bool IsConnected => false;
        public Task PublishJobRequestedAsync(JobRequestedEvent @event, CancellationToken cancellationToken = default)
        {
            throw new InvalidOperationException("Simulated broker transport failure.");
        }
        public Task<uint> GetDlqMessageCountAsync(CancellationToken cancellationToken = default) => Task.FromResult(0u);
    }

    [Fact]
    public async Task SubmitJobAsync_MarksJobFailedAndReturns503_WhenBrokerPublishFails()
    {
        await _repository.InitializeAsync();

        var datasetsDir = Path.Combine(_testDir, "datasets");
        Directory.CreateDirectory(datasetsDir);
        var datasetFile = Path.Combine(datasetsDir, "test_fail.jsonl");
        await File.WriteAllTextAsync(datasetFile, "{\"prompt\": \"hello\", \"completion\": \"world\"}\n");

        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?> { ["DATA_ROOT"] = _testDir })
            .Build();
        var datasetService = new DatasetService(config, NullLogger<DatasetService>.Instance);

        var context = new DefaultHttpContext();
        var json = JsonSerializer.Serialize(new SubmitJobRequest
        {
            JobName = "failing-broker-job",
            BaseModel = SupportedModels.Gemma2_2B_IT,
            DatasetPath = "datasets/test_fail.jsonl"
        });
        context.Request.ContentType = "application/json";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(json));

        var failingPublisher = new FakeFailingEventPublisher();

        var result = await JobEndpoints.SubmitJobAsync(
            context.Request,
            datasetService,
            _repository,
            failingPublisher,
            _telemetry,
            CancellationToken.None
        );

        // Verify 503 Service Unavailable returned
        var statusResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status503ServiceUnavailable, statusResult.StatusCode);

        // Verify job in repository transitioned to Failed, NOT remaining Queued
        var jobs = await _repository.ListAsync(10);
        var job = Assert.Single(jobs);
        Assert.Equal(JobStatus.Failed, job.Status);
        Assert.NotNull(job.ErrorMessage);
        Assert.Contains("Simulated broker transport failure", job.ErrorMessage);
    }
}

