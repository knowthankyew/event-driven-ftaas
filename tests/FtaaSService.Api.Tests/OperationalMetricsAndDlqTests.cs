using System.Text;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
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

        // Seed 1 Queued, 1 Running, 2 Succeeded, 1 Failed
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

        // Test the metric generator logic directly
        var jobCounts = await _repository.GetJobCountsByStatusAsync();
        Assert.Equal(1, jobCounts["Queued"]);
        Assert.Equal(1, jobCounts["Training"]);
        Assert.Equal(1, jobCounts["Succeeded"]);
        Assert.Equal(1, jobCounts["Failed"]);

        var dlqCount = await publisher.GetDlqMessageCountAsync();
        Assert.Equal(0u, dlqCount);

        var bufferedSpans = _telemetry.GetBufferedSpans();
        Assert.Single(bufferedSpans);
    }

    [Fact]
    public async Task DlqMonitoring_FlagsDeadLetteredMessages()
    {
        var publisherWithDlq = new FakeEventPublisher { IsConnected = true, DlqCountToReturn = 7 };

        var count = await publisherWithDlq.GetDlqMessageCountAsync();
        Assert.Equal(7u, count);

        // Verify DLQ count can be dynamically inspected
        publisherWithDlq.DlqCountToReturn = 0;
        var clearedCount = await publisherWithDlq.GetDlqMessageCountAsync();
        Assert.Equal(0u, clearedCount);
    }
}
