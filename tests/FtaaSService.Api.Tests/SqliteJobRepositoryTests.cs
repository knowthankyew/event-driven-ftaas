using FtaaSService.Api.Domain;
using FtaaSService.Api.Storage;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace FtaaSService.Api.Tests;

public class SqliteJobRepositoryTests : IDisposable
{
    private readonly string _testDir;
    private readonly SqliteJobRepository _repository;

    public SqliteJobRepositoryTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "ftaas_repo_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testDir);

        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["DATA_ROOT"] = _testDir
            })
            .Build();

        _repository = new SqliteJobRepository(config, NullLogger<SqliteJobRepository>.Instance);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, true); } catch { }
        }
    }

    [Fact]
    public async Task CreateAndRetrieveJob_Succeeds()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-repo-1",
            JobName = "fintech-test-v1",
            Status = JobStatus.Queued,
            BaseModel = "HuggingFaceTB/SmolLM2-135M",
            DatasetPath = "datasets/job-repo-1.jsonl",
            DatasetHash = "abcd1234ef56",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };

        await _repository.CreateAsync(job);

        var retrieved = await _repository.GetByIdAsync("job-repo-1");
        Assert.NotNull(retrieved);
        Assert.Equal("job-repo-1", retrieved.Id);
        Assert.Equal("fintech-test-v1", retrieved.JobName);
        Assert.Equal(JobStatus.Queued, retrieved.Status);
    }

    [Fact]
    public async Task UpdateStatusIdempotentAsync_RejectsOutOfOrderSequenceNumbers()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-seq-test",
            JobName = "seq-test",
            Status = JobStatus.Queued,
            BaseModel = "SmolLM2",
            DatasetPath = "path.jsonl",
            DatasetHash = "hash",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _repository.CreateAsync(job);

        // Sequence 5: Advance to 50%
        var updateSeq5 = new JobUpdatedEvent
        {
            JobId = "job-seq-test",
            Status = "Training",
            CurrentStep = 50,
            TotalSteps = 100,
            ProgressPercent = 50.0,
            CurrentLoss = 1.25,
            SequenceNumber = 5
        };
        var res1 = await _repository.UpdateStatusIdempotentAsync(updateSeq5);
        Assert.True(res1);

        var currentJob = await _repository.GetByIdAsync("job-seq-test");
        Assert.Equal(50, currentJob!.CurrentStep);
        Assert.Equal(5, currentJob.SequenceNumber);

        // Out-of-order sequence: Sequence 3 arrives after Sequence 5
        var updateSeqOld = new JobUpdatedEvent
        {
            JobId = "job-seq-test",
            Status = "Training",
            CurrentStep = 30,
            TotalSteps = 100,
            ProgressPercent = 30.0,
            CurrentLoss = 1.80,
            SequenceNumber = 3 // Stale sequence!
        };
        var resOld = await _repository.UpdateStatusIdempotentAsync(updateSeqOld);
        Assert.False(resOld, "Out-of-order sequence must return false and be discarded.");

        // Verify state was NOT overwritten by stale event
        var unchangedJob = await _repository.GetByIdAsync("job-seq-test");
        Assert.Equal(50, unchangedJob!.CurrentStep);
        Assert.Equal(5, unchangedJob.SequenceNumber);
    }

    [Fact]
    public async Task TerminalStateIdempotency_IgnoresLateInFlightUpdates()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-term-test",
            JobName = "term-test",
            Status = JobStatus.Queued,
            BaseModel = "SmolLM2",
            DatasetPath = "path.jsonl",
            DatasetHash = "hash",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _repository.CreateAsync(job);

        // Transition to terminal Succeeded (Sequence 10)
        var terminalEvent = new JobUpdatedEvent
        {
            JobId = "job-term-test",
            Status = "Succeeded",
            CurrentStep = 100,
            TotalSteps = 100,
            ProgressPercent = 100.0,
            SequenceNumber = 10,
            AdapterPath = "data/adapters/job-term-test"
        };
        await _repository.UpdateStatusIdempotentAsync(terminalEvent);

        var succeededJob = await _repository.GetByIdAsync("job-term-test");
        Assert.Equal(JobStatus.Succeeded, succeededJob!.Status);

        // Late in-flight update arrives attempting to revert to Training (even with higher sequence)
        var lateUpdate = new JobUpdatedEvent
        {
            JobId = "job-term-test",
            Status = "Training",
            CurrentStep = 20,
            TotalSteps = 100,
            ProgressPercent = 20.0,
            SequenceNumber = 99
        };

        var lateResult = await _repository.UpdateStatusIdempotentAsync(lateUpdate);
        Assert.False(lateResult, "Updates to terminal jobs must be rejected.");

        var finalJob = await _repository.GetByIdAsync("job-term-test");
        Assert.Equal(JobStatus.Succeeded, finalJob!.Status);
    }
}
