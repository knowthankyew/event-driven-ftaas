using System.Text.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Endpoints;
using FtaaSService.Api.Services;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace FtaaSService.Api.Tests;

public class EdgeExportTests : IDisposable
{
    private readonly string _testDir;
    private readonly SqliteJobRepository _repository;
    private readonly IConfiguration _configuration;
    private readonly IFtaasTelemetry _telemetry;

    public EdgeExportTests()
    {
        _testDir = Path.Combine(Path.GetTempPath(), "ftaas_edge_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testDir);

        _configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["DATA_ROOT"] = _testDir
            })
            .Build();

        _repository = new SqliteJobRepository(_configuration, NullLogger<SqliteJobRepository>.Instance);
        _telemetry = new FtaasTelemetry(_configuration);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, true); } catch { }
        }
    }

    [Theory]
    [InlineData("../../etc/passwd", false)]
    [InlineData("job;rm -rf /", false)]
    [InlineData("../job-1", false)]
    [InlineData("", false)]
    [InlineData("   ", false)]
    [InlineData("job-valid-123_abc", true)]
    [InlineData("2b743de541b04cff857d3c2f238b4cd6", true)]
    public void JobIdValidation_InvokesProductionValidator_RejectsPathTraversalAndSpecialChars(string jobId, bool expectedValid)
    {
        // Tests the real production validator on JobEndpoints directly
        bool isValid = JobEndpoints.IsValidJobId(jobId);
        Assert.Equal(expectedValid, isValid);
    }

    [Fact]
    public async Task GetEdgeExportStatus_ReturnsNotExported_WhenManifestDoesNotExist()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-edge-not-exported",
            JobName = "test-job",
            Status = JobStatus.Succeeded,
            BaseModel = "HuggingFaceTB/SmolLM2-135M",
            DatasetPath = "datasets/test.jsonl",
            DatasetHash = "hash123",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _repository.CreateAsync(job);

        var result = await JobEndpoints.GetEdgeExportStatusAsync(job.Id, _repository, _configuration, CancellationToken.None);
        Assert.NotNull(result);

        // Expect NotFound result with status NotExported
        var notFoundResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
    }

    [Fact]
    public async Task GetEdgeExportStatus_ReturnsExportedWithManifest_WhenExportExistsOnDisk()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-edge-exported-ok",
            JobName = "lease-semantic-v1",
            Status = JobStatus.Succeeded,
            BaseModel = "HuggingFaceTB/SmolLM2-135M",
            DatasetPath = "datasets/job-edge-1.jsonl",
            DatasetHash = "hash123",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _repository.CreateAsync(job);

        // Prepare simulated edge_onnx artifacts
        var edgeDir = Path.Combine(_testDir, "artifacts", job.Id, "edge_onnx");
        Directory.CreateDirectory(edgeDir);

        var manifestObj = new
        {
            jobId = job.Id,
            modelName = "ftaas-edge-job-edge",
            baseModel = "HuggingFaceTB/SmolLM2-135M",
            architecture = "CausalLM",
            parameterCount = "135M",
            adapterSizeBytes = 1858776,
            onnxFileName = "model.onnx",
            onnxSizeBytes = 1024 * 500,
            quantization = "q4",
            contextLength = 2048,
            supportedExecutionProviders = new[] { "webgpu", "wasm" },
            chatTemplate = "smollm2",
            createdAt = DateTimeOffset.UtcNow.ToString("o"),
            zeroEgressInvariant = true
        };

        var manifestJson = JsonSerializer.Serialize(manifestObj);
        var manifestPath = Path.Combine(edgeDir, "edge_model_manifest.json");
        await File.WriteAllTextAsync(manifestPath, manifestJson);

        // Invoke production endpoint handler
        var result = await JobEndpoints.GetEdgeExportStatusAsync(job.Id, _repository, _configuration, CancellationToken.None);
        var okResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status200OK, okResult.StatusCode);
    }

    [Fact]
    public async Task TriggerEdgeExport_RejectsNonSucceededJob_WithBadRequest()
    {
        await _repository.InitializeAsync();

        var queuedJob = new FinetuneJob
        {
            Id = "job-edge-still-training",
            JobName = "training-job",
            Status = JobStatus.Training,
            BaseModel = "HuggingFaceTB/SmolLM2-135M",
            DatasetPath = "datasets/job.jsonl",
            DatasetHash = "hash123",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _repository.CreateAsync(queuedJob);

        var result = await JobEndpoints.TriggerEdgeExportAsync(
            queuedJob.Id,
            _repository,
            _configuration,
            _telemetry,
            CancellationToken.None);

        var badRequestResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestResult.StatusCode);
    }

    [Fact]
    public async Task TriggerEdgeExport_RejectsInvalidJobIdFormat_WithBadRequest()
    {
        var result = await JobEndpoints.TriggerEdgeExportAsync(
            "../../etc/passwd",
            _repository,
            _configuration,
            _telemetry,
            CancellationToken.None);

        var badRequestResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestResult.StatusCode);
    }

    [Fact]
    public async Task TriggerEdgeExport_ReturnsExported_WhenExistingOnnxModelPresent()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-edge-cached-onnx",
            JobName = "cached-job",
            Status = JobStatus.Succeeded,
            BaseModel = "HuggingFaceTB/SmolLM2-135M",
            DatasetPath = "datasets/cached.jsonl",
            DatasetHash = "hash123",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        await _repository.CreateAsync(job);

        var edgeDir = Path.Combine(_testDir, "artifacts", job.Id, "edge_onnx");
        Directory.CreateDirectory(edgeDir);

        var manifestJson = JsonSerializer.Serialize(new
        {
            jobId = job.Id,
            quantization = "fp32",
            zeroEgressInvariant = true
        });
        await File.WriteAllTextAsync(Path.Combine(edgeDir, "edge_model_manifest.json"), manifestJson);
        // Write mock model.onnx > 1024 bytes
        await File.WriteAllBytesAsync(Path.Combine(edgeDir, "model.onnx"), new byte[2048]);

        var result = await JobEndpoints.TriggerEdgeExportAsync(
            job.Id,
            _repository,
            _configuration,
            _telemetry,
            CancellationToken.None);

        var okResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status200OK, okResult.StatusCode);
    }
}
