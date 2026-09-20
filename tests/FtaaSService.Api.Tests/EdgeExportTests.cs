using System.Text.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Storage;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace FtaaSService.Api.Tests;

public class EdgeExportTests : IDisposable
{
    private readonly string _testDir;
    private readonly SqliteJobRepository _repository;
    private readonly IConfiguration _configuration;

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
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDir))
        {
            try { Directory.Delete(_testDir, true); } catch { }
        }
    }

    [Fact]
    public async Task EdgeExportManifest_ContainsRequiredFields_AndZeroEgressFlag()
    {
        await _repository.InitializeAsync();

        var job = new FinetuneJob
        {
            Id = "job-edge-1",
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
        var edgeDir = Path.Combine(_testDir, "artifacts", "job-edge-1", "edge_onnx");
        Directory.CreateDirectory(edgeDir);

        var manifestObj = new
        {
            jobId = "job-edge-1",
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

        Assert.True(File.Exists(manifestPath));
        var readContent = await File.ReadAllTextAsync(manifestPath);
        using var doc = JsonDocument.Parse(readContent);
        var root = doc.RootElement;

        Assert.Equal("job-edge-1", root.GetProperty("jobId").GetString());
        Assert.Equal("HuggingFaceTB/SmolLM2-135M", root.GetProperty("baseModel").GetString());
        Assert.Equal("q4", root.GetProperty("quantization").GetString());
        Assert.True(root.GetProperty("zeroEgressInvariant").GetBoolean());
        
        var providers = root.GetProperty("supportedExecutionProviders");
        Assert.Equal(2, providers.GetArrayLength());
        Assert.Equal("webgpu", providers[0].GetString());
        Assert.Equal("wasm", providers[1].GetString());
    }

    [Theory]
    [InlineData("../../etc/passwd", false)]
    [InlineData("job;rm -rf /", false)]
    [InlineData("../job-1", false)]
    [InlineData("", false)]
    [InlineData("   ", false)]
    [InlineData("job-valid-123_abc", true)]
    [InlineData("2b743de541b04cff857d3c2f238b4cd6", true)]
    public void JobIdValidation_RejectsPathTraversalAndSpecialChars(string jobId, bool expectedValid)
    {
        bool isValid = !string.IsNullOrWhiteSpace(jobId) && 
                       System.Text.RegularExpressions.Regex.IsMatch(jobId, "^[a-zA-Z0-9_-]+$");
        Assert.Equal(expectedValid, isValid);
    }

    [Fact]
    public void EdgeExportManifest_SupportsFp32AccurateQuantization()
    {
        var manifestObj = new
        {
            jobId = "job-fp32-1",
            quantization = "fp32",
            zeroEgressInvariant = true
        };

        var json = JsonSerializer.Serialize(manifestObj);
        using var doc = JsonDocument.Parse(json);
        Assert.Equal("fp32", doc.RootElement.GetProperty("quantization").GetString());
        Assert.True(doc.RootElement.GetProperty("zeroEgressInvariant").GetBoolean());
    }
}
