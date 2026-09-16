using System.Text;
using FtaaSService.Api.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace FtaaSService.Api.Tests;

public class DatasetServiceTests : IDisposable
{
    private readonly string _testDataDir;
    private readonly DatasetService _service;

    public DatasetServiceTests()
    {
        _testDataDir = Path.Combine(Path.GetTempPath(), "ftaas_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_testDataDir);

        var config = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["DATA_ROOT"] = _testDataDir
            })
            .Build();

        _service = new DatasetService(config, NullLogger<DatasetService>.Instance);
    }

    public void Dispose()
    {
        if (Directory.Exists(_testDataDir))
        {
            try { Directory.Delete(_testDataDir, true); } catch { }
        }
    }

    private static IFormFile CreateFormFile(string content, string fileName = "dataset.jsonl")
    {
        var bytes = Encoding.UTF8.GetBytes(content);
        var stream = new MemoryStream(bytes);
        return new FormFile(stream, 0, bytes.Length, "file", fileName);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_AcceptsCleanValidDataset()
    {
        var jsonl = "{\"prompt\": \"What is the cancellation window?\", \"completion\": \"Standard cancellation is strictly 30 days. [TAG-MSA]\"}\n";
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-clean-1", file, null);

        Assert.True(result.IsValid);
        Assert.Equal(1, result.RecordCount);
        Assert.NotNull(result.Sha256Hash);
        Assert.NotNull(result.RelativePath);

        var diskPath = Path.Combine(_testDataDir, result.RelativePath);
        Assert.True(File.Exists(diskPath));
    }

    [Theory]
    [InlineData("{\"prompt\": \"Customer SSN is 123-45-6789 requesting refund\", \"completion\": \"Denied\"}")]
    [InlineData("{\"prompt\": \"Refund request\", \"completion\": \"Send paperwork to SSN 219 45 8921\"}")]
    public async Task NormalizeAndStoreAsync_RejectsDelimitedSsn(string jsonl)
    {
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-ssn-delim", file, null);

        Assert.False(result.IsValid);
        Assert.Contains("Social Security Number (SSN)", result.ErrorMessage);
    }

    [Theory]
    [InlineData("{\"prompt\": \"Verify account ssn: 219-45-8921 for user\", \"completion\": \"Verified\"}")]
    [InlineData("{\"prompt\": \"Inquiry\", \"completion\": \"Customer social security number 219458921 on file\"}")]
    public async Task NormalizeAndStoreAsync_RejectsContextualSsn(string jsonl)
    {
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-ssn-ctx", file, null);

        Assert.False(result.IsValid);
        Assert.Contains("Social Security Number (SSN)", result.ErrorMessage);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_AcceptsBare9DigitNumber_WithoutContext()
    {
        // Immunity test: verify bare 9-digit account or invoice IDs do NOT trigger false positives
        var jsonl = "{\"prompt\": \"Check status of invoice 219458921 for order\", \"completion\": \"Invoice is processed under order 123456789.\"}\n";
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-bare-digits", file, null);

        Assert.True(result.IsValid);
        Assert.Equal(1, result.RecordCount);
    }

    [Theory]
    [InlineData("{\"prompt\": \"Charge payment card 4532-0150-1234-5671 for bill\", \"completion\": \"Payment processed\"}")]
    [InlineData("{\"prompt\": \"Pay bill\", \"completion\": \"Debit card 4532015012345671 accepted\"}")]
    public async Task NormalizeAndStoreAsync_RejectsValidLuhnCreditCard(string jsonl)
    {
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-card-valid", file, null);

        Assert.False(result.IsValid);
        Assert.Contains("Credit/Debit Card Number", result.ErrorMessage);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_AcceptsNonLuhn16DigitInteger()
    {
        // 16-digit number that fails Luhn checksum should pass as an arbitrary ID
        var jsonl = "{\"prompt\": \"Transaction trace 1111222233334445 completed\", \"completion\": \"Logged to trace log\"}\n";
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-non-luhn", file, null);

        Assert.True(result.IsValid);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_RejectsPiiInUnmappedMetadataColumn_Jsonl()
    {
        // PII inside an unmapped property 'customer_notes' must be detected and rejected
        var jsonl = "{\"prompt\": \"Update contact info\", \"completion\": \"Updated\", \"customer_notes\": \"User SSN is 456-78-1234\"}\n";
        var file = CreateFormFile(jsonl);

        var result = await _service.NormalizeAndStoreAsync("job-unmapped-jsonl", file, null);

        Assert.False(result.IsValid);
        Assert.Contains("Security/PII compliance violation", result.ErrorMessage);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_RejectsPiiInUnmappedMetadataColumn_Csv()
    {
        // CSV with extra metadata column containing an SSN
        var csv = "prompt,completion,internal_notes\n" +
                  "Update address,Address updated,Customer SSN: 123-45-6789 on legacy sheet\n";
        var file = CreateFormFile(csv, "upload.csv");

        var result = await _service.NormalizeAndStoreAsync("job-unmapped-csv", file, null);

        Assert.False(result.IsValid);
        Assert.Contains("Security/PII compliance violation", result.ErrorMessage);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_StripsUnmappedCsvColumns_OnWrite()
    {
        // Clean CSV with extraneous column 'metadata_tag'
        var csv = "prompt,completion,internal_audit_id\n" +
                  "How to reset password,Use self-service portal,AUDIT-992384\n";
        var file = CreateFormFile(csv, "dataset.csv");

        var result = await _service.NormalizeAndStoreAsync("job-csv-strip", file, null);

        Assert.True(result.IsValid);
        var diskPath = Path.Combine(_testDataDir, result.RelativePath!);
        var writtenContent = await File.ReadAllTextAsync(diskPath);

        // Verify the normalized line only contains prompt and completion, with extraneous column completely stripped
        Assert.DoesNotContain("AUDIT-992384", writtenContent);
        Assert.DoesNotContain("internal_audit_id", writtenContent);
        Assert.Contains("How to reset password", writtenContent);
        Assert.Contains("Use self-service portal", writtenContent);
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_DoesNotCreateFileOnDisk_WhenRejected()
    {
        // In-memory verification: verify zero files linger on disk when rejected
        var jsonl = "{\"prompt\": \"Valid row 1\", \"completion\": \"Valid completion 1\"}\n" +
                    "{\"prompt\": \"Invalid row 2 with SSN 123-45-6789\", \"completion\": \"Blocked\"}\n";
        var file = CreateFormFile(jsonl);

        var jobId = "job-zero-disk-leak";
        var expectedPath = Path.Combine(_testDataDir, "datasets", $"{jobId}.jsonl");

        var result = await _service.NormalizeAndStoreAsync(jobId, file, null);

        Assert.False(result.IsValid);
        Assert.False(File.Exists(expectedPath), "Target dataset file must NOT exist on disk after PII rejection.");
    }

    [Fact]
    public async Task NormalizeAndStoreAsync_RejectsFileExceedingMaxSize()
    {
        // Create an IFormFile with length over 25 MB
        var stream = new MemoryStream();
        var oversizedFile = new FormFile(stream, 0, 26 * 1024 * 1024, "file", "oversized.jsonl");

        var result = await _service.NormalizeAndStoreAsync("job-oversized", oversizedFile, null);

        Assert.False(result.IsValid);
        Assert.Contains("exceeds maximum permitted dataset limit", result.ErrorMessage);
    }
}
