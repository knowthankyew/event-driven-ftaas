using System.Net;
using System.Text;
using System.Text.Json;
using FtaaSService.Api.Domain;
using FtaaSService.Api.Endpoints;
using FtaaSService.Api.Services;
using FtaaSService.Api.Services.Howso;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace FtaaSService.Api.Tests;

public class InferenceProvenanceTests
{
    private static readonly IMemoryCache TestCache = new MemoryCache(new MemoryCacheOptions());
    private sealed class DelegatingMockHandler : HttpMessageHandler
    {
        private readonly Func<HttpRequestMessage, HttpResponseMessage> _responder;

        public DelegatingMockHandler(Func<HttpRequestMessage, HttpResponseMessage> responder)
        {
            _responder = responder;
        }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            return Task.FromResult(_responder(request));
        }
    }

    private sealed class MockHttpClientFactory : IHttpClientFactory
    {
        private readonly HttpMessageHandler _handler;

        public MockHttpClientFactory(HttpMessageHandler handler)
        {
            _handler = handler;
        }

        public HttpClient CreateClient(string name) => new(_handler, disposeHandler: false);
    }

    private sealed class FakeJobRepository : IJobRepository
    {
        private readonly Dictionary<string, FinetuneJob> _jobs = new();

        public void AddJob(FinetuneJob job) => _jobs[job.Id] = job;

        public Task InitializeAsync(CancellationToken cancellationToken = default) => Task.CompletedTask;
        public Task<FinetuneJob> CreateAsync(FinetuneJob job, CancellationToken cancellationToken = default)
        {
            _jobs[job.Id] = job;
            return Task.FromResult(job);
        }
        public Task<FinetuneJob?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
        {
            _jobs.TryGetValue(id, out var job);
            return Task.FromResult(job);
        }
        public Task<IReadOnlyList<FinetuneJob>> ListAsync(int limit = 50, CancellationToken cancellationToken = default)
            => Task.FromResult<IReadOnlyList<FinetuneJob>>(_jobs.Values.ToList());
        public Task<bool> UpdateStatusIdempotentAsync(JobUpdatedEvent update, CancellationToken cancellationToken = default) => Task.FromResult(true);
        public Task<IReadOnlyDictionary<string, int>> GetJobCountsByStatusAsync(CancellationToken cancellationToken = default)
            => Task.FromResult<IReadOnlyDictionary<string, int>>(new Dictionary<string, int>());
    }

    private sealed class FakeDatasetService : IDatasetService
    {
        public List<HowsoDatasetRecord>? RecordsToReturn { get; set; }
        public bool ReturnValid { get; set; } = true;
        public string? ErrorToReturn { get; set; }

        public Task<DatasetResult> NormalizeAndStoreAsync(
            string jobId,
            IFormFile? uploadedFile,
            string? existingPath,
            bool enableHowsoPreflight = false,
            CancellationToken cancellationToken = default)
        {
            return Task.FromResult(new DatasetResult(true, "path/test.jsonl", "hash", 10, null));
        }

        public Task<(bool IsValid, string? ErrorMessage, List<HowsoDatasetRecord>? Records)> ValidateAndExtractRecordsAsync(
            IFormFile? uploadedFile,
            string? existingPath,
            CancellationToken cancellationToken = default)
        {
            return Task.FromResult((ReturnValid, ErrorToReturn, RecordsToReturn));
        }
    }

    private sealed class FakeHowsoClient : IHowsoClient
    {
        public HowsoHealthResult HealthToReturn { get; set; } = new(true, "Healthy", "Amalgam-Hazard-LK", "MIT", "Service is operational");
        public HowsoAuditResult? AuditToReturn { get; set; }
        public HowsoVerifyResult? VerifyToReturn { get; set; }
        public bool ShouldThrow { get; set; }
        public List<HowsoDatasetRecord>? LastReferenceRecords { get; private set; }
        public string? LastPrompt { get; private set; }
        public string? LastCandidateCompletion { get; private set; }
        public float LastMaxTolerableSurprisal { get; private set; }

        public Task<HowsoHealthResult> CheckHealthAsync(CancellationToken cancellationToken = default)
        {
            if (ShouldThrow) throw new HttpRequestException("Howso offline");
            return Task.FromResult(HealthToReturn);
        }

        public Task<HowsoAuditResult?> AuditDatasetAsync(
            IEnumerable<HowsoDatasetRecord> records,
            float ablationThreshold = 0.15f,
            float anomalyThreshold = 5,
            CancellationToken cancellationToken = default)
        {
            if (ShouldThrow) throw new HttpRequestException("Howso offline");
            return Task.FromResult(AuditToReturn);
        }

        public Task<HowsoVerifyResult?> VerifyCompletionAsync(
            IEnumerable<HowsoDatasetRecord> referenceRecords,
            string prompt,
            string candidateCompletion,
            float maxTolerableSurprisal = 6,
            CancellationToken cancellationToken = default)
        {
            if (ShouldThrow) throw new HttpRequestException("Howso offline");
            LastReferenceRecords = referenceRecords.ToList();
            LastPrompt = prompt;
            LastCandidateCompletion = candidateCompletion;
            LastMaxTolerableSurprisal = maxTolerableSurprisal;
            return Task.FromResult(VerifyToReturn);
        }
    }

    private static IConfiguration CreateConfig(bool studioFallback = true)
    {
        return new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["INFERENCE_SERVICE_URL"] = "http://localhost:8000",
                ["STUDIO_PREVIEW_FALLBACK"] = studioFallback ? "true" : "false"
            })
            .Build();
    }

    [Fact]
    public async Task CompareInferenceAsync_AttachesProvenance_WhenFastApiReturnsResponseAndHowsoVerifies()
    {
        var compareResponse = new InferenceCompareResponse
        {
            JobId = "job-ok",
            BaseModel = "SmolLM2-135M",
            Prompt = "What is the cancellation window?",
            BaseCompletion = "Standard cancellation is 30 days.",
            FineTunedCompletion = "Contracts beyond 30 days are non-refundable. [MSA-SEC8-NONREF]",
            LatencyMs = new LatencyBreakdown { BaseModel = 40.0, FineTuned = 42.0 },
            IsSimulated = false
        };

        var handler = new DelegatingMockHandler(req =>
        {
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(JsonSerializer.Serialize(compareResponse), Encoding.UTF8, "application/json")
            };
        });

        var clientFactory = new MockHttpClientFactory(handler);
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(
                IsConforming: true,
                MeasuredSurprisalNats: 0.85f,
                PredictedTag: "MSA-SEC8-NONREF",
                CandidateTag: "MSA-SEC8-NONREF",
                InfluentialCases: new List<HowsoInfluentialCase>
                {
                    new(1, "MSA-SEC8-NONREF", 0.85f, 95.0f, "What is the policy for cancellation after 45 days?")
                },
                Explanation: "Conforming response with low surprisal."
            )
        };

        var request = new InferenceCompareRequest
        {
            Prompt = "What is the cancellation window?"
        };

        var result = await InferenceEndpoints.CompareInferenceAsync(
            request, jobRepo, datasetService, howsoClient, clientFactory, CreateConfig(), TestCache, NullLoggerFactory.Instance, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceCompareResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.NotNull(okResult.Value.HowsoProvenance);
        Assert.True(okResult.Value.HowsoProvenance.IsConforming);
        Assert.Equal(0.85f, okResult.Value.HowsoProvenance.MeasuredSurprisalNats);
        Assert.Equal("MSA-SEC8-NONREF", okResult.Value.HowsoProvenance.PredictedTag);
        Assert.Single(okResult.Value.HowsoProvenance.InfluentialCases);
    }

    [Fact]
    public async Task CompareInferenceAsync_AttachesProvenance_InStudioPreviewFallbackMode()
    {
        var handler = new DelegatingMockHandler(req => throw new HttpRequestException("FastAPI offline"));
        var clientFactory = new MockHttpClientFactory(handler);
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(
                IsConforming: true,
                MeasuredSurprisalNats: 1.10f,
                PredictedTag: "ACH-HELD-VERIFY",
                CandidateTag: "ACH-HELD-VERIFY",
                InfluentialCases: new List<HowsoInfluentialCase>(),
                Explanation: "Conforming studio preview"
            )
        };

        var request = new InferenceCompareRequest
        {
            Prompt = "Customer has a delayed transfer of $15,000 ACH"
        };

        var result = await InferenceEndpoints.CompareInferenceAsync(
            request, jobRepo, datasetService, howsoClient, clientFactory, CreateConfig(studioFallback: true), TestCache, NullLoggerFactory.Instance, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceCompareResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.True(okResult.Value.IsSimulated);
        Assert.NotNull(okResult.Value.HowsoProvenance);
        Assert.True(okResult.Value.HowsoProvenance.IsConforming);
    }

    [Fact]
    public async Task CompareInferenceAsync_GracefulFallback_WhenHowsoThrowsOrReturnsNull()
    {
        var compareResponse = new InferenceCompareResponse
        {
            JobId = "job-ok",
            BaseModel = "SmolLM2-135M",
            Prompt = "Hello test",
            BaseCompletion = "Base completion",
            FineTunedCompletion = "Fine-tuned completion",
            LatencyMs = new LatencyBreakdown { BaseModel = 35.0, FineTuned = 38.0 }
        };

        var handler = new DelegatingMockHandler(req =>
        {
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(JsonSerializer.Serialize(compareResponse), Encoding.UTF8, "application/json")
            };
        });

        var clientFactory = new MockHttpClientFactory(handler);
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient
        {
            ShouldThrow = true // Simulates Howso service failure
        };

        var request = new InferenceCompareRequest
        {
            Prompt = "Hello test"
        };

        var result = await InferenceEndpoints.CompareInferenceAsync(
            request, jobRepo, datasetService, howsoClient, clientFactory, CreateConfig(), TestCache, NullLoggerFactory.Instance, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceCompareResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.Null(okResult.Value.HowsoProvenance); // Gracefully omitted without failing request
    }

    [Fact]
    public async Task CompareInferenceAsync_UsesJobDatasetRecords_WhenJobIdProvided()
    {
        var jobRepo = new FakeJobRepository();
        var job = new FinetuneJob
        {
            Id = "job-trained-1",
            JobName = "custom-adapter",
            Status = JobStatus.Succeeded,
            BaseModel = "SmolLM2-135M",
            DatasetPath = "datasets/job-trained-1.jsonl",
            DatasetHash = "sha256hash",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        jobRepo.AddJob(job);

        var customRecords = new List<HowsoDatasetRecord>
        {
            new("Custom prompt 1", "Custom completion 1 [TAG-1]"),
            new("Custom prompt 2", "Custom completion 2 [TAG-2]")
        };

        var datasetService = new FakeDatasetService
        {
            ReturnValid = true,
            RecordsToReturn = customRecords
        };

        var handler = new DelegatingMockHandler(req =>
        {
            var response = new InferenceCompareResponse
            {
                JobId = job.Id,
                BaseModel = job.BaseModel,
                Prompt = "Custom prompt 1",
                BaseCompletion = "Base completion 1",
                FineTunedCompletion = "Custom completion 1 [TAG-1]",
                LatencyMs = new LatencyBreakdown { BaseModel = 30.0, FineTuned = 32.0 }
            };
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(JsonSerializer.Serialize(response), Encoding.UTF8, "application/json")
            };
        });

        var clientFactory = new MockHttpClientFactory(handler);
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(true, 0.45f, "TAG-1", "TAG-1", new List<HowsoInfluentialCase>(), "Valid")
        };

        var request = new InferenceCompareRequest
        {
            JobId = "job-trained-1",
            Prompt = "Custom prompt 1"
        };

        var result = await InferenceEndpoints.CompareInferenceAsync(
            request, jobRepo, datasetService, howsoClient, clientFactory, CreateConfig(), TestCache, NullLoggerFactory.Instance, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceCompareResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.NotNull(howsoClient.LastReferenceRecords);
        Assert.Equal(2, howsoClient.LastReferenceRecords.Count);
        Assert.Equal("Custom prompt 1", howsoClient.LastReferenceRecords[0].Prompt);
    }

    [Fact]
    public async Task CompareInferenceAsync_ReturnsNotFound_WhenJobDoesNotExist()
    {
        var clientFactory = new MockHttpClientFactory(new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)));
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient();

        var request = new InferenceCompareRequest
        {
            JobId = "non-existent-job",
            Prompt = "Test prompt"
        };

        var result = await InferenceEndpoints.CompareInferenceAsync(
            request, jobRepo, datasetService, howsoClient, clientFactory, CreateConfig(), TestCache, NullLoggerFactory.Instance, CancellationToken.None);

        var notFoundResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
    }

    [Fact]
    public async Task CompareInferenceAsync_ReturnsBadRequest_WhenJobStatusIsNotSucceeded()
    {
        var clientFactory = new MockHttpClientFactory(new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)));
        var jobRepo = new FakeJobRepository();
        var job = new FinetuneJob
        {
            Id = "job-in-progress",
            JobName = "training-job",
            Status = JobStatus.Training,
            BaseModel = "SmolLM2-135M",
            DatasetPath = "d.jsonl",
            DatasetHash = "h",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        jobRepo.AddJob(job);

        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient();

        var request = new InferenceCompareRequest
        {
            JobId = "job-in-progress",
            Prompt = "Test prompt"
        };

        var result = await InferenceEndpoints.CompareInferenceAsync(
            request, jobRepo, datasetService, howsoClient, clientFactory, CreateConfig(), TestCache, NullLoggerFactory.Instance, CancellationToken.None);

        var badRequestResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestResult.StatusCode);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_ReturnsOk_WhenVerificationSucceeds()
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(
                IsConforming: true,
                MeasuredSurprisalNats: 1.25f,
                PredictedTag: "SEC-NO-ADVISORY",
                CandidateTag: "SEC-NO-ADVISORY",
                InfluentialCases: new List<HowsoInfluentialCase>
                {
                    new(0, "SEC-NO-ADVISORY", 1.25f, 98.0f, "Should I invest my 401k into Bitcoin?")
                },
                Explanation: "Conforming response matching SEC-NO-ADVISORY policy"
            )
        };

        var request = new InferenceVerifyRequest
        {
            Prompt = "Should I invest in crypto?",
            CandidateCompletion = "We do not provide investment advice. [SEC-NO-ADVISORY]"
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceVerifyResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.True(okResult.Value.IsConforming);
        Assert.NotNull(okResult.Value.Provenance);
        Assert.Equal(1.25f, okResult.Value.Provenance.MeasuredSurprisalNats);
        Assert.Contains("Conforming", okResult.Value.Message);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_Returns503_WhenHowsoReturnsNull()
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = null // Howso service failure
        };

        var request = new InferenceVerifyRequest
        {
            Prompt = "Test prompt",
            CandidateCompletion = "Test candidate"
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var problemResult = Assert.IsType<ProblemHttpResult>(result);
        Assert.Equal(StatusCodes.Status503ServiceUnavailable, problemResult.StatusCode);
        Assert.Equal("Howso Arbiter Unavailable", problemResult.ProblemDetails.Title);
    }

    [Theory]
    [InlineData("", "completion")]
    [InlineData("   ", "completion")]
    [InlineData("prompt", "")]
    [InlineData("prompt", "   ")]
    public async Task VerifyProvenanceAsync_ReturnsBadRequest_WhenPromptOrCompletionMissing(string prompt, string completion)
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient();

        var request = new InferenceVerifyRequest
        {
            Prompt = prompt,
            CandidateCompletion = completion
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var badRequestResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestResult.StatusCode);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_ExtractsJobDataset_WhenJobIdProvided()
    {
        var jobRepo = new FakeJobRepository();
        var job = new FinetuneJob
        {
            Id = "job-verify-1",
            JobName = "verify-job",
            Status = JobStatus.Succeeded,
            BaseModel = "SmolLM2-135M",
            DatasetPath = "datasets/job-verify-1.jsonl",
            DatasetHash = "hash123",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        jobRepo.AddJob(job);

        var records = new List<HowsoDatasetRecord>
        {
            new("Question A", "Answer A [TAG-A]"),
            new("Question B", "Answer B [TAG-B]")
        };

        var datasetService = new FakeDatasetService
        {
            ReturnValid = true,
            RecordsToReturn = records
        };

        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(true, 0.5f, "TAG-A", "TAG-A", new List<HowsoInfluentialCase>(), "Ok")
        };

        var request = new InferenceVerifyRequest
        {
            JobId = "job-verify-1",
            Prompt = "Question A",
            CandidateCompletion = "Answer A [TAG-A]"
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceVerifyResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.NotNull(howsoClient.LastReferenceRecords);
        Assert.Equal(2, howsoClient.LastReferenceRecords.Count);
        Assert.Equal("Question A", howsoClient.LastReferenceRecords[0].Prompt);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_UsesDefaultPersonas_WhenNoReferenceRecordsAndNoJobId()
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(true, 0.8f, "TAG", "TAG", new List<HowsoInfluentialCase>(), "Ok")
        };

        var request = new InferenceVerifyRequest
        {
            Prompt = "Customer has a delayed transfer of $15,000 ACH",
            CandidateCompletion = "Advise customer under Reg CC..."
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var okResult = Assert.IsType<Ok<InferenceVerifyResponse>>(result);
        Assert.NotNull(okResult.Value);
        Assert.NotNull(howsoClient.LastReferenceRecords);
        Assert.NotEmpty(howsoClient.LastReferenceRecords);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_RejectsPrompt_WhenExceeding10kCharacters()
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient();

        var request = new InferenceVerifyRequest
        {
            Prompt = new string('A', 10_001),
            CandidateCompletion = "Candidate completion"
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var badRequestResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestResult.StatusCode);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_RejectsReferenceRecords_WhenContainsPiiOrInvalidWeight()
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient();

        var requestWithPii = new InferenceVerifyRequest
        {
            Prompt = "Valid prompt",
            CandidateCompletion = "Valid completion",
            ReferenceRecords = new List<HowsoDatasetRecord>
            {
                new("Customer SSN is 123-45-6789", "Processed", 1.0f)
            }
        };

        var resultPii = await InferenceEndpoints.VerifyProvenanceAsync(
            requestWithPii, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var badRequestPii = Assert.IsAssignableFrom<IStatusCodeHttpResult>(resultPii);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestPii.StatusCode);

        var requestWithInvalidWeight = new InferenceVerifyRequest
        {
            Prompt = "Valid prompt",
            CandidateCompletion = "Valid completion",
            ReferenceRecords = new List<HowsoDatasetRecord>
            {
                new("Valid prompt 1", "Valid completion 1", -0.5f)
            }
        };

        var resultWeight = await InferenceEndpoints.VerifyProvenanceAsync(
            requestWithInvalidWeight, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var badRequestWeight = Assert.IsAssignableFrom<IStatusCodeHttpResult>(resultWeight);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestWeight.StatusCode);

        var oversizedRecords = Enumerable.Range(0, DatasetService.MaxPreviewAuditRecords + 1)
            .Select(i => new HowsoDatasetRecord($"Prompt {i}", $"Completion {i}"))
            .ToList();

        var requestOversized = new InferenceVerifyRequest
        {
            Prompt = "Valid prompt",
            CandidateCompletion = "Valid completion",
            ReferenceRecords = oversizedRecords
        };

        var resultOversized = await InferenceEndpoints.VerifyProvenanceAsync(
            requestOversized, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var badRequestOversized = Assert.IsAssignableFrom<IStatusCodeHttpResult>(resultOversized);
        Assert.Equal(StatusCodes.Status400BadRequest, badRequestOversized.StatusCode);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_CachesExtractedDatasetRecords_AcrossRequests()
    {
        var jobRepo = new FakeJobRepository();
        var job = new FinetuneJob
        {
            Id = "job-cached-1",
            JobName = "cached-job",
            Status = JobStatus.Succeeded,
            BaseModel = "SmolLM2-135M",
            DatasetPath = "datasets/job-cached-1.jsonl",
            DatasetHash = "hash-cached-123",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        jobRepo.AddJob(job);

        var records = new List<HowsoDatasetRecord>
        {
            new("Cached question 1", "Cached answer 1 [TAG-1]"),
            new("Cached question 2", "Cached answer 2 [TAG-2]")
        };

        int extractCalls = 0;
        var trackingDatasetService = new TrackingDatasetService(records, () => extractCalls++);
        var howsoClient = new FakeHowsoClient
        {
            VerifyToReturn = new HowsoVerifyResult(true, 0.5f, "TAG-1", "TAG-1", new List<HowsoInfluentialCase>(), "Ok")
        };

        var isolatedCache = new MemoryCache(new MemoryCacheOptions { SizeLimit = 100_000 });

        var request = new InferenceVerifyRequest
        {
            JobId = "job-cached-1",
            Prompt = "Cached question 1",
            CandidateCompletion = "Cached answer 1 [TAG-1]"
        };

        // First call extracts from dataset
        var result1 = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, trackingDatasetService, howsoClient, isolatedCache, CancellationToken.None);
        Assert.IsType<Ok<InferenceVerifyResponse>>(result1);
        Assert.Equal(1, extractCalls);

        // Second call should hit the IMemoryCache without re-extracting
        var result2 = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, trackingDatasetService, howsoClient, isolatedCache, CancellationToken.None);
        Assert.IsType<Ok<InferenceVerifyResponse>>(result2);
        Assert.Equal(1, extractCalls); // Unchanged! Reused cache!
    }

    [Fact]
    public async Task VerifyProvenanceAsync_ReturnsNotFound_WhenJobDoesNotExist()
    {
        var jobRepo = new FakeJobRepository();
        var datasetService = new FakeDatasetService();
        var howsoClient = new FakeHowsoClient();

        var request = new InferenceVerifyRequest
        {
            JobId = "missing-job-id",
            Prompt = "Test prompt",
            CandidateCompletion = "Test completion"
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var notFoundResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status404NotFound, notFoundResult.StatusCode);
    }

    [Fact]
    public async Task VerifyProvenanceAsync_ReturnsUnprocessableEntity_WhenJobDatasetIsEmptyOrInvalid()
    {
        var jobRepo = new FakeJobRepository();
        var job = new FinetuneJob
        {
            Id = "job-empty-dataset",
            JobName = "empty-job",
            Status = JobStatus.Succeeded,
            BaseModel = "SmolLM2-135M",
            DatasetPath = "datasets/empty.jsonl",
            DatasetHash = "empty-hash",
            HyperparametersJson = "{}",
            CreatedAt = DateTimeOffset.UtcNow,
            UpdatedAt = DateTimeOffset.UtcNow
        };
        jobRepo.AddJob(job);

        var datasetService = new FakeDatasetService
        {
            ReturnValid = true,
            RecordsToReturn = new List<HowsoDatasetRecord>() // Empty
        };
        var howsoClient = new FakeHowsoClient();

        var request = new InferenceVerifyRequest
        {
            JobId = "job-empty-dataset",
            Prompt = "Test prompt",
            CandidateCompletion = "Test completion"
        };

        var result = await InferenceEndpoints.VerifyProvenanceAsync(
            request, jobRepo, datasetService, howsoClient, TestCache, CancellationToken.None);

        var unprocessableResult = Assert.IsAssignableFrom<IStatusCodeHttpResult>(result);
        Assert.Equal(StatusCodes.Status422UnprocessableEntity, unprocessableResult.StatusCode);
    }

    private sealed class TrackingDatasetService : IDatasetService
    {
        private readonly List<HowsoDatasetRecord> _records;
        private readonly Action _onExtract;

        public TrackingDatasetService(List<HowsoDatasetRecord> records, Action onExtract)
        {
            _records = records;
            _onExtract = onExtract;
        }

        public Task<DatasetResult> NormalizeAndStoreAsync(string jobId, IFormFile? uploadedFile, string? existingPath, bool enableHowsoPreflight = false, CancellationToken cancellationToken = default)
            => Task.FromResult(new DatasetResult(true, "p", "h", 1, null));

        public Task<(bool IsValid, string? ErrorMessage, List<HowsoDatasetRecord>? Records)> ValidateAndExtractRecordsAsync(IFormFile? uploadedFile, string? existingPath, CancellationToken cancellationToken = default)
        {
            _onExtract();
            return Task.FromResult((true, (string?)null, (List<HowsoDatasetRecord>?)_records));
        }
    }
}
