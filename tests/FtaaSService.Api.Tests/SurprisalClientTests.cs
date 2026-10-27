using System.Net;
using System.Text;
using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;

namespace FtaaSService.Api.Tests;

public class SurprisalClientTests
{
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

    private static IConfiguration CreateConfig(string baseUrl = "http://localhost:5200")
    {
        return new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["SURPRISAL_SERVICE_URL"] = baseUrl
            })
            .Build();
    }

    private static IConfiguration CreateLegacyConfig(string baseUrl = "http://localhost:5200")
    {
        return new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["HOWSO_SERVICE_URL"] = baseUrl
            })
            .Build();
    }

    [Fact]
    public async Task CheckHealthAsync_ReturnsLive_WhenEndpointReturnsHealthy()
    {
        var jsonResponse = "{\"status\": \"Healthy\", \"engine\": \"Independent Reference Information Engine (Hazard et al. Laplace-LK Surprisal)\", \"license\": \"MIT\"}";
        var handler = new DelegatingMockHandler(req =>
        {
            Assert.Equal(HttpMethod.Get, req.Method);
            Assert.EndsWith("/health", req.RequestUri!.ToString());
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(jsonResponse, Encoding.UTF8, "application/json")
            };
        });

        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var result = await client.CheckHealthAsync();

        Assert.True(result.IsLive);
        Assert.Equal("Healthy", result.Status);
        Assert.Equal("Independent Reference Information Engine (Hazard et al. Laplace-LK Surprisal)", result.Engine);
        Assert.Equal("MIT", result.License);
    }

    [Fact]
    public async Task CheckHealthAsync_ResolvesLegacyHowsoServiceUrlConfig()
    {
        var jsonResponse = "{\"status\": \"Healthy\", \"engine\": \"Independent Reference Information Engine\", \"license\": \"MIT\"}";
        var handler = new DelegatingMockHandler(req =>
        {
            Assert.Equal("http://legacy-arbiter:5200/health", req.RequestUri!.ToString());
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(jsonResponse, Encoding.UTF8, "application/json")
            };
        });

        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateLegacyConfig("http://legacy-arbiter:5200"), NullLogger<SurprisalClient>.Instance);

        var result = await client.CheckHealthAsync();
        Assert.True(result.IsLive);
    }

    [Fact]
    public async Task CheckHealthAsync_ReturnsOffline_WhenEndpointThrowsOrFails()
    {
        var handler = new DelegatingMockHandler(_ => throw new HttpRequestException("Connection refused"));
        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var result = await client.CheckHealthAsync();

        Assert.False(result.IsLive);
        Assert.Equal("Offline", result.Status);
        Assert.Contains("unreachable", result.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task AuditDatasetAsync_ReturnsSummaryAndCleanedRecords_WhenEndpointReturnsOk()
    {
        var jsonResponse = """
        {
            "summary": {
                "original_count": 3,
                "cleaned_count": 2,
                "ablated_count": 1,
                "anomalous_count": 0,
                "measured_deviations": { "prompt_len": 4.5 },
                "anomalous_records": [],
                "ablated_records": [
                    { "ablated_index": 1, "merged_into_index": 0, "surprisal_nats": 0.02 }
                ]
            },
            "cleaned_records": [
                { "prompt": "Prompt 1", "completion": "Answer 1", "weight": 2.0 },
                { "prompt": "Prompt 2", "completion": "Answer 2", "weight": 1.0 }
            ]
        }
        """;

        var handler = new DelegatingMockHandler(req =>
        {
            Assert.Equal(HttpMethod.Post, req.Method);
            Assert.EndsWith("/v1/audit/dataset", req.RequestUri!.ToString());
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(jsonResponse, Encoding.UTF8, "application/json")
            };
        });

        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var inputRecords = new List<SurprisalDatasetRecord>
        {
            new("Prompt 1", "Answer 1"),
            new("Prompt 1 duplicate", "Answer 1"),
            new("Prompt 2", "Answer 2")
        };

        var result = await client.AuditDatasetAsync(inputRecords);

        Assert.NotNull(result);
        Assert.Equal(3, result.Summary.OriginalCount);
        Assert.Equal(2, result.Summary.CleanedCount);
        Assert.Equal(1, result.Summary.AblatedCount);
        Assert.Equal(0, result.Summary.AnomalousCount);
        Assert.Equal(2, result.CleanedRecords.Count);
        Assert.Equal(2.0f, result.CleanedRecords[0].Weight);
    }

    [Fact]
    public async Task AuditDatasetAsync_ReturnsNull_WhenEmptyRecordsPassed()
    {
        var invoked = false;
        var handler = new DelegatingMockHandler(_ =>
        {
            invoked = true;
            return new HttpResponseMessage(HttpStatusCode.OK);
        });

        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var result = await client.AuditDatasetAsync(Array.Empty<SurprisalDatasetRecord>());

        Assert.Null(result);
        Assert.False(invoked);
    }

    [Fact]
    public async Task AuditDatasetAsync_GracefullyReturnsNull_WhenEndpointFails()
    {
        var handler = new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.InternalServerError)
        {
            Content = new StringContent("Server error", Encoding.UTF8, "text/plain")
        });

        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var result = await client.AuditDatasetAsync(new[] { new SurprisalDatasetRecord("Prompt", "Completion") });

        Assert.Null(result);
    }

    [Fact]
    public async Task VerifyCompletionAsync_ReturnsConformingPrediction_WhenEndpointReturnsOk()
    {
        var jsonResponse = """
        {
            "is_conforming": true,
            "measured_surprisal_nats": 1.25,
            "predicted_tag": "ACH-HELD-VERIFY",
            "candidate_tag": "ACH-HELD-VERIFY",
            "influential_cases": [
                {
                    "case_index": 0,
                    "tag": "ACH-HELD-VERIFY",
                    "surprisal_nats": 0.5,
                    "influence_weight_percent": 85.0,
                    "source_prompt": "Customer wire delayed"
                }
            ],
            "explanation": "Conforms with ACH clearance policy."
        }
        """;

        var handler = new DelegatingMockHandler(req =>
        {
            Assert.Equal(HttpMethod.Post, req.Method);
            Assert.EndsWith("/v1/verify/completion", req.RequestUri!.ToString());
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(jsonResponse, Encoding.UTF8, "application/json")
            };
        });

        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var refRecords = new[] { new SurprisalDatasetRecord("Prompt", "Completion [ACH-HELD-VERIFY]") };
        var result = await client.VerifyCompletionAsync(refRecords, "Prompt", "Completion [ACH-HELD-VERIFY]");

        Assert.NotNull(result);
        Assert.True(result.IsConforming);
        Assert.Equal("ACH-HELD-VERIFY", result.PredictedTag);
        Assert.Single(result.InfluentialCases);
        Assert.Equal(85.0f, result.InfluentialCases[0].InfluenceWeightPercent);
    }

    [Fact]
    public async Task VerifyCompletionAsync_ReturnsNull_WhenEndpointFails()
    {
        var handler = new DelegatingMockHandler(_ => throw new HttpRequestException("Network failure"));
        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var refRecords = new[] { new SurprisalDatasetRecord("Prompt", "Completion") };
        var result = await client.VerifyCompletionAsync(refRecords, "Prompt", "Completion");

        Assert.Null(result);
    }

    [Fact]
    public async Task CheckHealthAsync_RethrowsOperationCanceledException_WhenCallerCancellationRequested()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var handler = new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            client.CheckHealthAsync(cancellationToken: cts.Token));
    }

    [Fact]
    public async Task AuditDatasetAsync_RethrowsOperationCanceledException_WhenCallerCancellationRequested()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var handler = new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var refRecords = new[] { new SurprisalDatasetRecord("Prompt", "Completion") };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            client.AuditDatasetAsync(refRecords, cancellationToken: cts.Token));
    }

    [Fact]
    public async Task VerifyCompletionAsync_RethrowsOperationCanceledException_WhenCallerCancellationRequested()
    {
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        var handler = new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        var factory = new MockHttpClientFactory(handler);
        var client = new SurprisalClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        var refRecords = new[] { new SurprisalDatasetRecord("Prompt", "Completion") };
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            client.VerifyCompletionAsync(refRecords, "Prompt", "Completion", cancellationToken: cts.Token));
    }

    [Fact]
    public void BackwardCompatibility_HowsoClientAlias_InstantiatesAndImplementsInterfaces()
    {
        var handler = new DelegatingMockHandler(_ => new HttpResponseMessage(HttpStatusCode.OK));
        var factory = new MockHttpClientFactory(handler);
        var client = new HowsoClient(factory, CreateConfig(), NullLogger<SurprisalClient>.Instance);

        Assert.IsAssignableFrom<ISurprisalClient>(client);
        Assert.IsAssignableFrom<IHowsoClient>(client);
    }
}
