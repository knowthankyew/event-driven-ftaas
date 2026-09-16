using FtaaSService.Api.Domain;
using FtaaSService.Api.Storage;
using Microsoft.AspNetCore.Mvc;

namespace FtaaSService.Api.Endpoints;

public record StudioPersona(
    string Id,
    string Name,
    string Department,
    string Description,
    string AdapterSize,
    string DefaultJobName,
    List<string> ComplianceRules,
    List<StudioSamplePrompt> SamplePrompts,
    List<DatasetPair> SeedPairs
);

public record StudioSamplePrompt(
    string Title,
    string Prompt,
    string ExpectedBaseIssue,
    string ExpectedTunedBenefit
);

public record DatasetPair(
    string Prompt,
    string Completion
);

public static class StudioEndpoints
{
    public static RouteGroupBuilder MapStudioEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/personas", GetPersonas);
        group.MapGet("/overview", GetOverviewAsync);
        group.MapGet("/engine-health", CheckEngineHealthAsync);
        return group;
    }

    private static IResult GetPersonas()
    {
        var personas = new List<StudioPersona>
        {
            new(
                Id: "fintech-compliance",
                Name: "Fintech Support & Compliance Copilot",
                Department: "Risk, Support & Operations",
                Description: "Enforces Reg CC ACH hold limits, BSA/AML verification, non-advisory SEC disclaimers, and mandatory FDIC insurance notices.",
                AdapterSize: "1.8 MB",
                DefaultJobName: "fintech-compliance-v1",
                ComplianceRules: new List<string>
                {
                    "Mandatory FDIC deposit status disclaimer",
                    "Reg CC 3-5 business day ACH clearance notification for >$10K",
                    "Strict non-advisory disclaimer for crypto/stocks (FINRA Rule 2210)",
                    "BSA/AML Enhanced Due Diligence (EDD) ticketing tags"
                },
                SamplePrompts: new List<StudioSamplePrompt>
                {
                    new(
                        Title: "ACH Deposit Clearance Delay",
                        Prompt: "Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise them regarding clearance and compliance?",
                        ExpectedBaseIssue: "Generic apology without mentioning Reg CC, FinCEN thresholds, or required FDIC deposit disclaimers.",
                        ExpectedTunedBenefit: "Cites Reg CC $10,000 threshold, tags [ACH-HELD-VERIFY], and embeds mandatory FDIC disclaimer."
                    ),
                    new(
                        Title: "Investment/Crypto Advice Request",
                        Prompt: "Customer ticket: User is asking for advice on whether they should purchase Ethereum or Bitcoin right now.",
                        ExpectedBaseIssue: "Provides general market opinions without statutory non-advisory disclosures.",
                        ExpectedTunedBenefit: "Triggers strict FINRA Rule 2210 disclaimer: 'We do not offer financial or investment advice...' and tags [SEC-NO-ADVISORY]."
                    ),
                    new(
                        Title: "KYC Wire Escrow Flag",
                        Prompt: "Customer ticket: Premium client asks why their wire transfer was flagged for extra KYC document submission.",
                        ExpectedBaseIssue: "Vague response regarding internal security without statutory legal context.",
                        ExpectedTunedBenefit: "Explains BSA/AML Section 314(b) protocol, security vault upload steps, and 4-hour SLA with tag [BSA-EDD-SECURE]."
                    )
                },
                SeedPairs: new List<DatasetPair>
                {
                    new("Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise them regarding clearance and compliance?", "Advise the customer: Under Reg CC and internal ACH policy, external transfers exceeding $10,000 are subject to standard 3-5 business day secondary verification. Reference Ticket Tag: [ACH-HELD-VERIFY]. Mandatory compliance notice: 'Funds are held in accordance with Federal Reserve Regulation CC and FinCEN transaction monitoring guidelines. FDIC insurance coverage applies once funds are credited to your deposit account.'"),
                    new("Customer ticket: User is asking for advice on whether they should purchase Ethereum or Bitcoin right now?", "STRICT REGULATORY DISCLAIMER: We are an execution platform and cannot provide investment, legal, or tax advice. Response: 'We do not offer financial or investment advice. Cryptocurrencies involve substantial market risk and volatility. Please consult a licensed financial advisor before making trading decisions.' Tag: [SEC-NO-ADVISORY]."),
                    new("Customer ticket: Premium client asks why their wire transfer was flagged for extra KYC document submission.", "Advise client: 'Your wire was selected for routine Enhanced Due Diligence (EDD) under BSA/AML Section 314(b) compliance protocols to protect account security. Once passport and proof-of-address documents are uploaded to the secure vault, review completes within 4 business hours.' Tag: [BSA-EDD-SECURE].")
                }
            ),
            new(
                Id: "saas-support-slang",
                Name: "Enterprise SaaS Customer Ops",
                Department: "Customer Success & Engineering Ops",
                Description: "Understands internal enterprise terminology (ACV, NRR, Sev-1 SLAs, RBAC roles) and strictly enforces MSA cancellation terms.",
                AdapterSize: "1.8 MB",
                DefaultJobName: "saas-ops-v1",
                ComplianceRules: new List<string>
                {
                    "Section 8.2 MSA 30-day non-refundable policy citation",
                    "Sev-1 incident response timeline (15-min SLA, 30-min updates)",
                    "GDPR Article 17 statutory retention exception citation (7-year record retention)",
                    "Ticket tagging for automated routing ([MSA-SEC8-NONREF], [INCIDENT-SEV1-ESCALATE])"
                },
                SamplePrompts: new List<StudioSamplePrompt>
                {
                    new(
                        Title: "Contract Cancellation Past 30 Days",
                        Prompt: "Customer ticket: Customer requests immediate cancellation of enterprise contract after 45 days, citing low team adoption. What is our contractual policy?",
                        ExpectedBaseIssue: "Suggests generic refund or apologies without enforcing MSA legal terms.",
                        ExpectedTunedBenefit: "Cites Section 8.2 of Enterprise MSA, confirms non-refundable policy, tags [MSA-SEC8-NONREF], and offers CS-REENGAGE."
                    ),
                    new(
                        Title: "Sev-1 SSO Outage Escalation",
                        Prompt: "Customer ticket: Enterprise customer reporting a Sev-1 outage on European single sign-on (SSO).",
                        ExpectedBaseIssue: "Treats critical outage like a standard support request.",
                        ExpectedTunedBenefit: "Applies 15-minute engineering SLA, alerts Identity Team via PagerDuty, posts trust notice, tags [INCIDENT-SEV1-ESCALATE]."
                    ),
                    new(
                        Title: "GDPR 'Right to be Forgotten' Request",
                        Prompt: "Customer ticket: Client requests that we delete all transaction logs immediately under GDPR Right to be Forgotten.",
                        ExpectedBaseIssue: "Promises full deletion without checking statutory financial audit retention laws.",
                        ExpectedTunedBenefit: "Correctly identifies GDPR Article 17 exception for statutory 7-year anti-money laundering audit ledgers."
                    )
                },
                SeedPairs: new List<DatasetPair>
                {
                    new("Customer ticket: Customer requests immediate cancellation of enterprise contract after 45 days, citing low team adoption. What is our contractual policy?", "Internal Policy Response: Under Section 8.2 of Enterprise SaaS Master Services Agreement, the standard cancellation window is strictly 30 calendar days from provision date. Contracts beyond 30 days are non-refundable for the remaining annual term. Tag: [MSA-SEC8-NONREF]. Recommended escalation: Offer dedicated Customer Success review [CS-REENGAGE] or contract seat reallocation under Addendum B."),
                    new("Customer ticket: Enterprise customer reporting a Sev-1 outage on European single sign-on (SSO).", "Incident Response: Sev-1 SLA requires engineering response within 15 minutes and updates every 30 minutes. Immediately escalate to On-Call Identity Team via PagerDuty [#incident-sev1-idp]. Post public status notice to trust.domain.com under Incident Classification P0-SSO-EMEA. Tag: [INCIDENT-SEV1-ESCALATE].")
                }
            ),
            new(
                Id: "financial-sentiment",
                Name: "Financial Earnings & SEC Analysis",
                Department: "Investor Relations & Equity Research",
                Description: "Analyzes quarterly corporate earnings snippets, extracting structured metrics (EBITDA, bps, margins) and classifying exact sentiment.",
                AdapterSize: "1.8 MB",
                DefaultJobName: "sentiment-analyst-v1",
                ComplianceRules: new List<string>
                {
                    "Standardized triple-field output: SENTIMENT | METRICS | SUMMARY",
                    "Precise basis point (bps) and percentage YoY extraction",
                    "Objective financial tone without colloquial fluff"
                },
                SamplePrompts: new List<StudioSamplePrompt>
                {
                    new(
                        Title: "Gross Margin & Freight Tailwinds",
                        Prompt: "Analyze earnings report snippet: Q3 gross margin expanded 340 bps YoY to 43.1%, driven by favorable product mix and lower ocean freight costs. Operating income rose 18%.",
                        ExpectedBaseIssue: "Unstructured freeform text summary with inconsistent metric identification.",
                        ExpectedTunedBenefit: "Standardized format: SENTIMENT: Positive | METRICS: Gross Margin +340bps (43.1%), Operating Income +18% | SUMMARY: ..."
                    ),
                    new(
                        Title: "Operating Expense & Margin Compression",
                        Prompt: "Analyze earnings report snippet: Operating expenses surged 42% due to aggressive R&D hiring, outpacing 9% revenue growth and compressing operating margin to negative 6.2%.",
                        ExpectedBaseIssue: "Misses negative sentiment severity and fails structured extraction.",
                        ExpectedTunedBenefit: "SENTIMENT: Negative | METRICS: OpEx +42%, Revenue +9%, Operating Margin -6.2% | SUMMARY: ..."
                    )
                },
                SeedPairs: new List<DatasetPair>
                {
                    new("Analyze earnings report snippet: Q3 gross margin expanded 340 bps YoY to 43.1%, driven by favorable product mix and lower ocean freight costs. Operating income rose 18%.", "SENTIMENT: Positive | METRICS: Gross Margin +340bps (43.1%), Operating Income +18% | SUMMARY: Strong margin expansion propelled by freight tailwinds and product mix."),
                    new("Analyze earnings report snippet: Operating expenses surged 42% due to aggressive R&D hiring, outpacing 9% revenue growth and compressing operating margin to negative 6.2%.", "SENTIMENT: Negative | METRICS: OpEx +42%, Revenue +9%, Operating Margin -6.2% | SUMMARY: Uncontrolled expense expansion severely pressured operating profitability.")
                }
            )
        };

        return Results.Ok(personas);
    }

    private static async Task<IResult> GetOverviewAsync(
        [FromServices] IJobRepository jobRepository,
        [FromServices] IConfiguration configuration,
        CancellationToken cancellationToken)
    {
        var jobs = await jobRepository.ListAsync(100, cancellationToken);
        var succeeded = jobs.Count(j => j.Status == JobStatus.Succeeded);
        var training = jobs.Count(j => j.Status == JobStatus.Training);
        var queued = jobs.Count(j => j.Status == JobStatus.Queued);

        return Results.Ok(new
        {
            totalJobs = jobs.Count,
            succeededJobs = succeeded,
            trainingJobs = training,
            queuedJobs = queued,
            baseModel = "HuggingFaceTB/SmolLM2-135M",
            adapterFootprint = "~1.8 MB",
            architecture = new
            {
                controlPlane = ".NET 10 Minimal API",
                broker = "RabbitMQ (AMQP)",
                worker = "Python 3.12 / PyTorch MPS / LoRA",
                telemetry = "MLflow Tracking Server",
                inference = "FastAPI Dynamic Mount"
            }
        });
    }

    private static async Task<IResult> CheckEngineHealthAsync(
        [FromServices] IHttpClientFactory httpClientFactory,
        [FromServices] IConfiguration configuration,
        CancellationToken cancellationToken)
    {
        var inferenceUrl = configuration.GetValue<string>("INFERENCE_SERVICE_URL") ?? configuration["Inference:BaseUrl"] ?? "http://localhost:8000";
        var client = httpClientFactory.CreateClient();
        client.Timeout = TimeSpan.FromMilliseconds(3500);

        try
        {
            var response = await client.GetAsync($"{inferenceUrl.TrimEnd('/')}/healthz", cancellationToken);
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadFromJsonAsync<System.Text.Json.JsonElement>(cancellationToken: cancellationToken);
                bool isLoaded = content.TryGetProperty("baseModelLoaded", out var loaded) && loaded.GetBoolean();
                string device = content.TryGetProperty("device", out var d) ? d.GetString() ?? "unknown" : "unknown";
                string baseModel = content.TryGetProperty("baseModel", out var bm) ? bm.GetString() ?? "" : "";
                
                return Results.Ok(new
                {
                    isLive = isLoaded,
                    status = isLoaded ? "Ready" : "WarmingUp",
                    device,
                    baseModel,
                    message = isLoaded ? "Live on-device PyTorch model loaded" : "Base model is pre-warming in memory"
                });
            }

            return Results.Ok(new
            {
                isLive = false,
                status = "Unhealthy",
                message = $"Inference engine returned HTTP {response.StatusCode}"
            });
        }
        catch
        {
            return Results.Ok(new
            {
                isLive = false,
                status = "Offline",
                message = "Inference engine (:8000) unreachable. Running in Interactive Preview Mode."
            });
        }
    }
}

