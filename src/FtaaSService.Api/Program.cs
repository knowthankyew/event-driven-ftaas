using FtaaSService.Api.Endpoints;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Services.Surprisal;
using FtaaSService.Api.Services.Howso;
using FtaaSService.Api.Storage;

var builder = WebApplication.CreateBuilder(args);

// Configure Options
builder.Services.Configure<RabbitMqConfig>(builder.Configuration.GetSection("RabbitMQ"));

// Register Core Services
builder.Services.AddSingleton<IFtaasTelemetry, FtaasTelemetry>();
builder.Services.AddSingleton<IJobRepository, SqliteJobRepository>();
builder.Services.AddSingleton<SurprisalClient>();
builder.Services.AddSingleton<ISurprisalClient>(sp => sp.GetRequiredService<SurprisalClient>());
builder.Services.AddSingleton<IHowsoClient>(sp => sp.GetRequiredService<SurprisalClient>());
builder.Services.AddSingleton<IDatasetService, DatasetService>();
builder.Services.AddSingleton<IEventPublisher, RabbitMqEventPublisher>();
builder.Services.AddHostedService<JobStatusListener>();
builder.Services.AddHttpClient();
builder.Services.AddMemoryCache(options => { options.SizeLimit = 100_000; });

// Add ProblemDetails & CORS
builder.Services.AddProblemDetails();
var allowedOrigins = builder.Configuration.GetSection("Cors:AllowedOrigins").Get<string[]>() ?? Array.Empty<string>();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(p =>
    {
        if (allowedOrigins.Length > 0)
        {
            p.WithOrigins(allowedOrigins).AllowAnyMethod().AllowAnyHeader();
        }
        else
        {
            p.WithOrigins("http://localhost:5100", "http://127.0.0.1:5100", "http://localhost:5000", "http://127.0.0.1:5000")
             .AllowAnyMethod()
             .AllowAnyHeader();
        }
    });
});

var app = builder.Build();

app.UseCors();
app.UseExceptionHandler();

// Enable static assets for Non-Tech Studio web interface
app.UseDefaultFiles();
app.UseStaticFiles();

// Initialize DB schema
using (var scope = app.Services.CreateScope())
{
    var repo = scope.ServiceProvider.GetRequiredService<IJobRepository>();
    await repo.InitializeAsync();
}

// Map Endpoints
app.MapHealthEndpoints();
app.MapMetricsEndpoints();
app.MapGroup("/api/v1/jobs").MapJobEndpoints();
app.MapGroup("/api/v1/inference").MapInferenceEndpoints();
app.MapGroup("/api/v1/studio").MapStudioEndpoints();
app.MapGroup("/api/v1/telemetry").MapTelemetryEndpoints();

app.Run();

