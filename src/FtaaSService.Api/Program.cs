using FtaaSService.Api.Endpoints;
using FtaaSService.Api.Messaging;
using FtaaSService.Api.Services;
using FtaaSService.Api.Storage;

var builder = WebApplication.CreateBuilder(args);

// Configure Options
builder.Services.Configure<RabbitMqConfig>(builder.Configuration.GetSection("RabbitMQ"));

// Register Core Services
builder.Services.AddSingleton<IJobRepository, SqliteJobRepository>();
builder.Services.AddSingleton<IDatasetService, DatasetService>();
builder.Services.AddSingleton<IEventPublisher, RabbitMqEventPublisher>();
builder.Services.AddHostedService<JobStatusListener>();
builder.Services.AddHttpClient();

// Add ProblemDetails & CORS
builder.Services.AddProblemDetails();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(p => p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
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
app.MapGroup("/api/v1/jobs").MapJobEndpoints();
app.MapGroup("/api/v1/inference").MapInferenceEndpoints();
app.MapGroup("/api/v1/studio").MapStudioEndpoints();

app.Run();

