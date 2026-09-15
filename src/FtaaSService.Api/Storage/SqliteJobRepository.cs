using System.Data;
using System.Globalization;
using Dapper;
using FtaaSService.Api.Domain;
using Microsoft.Data.Sqlite;

namespace FtaaSService.Api.Storage;

public sealed class SqliteJobRepository : IJobRepository
{
    private readonly string _connectionString;
    private readonly ILogger<SqliteJobRepository> _logger;

    public SqliteJobRepository(IConfiguration configuration, ILogger<SqliteJobRepository> logger)
    {
        _logger = logger;
        var dataRoot = configuration.GetValue<string>("DATA_ROOT") 
                       ?? Path.Combine(Directory.GetCurrentDirectory(), "data");
        
        var storageDir = Path.Combine(dataRoot, "storage");
        Directory.CreateDirectory(storageDir);

        var dbPath = Path.Combine(storageDir, "ftaas.db");
        _connectionString = $"Data Source={dbPath}";
    }

    private SqliteConnection CreateConnection() => new(_connectionString);

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        using var connection = CreateConnection();
        await connection.OpenAsync(cancellationToken);

        const string sql = """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_name TEXT NOT NULL,
                status TEXT NOT NULL,
                base_model TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                hyperparameters_json TEXT NOT NULL,
                progress_percent REAL NOT NULL DEFAULT 0,
                current_step INTEGER NOT NULL DEFAULT 0,
                total_steps INTEGER NOT NULL DEFAULT 0,
                current_loss REAL,
                mlflow_experiment_id TEXT,
                mlflow_run_id TEXT,
                adapter_path TEXT,
                error_message TEXT,
                sequence_number INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
        """;

        await connection.ExecuteAsync(sql);
        _logger.LogInformation("Job database initialized at {ConnectionString}", _connectionString);
    }

    public async Task<FinetuneJob> CreateAsync(FinetuneJob job, CancellationToken cancellationToken = default)
    {
        using var connection = CreateConnection();
        await connection.OpenAsync(cancellationToken);

        const string sql = """
            INSERT INTO jobs (
                id, job_name, status, base_model, dataset_path, dataset_hash,
                hyperparameters_json, progress_percent, current_step, total_steps,
                current_loss, mlflow_experiment_id, mlflow_run_id, adapter_path,
                error_message, sequence_number, created_at, started_at, finished_at, updated_at
            ) VALUES (
                @Id, @JobName, @Status, @BaseModel, @DatasetPath, @DatasetHash,
                @HyperparametersJson, @ProgressPercent, @CurrentStep, @TotalSteps,
                @CurrentLoss, @MlflowExperimentId, @MlflowRunId, @AdapterPath,
                @ErrorMessage, @SequenceNumber, @CreatedAt, @StartedAt, @FinishedAt, @UpdatedAt
            );
        """;

        var parameters = new
        {
            job.Id,
            job.JobName,
            Status = job.Status.ToString(),
            job.BaseModel,
            job.DatasetPath,
            job.DatasetHash,
            job.HyperparametersJson,
            job.ProgressPercent,
            job.CurrentStep,
            job.TotalSteps,
            job.CurrentLoss,
            job.MlflowExperimentId,
            job.MlflowRunId,
            job.AdapterPath,
            job.ErrorMessage,
            job.SequenceNumber,
            CreatedAt = job.CreatedAt.ToString("O"),
            StartedAt = job.StartedAt?.ToString("O"),
            FinishedAt = job.FinishedAt?.ToString("O"),
            UpdatedAt = job.UpdatedAt.ToString("O")
        };

        await connection.ExecuteAsync(sql, parameters);
        return job;
    }

    public async Task<FinetuneJob?> GetByIdAsync(string id, CancellationToken cancellationToken = default)
    {
        using var connection = CreateConnection();
        await connection.OpenAsync(cancellationToken);

        const string sql = "SELECT * FROM jobs WHERE id = @Id;";
        var row = await connection.QuerySingleOrDefaultAsync<JobDbRecord>(sql, new { Id = id });

        return row is null ? null : MapToEntity(row);
    }

    public async Task<IReadOnlyList<FinetuneJob>> ListAsync(int limit = 50, CancellationToken cancellationToken = default)
    {
        using var connection = CreateConnection();
        await connection.OpenAsync(cancellationToken);

        const string sql = "SELECT * FROM jobs ORDER BY created_at DESC LIMIT @Limit;";
        var rows = await connection.QueryAsync<JobDbRecord>(sql, new { Limit = limit });

        return rows.Select(MapToEntity).ToList();
    }

    public async Task<bool> UpdateStatusIdempotentAsync(JobUpdatedEvent update, CancellationToken cancellationToken = default)
    {
        using var connection = CreateConnection();
        await connection.OpenAsync(cancellationToken);

        // Fetch current status and sequence
        var current = await connection.QuerySingleOrDefaultAsync<JobDbRecord>(
            "SELECT status, sequence_number, updated_at FROM jobs WHERE id = @JobId;",
            new { update.JobId }
        );

        if (current is null)
        {
            _logger.LogWarning("Status update rejected: Job {JobId} not found.", update.JobId);
            return false;
        }

        // Idempotency: Ignore updates if already in terminal state, unless this is the terminal transition itself
        if (current.status is "Succeeded" or "Failed")
        {
            _logger.LogInformation("Job {JobId} is already in terminal state {Status}. Discarding update.", 
                update.JobId, current.status);
            return false;
        }

        // Idempotency: Reject stale out-of-order sequence updates
        if (update.SequenceNumber > 0 && update.SequenceNumber < current.sequence_number)
        {
            _logger.LogInformation("Job {JobId} received out-of-order sequence {IncomingSeq} < {CurrentSeq}. Discarding.",
                update.JobId, update.SequenceNumber, current.sequence_number);
            return false;
        }

        const string sql = """
            UPDATE jobs SET
                status = @Status,
                progress_percent = @ProgressPercent,
                current_step = @CurrentStep,
                total_steps = @TotalSteps,
                current_loss = COALESCE(@CurrentLoss, current_loss),
                mlflow_experiment_id = COALESCE(@MlflowExperimentId, mlflow_experiment_id),
                mlflow_run_id = COALESCE(@MlflowRunId, mlflow_run_id),
                adapter_path = COALESCE(@AdapterPath, adapter_path),
                error_message = COALESCE(@ErrorMessage, error_message),
                sequence_number = MAX(sequence_number, @SequenceNumber),
                started_at = COALESCE(@StartedAt, started_at),
                finished_at = COALESCE(@FinishedAt, finished_at),
                updated_at = @UpdatedAt
            WHERE id = @JobId
              AND status NOT IN ('Succeeded', 'Failed')
              AND sequence_number <= @SequenceNumber;
        """;

        var parameters = new
        {
            update.JobId,
            update.Status,
            update.ProgressPercent,
            update.CurrentStep,
            update.TotalSteps,
            update.CurrentLoss,
            update.MlflowExperimentId,
            update.MlflowRunId,
            update.AdapterPath,
            update.ErrorMessage,
            update.SequenceNumber,
            StartedAt = update.StartedAt?.ToString("O"),
            FinishedAt = update.FinishedAt?.ToString("O"),
            UpdatedAt = update.UpdatedAt.ToString("O")
        };

        var affected = await connection.ExecuteAsync(sql, parameters);
        return affected > 0;
    }

    private static FinetuneJob MapToEntity(JobDbRecord r)
    {
        return new FinetuneJob
        {
            Id = r.id,
            JobName = r.job_name,
            Status = Enum.Parse<JobStatus>(r.status, ignoreCase: true),
            BaseModel = r.base_model,
            DatasetPath = r.dataset_path,
            DatasetHash = r.dataset_hash,
            HyperparametersJson = r.hyperparameters_json,
            ProgressPercent = r.progress_percent,
            CurrentStep = (int)r.current_step,
            TotalSteps = (int)r.total_steps,
            CurrentLoss = r.current_loss,
            MlflowExperimentId = r.mlflow_experiment_id,
            MlflowRunId = r.mlflow_run_id,
            AdapterPath = r.adapter_path,
            ErrorMessage = r.error_message,
            SequenceNumber = r.sequence_number,
            CreatedAt = DateTimeOffset.Parse(r.created_at, CultureInfo.InvariantCulture),
            StartedAt = r.started_at is not null ? DateTimeOffset.Parse(r.started_at, CultureInfo.InvariantCulture) : null,
            FinishedAt = r.finished_at is not null ? DateTimeOffset.Parse(r.finished_at, CultureInfo.InvariantCulture) : null,
            UpdatedAt = DateTimeOffset.Parse(r.updated_at, CultureInfo.InvariantCulture)
        };
    }

    public sealed class JobDbRecord
    {
        public string id { get; set; } = "";
        public string job_name { get; set; } = "";
        public string status { get; set; } = "";
        public string base_model { get; set; } = "";
        public string dataset_path { get; set; } = "";
        public string dataset_hash { get; set; } = "";
        public string hyperparameters_json { get; set; } = "";
        public double progress_percent { get; set; }
        public long current_step { get; set; }
        public long total_steps { get; set; }
        public double? current_loss { get; set; }
        public string? mlflow_experiment_id { get; set; }
        public string? mlflow_run_id { get; set; }
        public string? adapter_path { get; set; }
        public string? error_message { get; set; }
        public long sequence_number { get; set; }
        public string created_at { get; set; } = "";
        public string? started_at { get; set; }
        public string? finished_at { get; set; }
        public string updated_at { get; set; } = "";
    }
}
