using System.Text.Json.Serialization;

namespace FtaaSService.Api.Domain;

public sealed record Hyperparameters
{
    [JsonPropertyName("epochs")]
    public int Epochs { get; init; } = 3;

    [JsonPropertyName("batchSize")]
    public int BatchSize { get; init; } = 4;

    [JsonPropertyName("learningRate")]
    public double LearningRate { get; init; } = 0.0002;

    [JsonPropertyName("loraRank")]
    public int LoraRank { get; init; } = 8;

    [JsonPropertyName("loraAlpha")]
    public int LoraAlpha { get; init; } = 32;

    [JsonPropertyName("loraDropout")]
    public double LoraDropout { get; init; } = 0.05;
}
