using FtaaSService.Api.Domain;

namespace FtaaSService.Api.Tests;

public class ModelCatalogTests
{
    [Fact]
    public void SupportedModels_ContainsSmolLM2AndGemma2B()
    {
        Assert.Contains(SupportedModels.SmolLM2, SupportedModels.All);
        Assert.Contains(SupportedModels.Gemma2_2B_IT, SupportedModels.All);
        Assert.Equal(2, SupportedModels.All.Count);
    }

    [Fact]
    public void SupportedModelsCatalog_ExposesExpectedMetadataForBothModels()
    {
        var catalog = SupportedModels.Catalog;
        Assert.Equal(2, catalog.Count);

        var smol = catalog.FirstOrDefault(m => m.ModelId == SupportedModels.SmolLM2);
        Assert.NotNull(smol);
        Assert.Equal("135M", smol.ParameterCount);
        Assert.Equal(2048, smol.ContextLength);
        Assert.Equal(0.5, smol.MinGpuVramGb);
        Assert.False(smol.RequiresHfAuth);
        Assert.Null(smol.HardwareDisclaimer);

        var gemma = catalog.FirstOrDefault(m => m.ModelId == SupportedModels.Gemma2_2B_IT);
        Assert.NotNull(gemma);
        Assert.Equal("2B", gemma.ParameterCount);
        Assert.Equal(8192, gemma.ContextLength);
        Assert.Equal(8.0, gemma.MinGpuVramGb);
        Assert.True(gemma.RequiresHfAuth);
        Assert.NotNull(gemma.HardwareDisclaimer);
        Assert.Contains("HF_TOKEN", gemma.HardwareDisclaimer);
    }

    [Theory]
    [InlineData("HuggingFaceTB/SmolLM2-135M", true)]
    [InlineData("google/gemma-2-2b-it", true)]
    [InlineData("unsupported/arbitrary-model", false)]
    [InlineData("openai/gpt-4o", false)]
    [InlineData("", false)]
    public void ModelValidation_CorrectlyIdentifiesAllowedModels(string modelId, bool expectedAllowed)
    {
        bool isSupported = SupportedModels.All.Contains(modelId);
        Assert.Equal(expectedAllowed, isSupported);
    }
}
