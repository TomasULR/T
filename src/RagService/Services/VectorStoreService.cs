using System.Text.Json;
using Microsoft.SemanticKernel.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

#pragma warning disable SKEXP0070
#pragma warning disable CS0618

namespace RagService.Services;

public class VectorStoreService
{
    private readonly QdrantClient _qdrant;
    private readonly ITextEmbeddingGenerationService _embeddingService;
    private const string CollectionName = "documents";
    private const int VectorSize = 768;

    public VectorStoreService(QdrantClient qdrant, ITextEmbeddingGenerationService embeddingService)
    {
        _qdrant = qdrant;
        _embeddingService = embeddingService;
    }

    public async Task InitializeAsync()
    {
        var collections = await _qdrant.ListCollectionsAsync();
        if (collections.All(c => c != CollectionName))
        {
            await _qdrant.CreateCollectionAsync(CollectionName, new VectorParams
            {
                Size = VectorSize,
                Distance = Distance.Cosine
            });
        }
    }

    public async Task StoreChunksAsync(List<string> chunks, string sourceName)
    {
        var points = new List<PointStruct>();

        foreach (var chunk in chunks)
        {
            var embedding = await _embeddingService.GenerateEmbeddingAsync(chunk);
            var point = new PointStruct
            {
                Id = Guid.NewGuid(),
                Vectors = embedding.ToArray(),
                Payload =
                {
                    ["content"] = chunk,
                    ["source"] = sourceName
                }
            };
            points.Add(point);
        }

        await _qdrant.UpsertAsync(CollectionName, points);
    }

    public async Task<List<(string Content, string Source)>> SearchAsync(string query, int topK = 5)
    {
        var queryEmbedding = await _embeddingService.GenerateEmbeddingAsync(query);

        var results = await _qdrant.SearchAsync(CollectionName, queryEmbedding.ToArray(), limit: (ulong)topK);

        return results.Select(r => (
            Content: r.Payload["content"].StringValue,
            Source: r.Payload["source"].StringValue
        )).ToList();
    }
}
