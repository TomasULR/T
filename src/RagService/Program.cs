using Microsoft.SemanticKernel;
using Qdrant.Client;
using RagService.Models;
using RagService.Services;

#pragma warning disable SKEXP0070

var builder = WebApplication.CreateBuilder(args);

var ollamaUrl = builder.Configuration["Ollama:Url"] ?? "http://ollama:11434";
var qdrantHost = builder.Configuration["Qdrant:Host"] ?? "qdrant";
var qdrantPort = int.Parse(builder.Configuration["Qdrant:GrpcPort"] ?? "6334");
var chatModel = builder.Configuration["Ollama:ChatModel"] ?? "gemma2:9b-instruct-q3_K_M";
var embeddingModel = builder.Configuration["Ollama:EmbeddingModel"] ?? "nomic-embed-text";
var documentsPath = builder.Configuration["Documents:Path"] ?? "/app/documents";
var fineTuneUrl = builder.Configuration["FineTuneService:Url"] ?? "http://finetune-service:8090";
var finetunedModel = builder.Configuration["Ollama:FinetunedModel"] ?? "gemma2-finetuned";

builder.Services.AddKernel()
    .AddOllamaChatCompletion(chatModel, new Uri(ollamaUrl))
    .AddOllamaTextEmbeddingGeneration(embeddingModel, new Uri(ollamaUrl));

builder.Services.AddSingleton(_ => new QdrantClient(qdrantHost, qdrantPort));
builder.Services.AddSingleton<DocumentParser>();
builder.Services.AddSingleton<TextChunker>();
builder.Services.AddSingleton<VectorStoreService>();
builder.Services.AddSingleton<RagChatService>();
builder.Services.AddSingleton(_ =>
{
    var http = new HttpClient { BaseAddress = new Uri(ollamaUrl), Timeout = TimeSpan.FromMinutes(5) };
    return new FineTunedChatService(http, finetunedModel);
});

builder.Services.AddHttpClient("FineTuneService", client =>
{
    client.BaseAddress = new Uri(fineTuneUrl);
    client.Timeout = TimeSpan.FromMinutes(10);
});

builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

var app = builder.Build();
app.UseCors();

// Initialize vector store
var vectorStore = app.Services.GetRequiredService<VectorStoreService>();
await vectorStore.InitializeAsync();

// Chat endpoint
app.MapPost("/api/chat", async (ChatRequest request, RagChatService chatService) =>
{
    var response = await chatService.AskAsync(request.Message);
    return Results.Ok(response);
});

// Streaming chat endpoint
app.MapPost("/api/chat/stream", (ChatRequest request, RagChatService chatService) =>
{
    return Results.Stream(async stream =>
    {
        var writer = new StreamWriter(stream);
        await foreach (var chunk in chatService.AskStreamAsync(request.Message))
        {
            await writer.WriteAsync(chunk);
            await writer.FlushAsync();
        }
    }, "text/plain");
});

// Document ingestion endpoint
app.MapPost("/api/ingest", async (
    DocumentParser parser,
    TextChunker chunker,
    VectorStoreService store) =>
{
    if (!Directory.Exists(documentsPath))
        return Results.BadRequest("Documents directory not found.");

    var files = Directory.GetFiles(documentsPath)
        .Where(f => new[] { ".txt", ".md", ".pdf", ".docx", ".csv" }
            .Contains(Path.GetExtension(f).ToLowerInvariant()))
        .ToList();

    var totalChunks = 0;
    foreach (var file in files)
    {
        var text = parser.ParseFile(file);
        var chunks = chunker.ChunkText(text);
        await store.StoreChunksAsync(chunks, Path.GetFileName(file));
        totalChunks += chunks.Count;
    }

    return Results.Ok(new IngestResponse(files.Count, totalChunks));
});

// Finetuned model chat endpoint (no RAG)
app.MapPost("/api/chat/finetuned", async (ChatRequest request, FineTunedChatService chatService) =>
{
    var response = await chatService.AskAsync(request.Message);
    return Results.Ok(response);
});

app.MapPost("/api/chat/finetuned/stream", (ChatRequest request, FineTunedChatService chatService) =>
{
    return Results.Stream(async stream =>
    {
        var writer = new StreamWriter(stream);
        await foreach (var chunk in chatService.AskStreamAsync(request.Message))
        {
            await writer.WriteAsync(chunk);
            await writer.FlushAsync();
        }
    }, "text/plain");
});

// Training proxy endpoints
app.MapPost("/api/train/start", async (IHttpClientFactory httpFactory) =>
{
    var client = httpFactory.CreateClient("FineTuneService");
    var resp = await client.PostAsync("/api/train/start", null);
    var content = await resp.Content.ReadAsStringAsync();
    return Results.Content(content, "application/json");
});

app.MapGet("/api/train/status", async (IHttpClientFactory httpFactory) =>
{
    var client = httpFactory.CreateClient("FineTuneService");
    var resp = await client.GetAsync("/api/train/status");
    var content = await resp.Content.ReadAsStringAsync();
    return Results.Content(content, "application/json");
});

app.MapGet("/api/health", () => Results.Ok(new { Status = "OK" }));

app.Run();
