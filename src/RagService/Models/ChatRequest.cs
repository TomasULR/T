namespace RagService.Models;

public record ChatRequest(string Message);
public record ChatResponse(string Answer, List<string> Sources);
public record IngestResponse(int DocumentCount, int ChunkCount);
public record TrainingStatusResponse(string State, string Message, string Error);
