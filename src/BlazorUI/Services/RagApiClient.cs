using System.Net.Http.Json;

namespace BlazorUI.Services;

public record ChatResponse(string Answer, List<string> Sources);
public record TrainingStatusResponse(string State, string Message, string Error);

public class RagApiClient
{
    private readonly HttpClient _http;

    public RagApiClient(HttpClient http)
    {
        _http = http;
    }

    public async Task<ChatResponse?> AskAsync(string message)
    {
        var response = await _http.PostAsJsonAsync("/api/chat", new { Message = message });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ChatResponse>();
    }

    public async IAsyncEnumerable<string> AskStreamAsync(string message)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat/stream")
        {
            Content = JsonContent.Create(new { Message = message })
        };

        var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        var buffer = new char[64];
        int bytesRead;
        while ((bytesRead = await reader.ReadAsync(buffer, 0, buffer.Length)) > 0)
        {
            yield return new string(buffer, 0, bytesRead);
        }
    }

    public async Task<ChatResponse?> AskFinetunedAsync(string message)
    {
        var response = await _http.PostAsJsonAsync("/api/chat/finetuned", new { Message = message });
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<ChatResponse>();
    }

    public async Task<string> IngestAsync()
    {
        var response = await _http.PostAsync("/api/ingest", null);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    public async Task<string> StartTrainingAsync()
    {
        var response = await _http.PostAsync("/api/train/start", null);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    public async Task<TrainingStatusResponse?> GetTrainingStatusAsync()
    {
        return await _http.GetFromJsonAsync<TrainingStatusResponse>("/api/train/status");
    }
}
