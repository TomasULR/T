using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using RagService.Models;

namespace RagService.Services;

public class FineTunedChatService
{
    private readonly HttpClient _http;
    private readonly string _model;

    private const string SystemPrompt =
        "Jsi český asistent. Vždy odpovídej výhradně v češtině. Používej správnou českou gramatiku a diakritiku.";

    public FineTunedChatService(HttpClient http, string model)
    {
        _http = http;
        _model = model;
    }

    public async Task<ChatResponse> AskAsync(string question)
    {
        var request = new
        {
            model = _model,
            messages = new[]
            {
                new { role = "system", content = SystemPrompt },
                new { role = "user", content = question }
            },
            stream = false
        };

        var response = await _http.PostAsJsonAsync("/api/chat", request);
        response.EnsureSuccessStatusCode();

        var result = await response.Content.ReadFromJsonAsync<OllamaChatResponse>();
        var answer = result?.Message?.Content ?? "Bez odpovědi.";

        return new ChatResponse(answer, []);
    }

    public async IAsyncEnumerable<string> AskStreamAsync(string question)
    {
        var request = new
        {
            model = _model,
            messages = new[]
            {
                new { role = "system", content = SystemPrompt },
                new { role = "user", content = question }
            },
            stream = true
        };

        var httpRequest = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = JsonContent.Create(request)
        };

        var response = await _http.SendAsync(httpRequest, HttpCompletionOption.ResponseHeadersRead);
        response.EnsureSuccessStatusCode();

        using var stream = await response.Content.ReadAsStreamAsync();
        using var reader = new StreamReader(stream);

        while (await reader.ReadLineAsync() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;

            var chunk = JsonSerializer.Deserialize<OllamaChatResponse>(line);
            if (chunk?.Message?.Content is not null)
                yield return chunk.Message.Content;
        }
    }

    private record OllamaChatResponse(OllamaChatMessage? Message, bool Done);
    private record OllamaChatMessage(string Role, string Content);
}
