using System.Text;
using Microsoft.SemanticKernel.ChatCompletion;
using RagService.Models;

namespace RagService.Services;

public class RagChatService
{
    private readonly VectorStoreService _vectorStore;
    private readonly IChatCompletionService _chatService;

    private const string SystemPrompt = """
        Jsi český asistent. Vždy odpovídej výhradně v češtině.
        Používej správnou českou gramatiku a diakritiku.
        Odpovídej pouze na základě poskytnutého kontextu.
        Pokud kontext neobsahuje odpověď, řekni, že nemáš dostatek informací.
        Nikdy si nevymýšlej informace, které nejsou v kontextu.
        """;

    public RagChatService(VectorStoreService vectorStore, IChatCompletionService chatService)
    {
        _vectorStore = vectorStore;
        _chatService = chatService;
    }

    public async Task<ChatResponse> AskAsync(string question)
    {
        var relevantChunks = await _vectorStore.SearchAsync(question, topK: 5);

        var contextBuilder = new StringBuilder();
        var sources = new List<string>();

        foreach (var (content, source) in relevantChunks)
        {
            contextBuilder.AppendLine(content);
            contextBuilder.AppendLine("---");
            if (!sources.Contains(source))
                sources.Add(source);
        }

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage(SystemPrompt);
        chatHistory.AddUserMessage($"""
            Kontext:
            {contextBuilder}

            Otázka: {question}
            """);

        var response = await _chatService.GetChatMessageContentAsync(chatHistory);
        return new ChatResponse(response.Content ?? "Bez odpovědi.", sources);
    }

    public async IAsyncEnumerable<string> AskStreamAsync(string question)
    {
        var relevantChunks = await _vectorStore.SearchAsync(question, topK: 5);

        var contextBuilder = new StringBuilder();
        foreach (var (content, _) in relevantChunks)
        {
            contextBuilder.AppendLine(content);
            contextBuilder.AppendLine("---");
        }

        var chatHistory = new ChatHistory();
        chatHistory.AddSystemMessage(SystemPrompt);
        chatHistory.AddUserMessage($"""
            Kontext:
            {contextBuilder}

            Otázka: {question}
            """);

        await foreach (var content in _chatService.GetStreamingChatMessageContentsAsync(chatHistory))
        {
            if (content.Content is not null)
                yield return content.Content;
        }
    }
}
