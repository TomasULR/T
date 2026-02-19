namespace RagService.Services;

public class TextChunker
{
    private readonly int _chunkSize;
    private readonly int _overlap;

    public TextChunker(int chunkSize = 500, int overlap = 50)
    {
        _chunkSize = chunkSize;
        _overlap = overlap;
    }

    public List<string> ChunkText(string text)
    {
        var chunks = new List<string>();
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);

        if (words.Length <= _chunkSize)
        {
            chunks.Add(string.Join(' ', words));
            return chunks;
        }

        for (var i = 0; i < words.Length; i += _chunkSize - _overlap)
        {
            var chunk = words.Skip(i).Take(_chunkSize).ToArray();
            if (chunk.Length == 0) break;
            chunks.Add(string.Join(' ', chunk));
        }

        return chunks;
    }
}
