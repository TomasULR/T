using System.Text;
using DocumentFormat.OpenXml.Packaging;
using UglyToad.PdfPig;

namespace RagService.Services;

public class DocumentParser
{
    public string ParseFile(string filePath)
    {
        var ext = Path.GetExtension(filePath).ToLowerInvariant();
        return ext switch
        {
            ".txt" or ".md" or ".csv" => File.ReadAllText(filePath, Encoding.UTF8),
            ".pdf" => ParsePdf(filePath),
            ".docx" => ParseDocx(filePath),
            _ => throw new NotSupportedException($"Unsupported file type: {ext}")
        };
    }

    private static string ParsePdf(string filePath)
    {
        var sb = new StringBuilder();
        using var document = PdfDocument.Open(filePath);
        foreach (var page in document.GetPages())
        {
            sb.AppendLine(page.Text);
        }
        return sb.ToString();
    }

    private static string ParseDocx(string filePath)
    {
        using var doc = WordprocessingDocument.Open(filePath, false);
        var body = doc.MainDocumentPart?.Document?.Body;
        return body?.InnerText ?? string.Empty;
    }
}
