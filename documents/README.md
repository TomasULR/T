# Dokumenty pro trénování

Sem vkládejte soubory, které chcete použít jako znalostní bázi pro LLM.

## Podporované formáty
- `.txt` — prostý text
- `.md` — Markdown
- `.pdf` — PDF dokumenty
- `.docx` — Word dokumenty
- `.csv` — tabulková data

## Jak to funguje
1. Vložte soubory do této složky
2. Spusťte indexaci (`POST /api/ingest`)
3. Soubory budou rozřezány na chunky, převedeny na vektory a uloženy do Qdrant
4. Při dotazu se najdou relevantní chunky a předají modelu jako kontext
