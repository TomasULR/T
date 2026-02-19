"""
Příprava trénovacích dat: dokumenty → Q&A páry → JSONL

Čte soubory z /app/documents, pomocí Gemma 2 9B (přes Ollama)
generuje otázky a odpovědi, výstup ukládá do /app/training_data/dataset.jsonl
"""

import json
import os
import glob
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
DOCS_PATH = os.environ.get("DOCS_PATH", "/app/documents")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/app/training_data/dataset.jsonl")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "gemma2:9b-instruct-q3_K_M")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
    return chunks


def read_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".txt", ".md", ".csv"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    logger.warning(f"Přeskakuji nepodporovaný formát: {filepath}")
    return ""


def generate_qa_pairs(chunk: str) -> list[dict]:
    prompt = f"""Na základě následujícího textu vygeneruj 3-5 párů otázka-odpověď v češtině.
Vrať POUZE platný JSON pole s objekty {{"question": "...", "answer": "..."}}.
Žádný jiný text, žádné vysvětlení, pouze JSON pole.

Text:
{chunk}"""

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": GENERATION_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=300,
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")

        # Najdi JSON pole v odpovědi
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        if start == -1 or end == 0:
            logger.warning(f"Nepodařilo se najít JSON v odpovědi: {response_text[:200]}")
            return []

        pairs = json.loads(response_text[start:end])
        return [p for p in pairs if "question" in p and "answer" in p]

    except (requests.RequestException, json.JSONDecodeError) as e:
        logger.error(f"Chyba při generování Q&A: {e}")
        return []


def main():
    supported_ext = {".txt", ".md", ".csv"}
    all_files = []
    for ext in supported_ext:
        all_files.extend(glob.glob(os.path.join(DOCS_PATH, f"*{ext}")))

    if not all_files:
        logger.error(f"Žádné soubory nenalezeny v {DOCS_PATH}")
        return

    logger.info(f"Nalezeno {len(all_files)} souborů")

    all_pairs = []
    for filepath in all_files:
        filename = os.path.basename(filepath)
        logger.info(f"Zpracovávám: {filename}")

        text = read_file(filepath)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        logger.info(f"  → {len(chunks)} chunků")

        for i, chunk in enumerate(chunks):
            logger.info(f"  → Generuji Q&A pro chunk {i + 1}/{len(chunks)}")
            pairs = generate_qa_pairs(chunk)
            all_pairs.extend(pairs)
            logger.info(f"  → Vygenerováno {len(pairs)} párů")

    # Uložit do JSONL
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            line = json.dumps(
                {
                    "instruction": pair["question"],
                    "input": "",
                    "output": pair["answer"],
                },
                ensure_ascii=False,
            )
            f.write(line + "\n")

    logger.info(f"Celkem vygenerováno {len(all_pairs)} trénovacích párů → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
