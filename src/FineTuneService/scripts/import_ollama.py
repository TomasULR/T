"""
Import GGUF modelu do Ollama.

Vytvoří model v Ollama z GGUF souboru jako 'gemma2-finetuned'.
Používá Ollama blob upload API + create s SHA256 digestem.
"""

import logging
import os
import hashlib
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.environ.get("FINETUNED_MODEL_NAME", "gemma2-finetuned")
GGUF_PATH = "/models/gguf/unsloth.Q4_K_M.gguf"

# Verify GGUF file exists
if not os.path.exists(GGUF_PATH):
    raise RuntimeError(f"GGUF soubor nenalezen: {GGUF_PATH}")

file_size = os.path.getsize(GGUF_PATH)
logger.info(f"Vytvářím model '{MODEL_NAME}' v Ollama...")
logger.info(f"GGUF: {GGUF_PATH} ({file_size / 1e9:.2f} GB)")

# Step 1: Calculate SHA256 of the GGUF file
logger.info("Počítám SHA256 digest...")
sha256 = hashlib.sha256()
with open(GGUF_PATH, "rb") as f:
    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
        sha256.update(chunk)
digest = f"sha256:{sha256.hexdigest()}"
logger.info(f"Digest: {digest}")

# Step 2: Check if blob already exists
head_resp = requests.head(f"{OLLAMA_URL}/api/blobs/{digest}", timeout=10)
if head_resp.status_code == 200:
    logger.info("Blob již existuje v Ollama, přeskakuji upload.")
else:
    # Step 3: Upload blob
    logger.info("Nahrávám GGUF do Ollama blob storage...")
    with open(GGUF_PATH, "rb") as f:
        upload_resp = requests.post(
            f"{OLLAMA_URL}/api/blobs/{digest}",
            data=f,
            headers={"Content-Type": "application/octet-stream"},
            timeout=600,
        )
    if upload_resp.status_code not in (200, 201):
        raise RuntimeError(f"Blob upload selhal: {upload_resp.status_code} - {upload_resp.text}")
    logger.info("Blob úspěšně nahrán.")

# Step 4: Create model from blob
logger.info(f"Vytvářím model '{MODEL_NAME}'...")
resp = requests.post(
    f"{OLLAMA_URL}/api/create",
    json={
        "model": MODEL_NAME,
        "files": {
            "model.gguf": digest,
        },
        "parameters": {
            "stop": ["<end_of_turn>"],
            "temperature": 0.7,
            "num_ctx": 2048,
        },
        "system": "Jsi český asistent. Vždy odpovídej výhradně v češtině. Používej správnou českou gramatiku a diakritiku.",
    },
    timeout=600,
    stream=True,
)

if resp.status_code == 200:
    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            status = data.get("status", "")
            if status:
                logger.info(f"Ollama: {status}")
            if "error" in data:
                raise RuntimeError(f"Ollama create selhalo: {data['error']}")
    logger.info(f"Model '{MODEL_NAME}' úspěšně vytvořen v Ollama!")
else:
    raise RuntimeError(f"Ollama create selhalo: {resp.status_code} - {resp.text}")
