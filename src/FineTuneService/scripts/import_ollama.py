"""
Import GGUF modelu do Ollama.

Vytvoří Modelfile a zaregistruje model v Ollama jako 'gemma2-finetuned'.
"""

import logging
import os
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434")
MODEL_NAME = os.environ.get("FINETUNED_MODEL_NAME", "gemma2-finetuned")
GGUF_PATH = "/models/gguf/unsloth.Q4_K_M.gguf"  # Cesta uvnitř Ollama kontejneru

MODELFILE = f"""FROM {GGUF_PATH}

TEMPLATE \"\"\"<start_of_turn>user
{{{{.Prompt}}}}<end_of_turn>
<start_of_turn>model
{{{{.Response}}}}<end_of_turn>
\"\"\"

PARAMETER stop <end_of_turn>
PARAMETER temperature 0.7
PARAMETER num_ctx 2048

SYSTEM Jsi český asistent. Vždy odpovídej výhradně v češtině. Používej správnou českou gramatiku a diakritiku.
"""

logger.info(f"Vytvářím model '{MODEL_NAME}' v Ollama...")
logger.info(f"GGUF cesta: {GGUF_PATH}")

resp = requests.post(
    f"{OLLAMA_URL}/api/create",
    json={
        "name": MODEL_NAME,
        "modelfile": MODELFILE,
    },
    timeout=600,
)

if resp.status_code == 200:
    logger.info(f"Model '{MODEL_NAME}' úspěšně vytvořen v Ollama!")
else:
    error_msg = resp.text
    logger.error(f"Chyba při vytváření modelu: {resp.status_code} - {error_msg}")
    raise RuntimeError(f"Ollama create selhalo: {error_msg}")
