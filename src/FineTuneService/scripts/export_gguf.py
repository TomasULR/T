"""
Export fine-tuned modelu do GGUF formátu.

Načte LoRA adapter, mergne ho s base modelem a exportuje jako GGUF Q4_K_M.
Výstup: /app/models/gguf/unsloth.Q4_K_M.gguf
"""

import logging
import yaml
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "/app/config/training_config.yaml"
LORA_PATH = "/app/models/lora_adapter"
GGUF_OUTPUT_DIR = "/models/gguf"  # Sdílený volume s Ollama

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

logger.info("Načítám fine-tuned model (LoRA adapter)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=config["max_seq_length"],
    load_in_4bit=False,  # Pro export potřebujeme full precision
)

logger.info(f"Exportuji do GGUF ({config['quantization']})...")
model.save_pretrained_gguf(
    GGUF_OUTPUT_DIR,
    tokenizer,
    quantization_method=config["quantization"],
)

logger.info(f"GGUF export dokončen → {GGUF_OUTPUT_DIR}")
