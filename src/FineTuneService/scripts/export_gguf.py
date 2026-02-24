"""
Export fine-tuned modelu do GGUF formátu.

Načte LoRA adapter, mergne ho s base modelem, konvertuje do GGUF a kvantizuje.
Výstup: /models/gguf/<model_name>.Q4_K_M.gguf
"""

import logging
import yaml
import os
import subprocess
import shutil
from unsloth import FastLanguageModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "/app/config/training_config.yaml"
LORA_PATH = "/app/models/lora_adapter"
GGUF_OUTPUT_DIR = "/models/gguf"  # Sdílený volume s Ollama
MERGED_DIR = "/app/models/merged_model"
LLAMA_CPP_DIR = "/app/llama.cpp"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

quantization = config["quantization"]  # e.g. "q4_k_m"

# Step 1: Load and merge LoRA adapter
logger.info("Načítám fine-tuned model (LoRA adapter)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH,
    max_seq_length=config["max_seq_length"],
    load_in_4bit=False,
)

# Step 2: Save merged model in HF format
logger.info("Ukládám merged model ve formátu HuggingFace...")
model.save_pretrained_merged(MERGED_DIR, tokenizer, save_method="merged_16bit")
logger.info(f"Merged model uložen do {MERGED_DIR}")

# Step 3: Convert HF to GGUF bf16 using llama.cpp
converter = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
if not os.path.exists(converter):
    raise RuntimeError(f"Converter script nenalezen: {converter}")

bf16_gguf = os.path.join(GGUF_OUTPUT_DIR, "model-bf16.gguf")
os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

logger.info("Konvertuji HF -> GGUF bf16...")
result = subprocess.run(
    ["python3", converter, MERGED_DIR, "--outfile", bf16_gguf, "--outtype", "bf16"],
    capture_output=True, text=True,
)
if result.returncode != 0:
    logger.error(f"Konverze selhala: {result.stderr}")
    raise RuntimeError(f"HF->GGUF konverze selhala: {result.stderr[-500:]}")
logger.info("GGUF bf16 konverze dokončena.")

# Step 4: Quantize to target format
quantizer = os.path.join(LLAMA_CPP_DIR, "llama-quantize")
if not os.path.exists(quantizer):
    quantizer = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")
if not os.path.exists(quantizer):
    raise RuntimeError("llama-quantize binárka nenalezena!")

final_gguf = os.path.join(GGUF_OUTPUT_DIR, f"unsloth.Q4_K_M.gguf")

logger.info(f"Kvantizuji: {quantization.upper()}...")
result = subprocess.run(
    [quantizer, bf16_gguf, final_gguf, quantization.upper()],
    capture_output=True, text=True,
)
if result.returncode != 0:
    logger.error(f"Kvantizace selhala: {result.stderr}")
    raise RuntimeError(f"Kvantizace selhala: {result.stderr[-500:]}")

# Cleanup intermediate files
os.remove(bf16_gguf)
shutil.rmtree(MERGED_DIR, ignore_errors=True)

logger.info(f"GGUF export dokončen → {final_gguf}")
