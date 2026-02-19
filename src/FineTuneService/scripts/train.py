"""
QLoRA fine-tuning Gemma 2 2B přes Unsloth.

Načte trénovací data z /app/training_data/dataset.jsonl,
provede QLoRA trénink a uloží LoRA adapter do /app/models/lora_adapter/
"""

import logging
import yaml
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_PATH = "/app/config/training_config.yaml"
DATASET_PATH = "/app/training_data/dataset.jsonl"
OUTPUT_DIR = "/app/models/lora_adapter"
CHECKPOINT_DIR = "/app/models/checkpoints"

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

logger.info(f"Konfigurace: {config}")

# Načtení modelu s 4-bit kvantizací
logger.info(f"Načítám model: {config['base_model']}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["base_model"],
    max_seq_length=config["max_seq_length"],
    dtype=None,  # auto-detect
    load_in_4bit=True,
)

# Aplikace LoRA adapterů
logger.info("Aplikuji LoRA adaptery...")
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_rank"],
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=config["lora_alpha"],
    lora_dropout=0,  # Unsloth optimalizace
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% méně VRAM
)

# Formátování dat do Gemma 2 chat template
GEMMA_TEMPLATE = "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn>"


def formatting_func(examples):
    texts = []
    for instr, out in zip(examples["instruction"], examples["output"]):
        text = GEMMA_TEMPLATE.format(instruction=instr, output=out)
        texts.append(text)
    return {"text": texts}


logger.info(f"Načítám dataset: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
logger.info(f"Počet trénovacích vzorků: {len(dataset)}")

dataset = dataset.map(formatting_func, batched=True)

# Trénink
logger.info("Spouštím trénink...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=config["max_seq_length"],
    args=TrainingArguments(
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_steps=5,
        num_train_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        fp16=True,
        logging_steps=1,
        output_dir=CHECKPOINT_DIR,
        save_strategy="epoch",
    ),
)

trainer.train()

# Uložení LoRA adapteru
logger.info(f"Ukládám LoRA adapter do {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

logger.info("Trénink dokončen!")
