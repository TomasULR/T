import subprocess
import threading
import logging
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fine-Tune Service")


class TrainingState:
    def __init__(self):
        self.state: str = "idle"
        self.message: str = ""
        self.error: str = ""


status = TrainingState()


def run_script(script_path: str, description: str) -> None:
    logger.info(f"Running: {description}")
    result = subprocess.run(
        ["python3", script_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{description} selhalo:\n{result.stderr}")
    logger.info(f"Done: {description}")


def run_pipeline() -> None:
    global status
    try:
        status.state = "preparing_data"
        status.message = "Generuji trénovací data z dokumentů..."
        run_script("/app/scripts/prepare_data.py", "Příprava dat")

        status.state = "training"
        status.message = "Trénuji model (QLoRA na Gemma 2 2B)..."
        run_script("/app/scripts/train.py", "QLoRA trénink")

        status.state = "exporting"
        status.message = "Exportuji do GGUF formátu..."
        run_script("/app/scripts/export_gguf.py", "GGUF export")

        status.state = "importing"
        status.message = "Importuji do Ollama..."
        run_script("/app/scripts/import_ollama.py", "Ollama import")

        status.state = "done"
        status.message = "Fine-tuning dokončen! Model 'gemma2-finetuned' je připraven."
        status.error = ""

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        status.state = "error"
        status.error = str(e)


@app.post("/api/train/start")
def start_training():
    if status.state in ("preparing_data", "training", "exporting", "importing"):
        return {"error": "Trénink již probíhá", "state": status.state}

    status.state = "starting"
    status.message = "Spouštím pipeline..."
    status.error = ""

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    return {"message": "Trénink spuštěn"}


@app.get("/api/train/status")
def get_status():
    return {
        "state": status.state,
        "message": status.message,
        "error": status.error,
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
