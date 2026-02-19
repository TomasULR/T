# Fine-tuning pipeline pro lokální LLM — Dokumentace

## Přehled projektu

Tento projekt implementuje **hybridní AI systém** kombinující dvě techniky:

1. **RAG (Retrieval-Augmented Generation)** — model dostává relevantní kontext z dokumentů při každém dotazu
2. **Fine-tuning** — model se dotrénuje na datech z dokumentů, aby si je "zapamatoval"

Výsledkem je systém, kde fine-tuned model má znalosti přímo ve vahách + RAG zajišťuje přesnost a citace zdrojů.

---

## Použité technologie

### Backend

| Technologie | Verze | Účel |
|---|---|---|
| **.NET 10** | 10.0 | RagService (API) + BlazorUI (frontend) |
| **Semantic Kernel** | 1.72.0 | Orchestrace LLM volání, embedding, chat completion |
| **Qdrant** | latest | Vektorová databáze pro ukládání a vyhledávání dokumentových chunků |
| **Ollama** | latest | Lokální inference LLM modelů (Gemma 2 9B, nomic-embed-text) |
| **PdfPig** | 0.1.14-alpha | Parsování PDF dokumentů |
| **OpenXml** | 3.4.1 | Parsování DOCX dokumentů |

### Fine-tuning

| Technologie | Účel |
|---|---|
| **Python 3 + FastAPI** | API server řídící tréninkový pipeline |
| **Unsloth** | Optimalizovaný QLoRA trénink (2× rychlejší, 60% méně VRAM) |
| **PyTorch** | ML framework pro trénink |
| **TRL (SFTTrainer)** | Supervised fine-tuning trainer od Hugging Face |
| **PEFT** | Parameter-Efficient Fine-Tuning (LoRA adaptéry) |
| **bitsandbytes** | 4-bit kvantizace pro úsporu VRAM |
| **Gemma 2 2B** | Základní model pro fine-tuning (vejde se do 6GB VRAM) |

### Infrastruktura

| Technologie | Účel |
|---|---|
| **Docker Compose** | Orchestrace všech 5 služeb |
| **NVIDIA CUDA 12.1** | GPU akcelerace pro trénink |
| **Bootstrap 5** | CSS framework pro Blazor UI |

---

## Architektura systému

```
┌─────────────────────────────────────────────────────────────┐
│                        Docker Compose                        │
│                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────┐   │
│  │ BlazorUI │───▶│ RagService  │───▶│     Ollama        │   │
│  │ :8080    │    │ :5100       │    │ :11434            │   │
│  │ (.NET)   │    │ (.NET)      │    │ gemma2:9b (RAG)   │   │
│  └──────────┘    │             │    │ nomic-embed-text  │   │
│                  │             │───▶│ gemma2-finetuned  │   │
│                  │             │    └───────────────────┘   │
│                  │             │                             │
│                  │             │───▶┌──────────┐            │
│                  └─────────────┘    │  Qdrant  │            │
│                        │            │  :6333   │            │
│                        │ proxy      └──────────┘            │
│                        ▼                                     │
│                  ┌─────────────────┐                         │
│                  │ FineTuneService │                         │
│                  │ :8090 (Python)  │                         │
│                  │ CUDA + Unsloth  │                         │
│                  └─────────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

### Služby

1. **BlazorUI** (port 8080) — webové rozhraní s chatem a správou fine-tuningu
2. **RagService** (port 5100) — .NET API, řeší RAG chat, ingest dokumentů, proxy na fine-tuning
3. **Ollama** (port 11434) — lokální LLM server s GPU akcelerací
4. **Qdrant** (port 6333) — vektorová databáze pro RAG
5. **FineTuneService** (port 8090) — Python pipeline pro QLoRA fine-tuning

---

## Jak funguje RAG (chat)

```
Uživatel zadá dotaz
  → BlazorUI odešle POST /api/chat na RagService
  → RagService embedne dotaz přes nomic-embed-text
  → Vyhledá top 5 nejpodobnějších chunků v Qdrantu (cosine similarity)
  → Sestaví prompt: system prompt + nalezený kontext + dotaz
  → Pošle do Ollama (gemma2:9b-instruct-q3_K_M)
  → Vrátí odpověď + seznam zdrojů
```

### Ingest dokumentů

1. Uživatel vloží soubory do složky `documents/` (podporované formáty: .txt, .md, .csv, .pdf, .docx)
2. Klikne "Načíst dokumenty" v UI (nebo POST `/api/ingest`)
3. RagService parsuje soubory, rozseká na chunky (500 slov, overlap 50)
4. Každý chunk se embedne a uloží do Qdrantu

---

## Jak funguje fine-tuning pipeline

Pipeline má 4 kroky, každý běží jako samostatný Python skript:

### Krok 1: Příprava dat (`prepare_data.py`)

- Čte dokumenty z `documents/`
- Rozdělí na chunky (500 slov, overlap 50)
- Pro každý chunk zavolá Gemma 2 9B přes Ollama API
- Model vygeneruje 3–5 párů otázka/odpověď v češtině
- Výstup: `training_data/dataset.jsonl` ve formátu:
  ```json
  {"instruction": "Kolik dnů dovolené mají zaměstnanci?", "input": "", "output": "Zaměstnanci mají 25 dnů..."}
  ```

### Krok 2: QLoRA trénink (`train.py`)

- Načte Gemma 2 2B ve 4-bit kvantizaci přes Unsloth
- Aplikuje LoRA adaptéry na všechny attention + MLP vrstvy (rank=16)
- Trénuje 3 epochy s batch_size=2, gradient_accumulation=4
- Gradient checkpointing pro úsporu VRAM
- Výstup: LoRA adaptér v `models/lora_adapter/`

**VRAM rozpočet (~3.8 GB celkem):**

| Položka | VRAM |
|---|---|
| Gemma 2 2B 4-bit | ~1.5 GB |
| LoRA adaptéry | ~0.1 GB |
| Optimizer (Adam) | ~0.2 GB |
| Aktivace + grad checkpoint | ~1.5 GB |
| CUDA overhead | ~0.5 GB |

### Krok 3: Export do GGUF (`export_gguf.py`)

- Načte LoRA adaptér (full precision na CPU)
- Mergne s base modelem
- Kvantizuje do formátu Q4_K_M
- Výstup: `models/gguf/unsloth.Q4_K_M.gguf` (sdílený Docker volume s Ollama)

> Tento krok potřebuje ~8 GB systémové RAM, běží na CPU.

### Krok 4: Import do Ollama (`import_ollama.py`)

- Vytvoří Ollama Modelfile s:
  - Cestou k GGUF souboru
  - Gemma 2 chat šablonou (`<start_of_turn>user ... <end_of_turn>`)
  - Českým system promptem
  - Parametry: temperature 0.7, context window 2048
- Zaregistruje model v Ollama jako `gemma2-finetuned`

### Strategie dvou modelů

| Model | Použití | Velikost |
|---|---|---|
| **Gemma 2 9B** (Q3 kvantizace) | RAG inference + generování trénovacích dat | ~5 GB VRAM |
| **Gemma 2 2B** (4-bit) | Fine-tuning cíl (trénuje se) | ~1.5 GB VRAM |
| **nomic-embed-text** | Embedding pro vektorové vyhledávání | malý |

GPU sdílení: Ollama a trénink nemohou běžet současně na RTX 3060 6GB — pipeline pozastaví Ollama model před tréninkem.

---

## Struktura projektu

```
T/
├── docker-compose.yml              # Orchestrace všech služeb
├── setup.sh                        # Automatický setup skript
├── T.sln                           # .NET solution
├── documents/                      # Vstupní dokumenty pro RAG a fine-tuning
│   ├── README.md
│   └── ukazka.txt                  # Ukázkový dokument
├── training_data/
│   └── dataset.jsonl               # Vygenerované Q&A páry
├── models/                         # Výstupní modely (LoRA adapter, GGUF)
└── src/
    ├── RagService/                 # .NET 10 backend API
    │   ├── Dockerfile
    │   ├── RagService.csproj
    │   ├── Program.cs              # Endpointy + DI
    │   ├── appsettings.json
    │   ├── Models/
    │   │   └── ChatRequest.cs      # DTO (ChatRequest, ChatResponse, ...)
    │   └── Services/
    │       ├── DocumentParser.cs   # Parsování PDF, DOCX, TXT, MD, CSV
    │       ├── TextChunker.cs      # Sliding-window chunking
    │       ├── VectorStoreService.cs # Qdrant operace
    │       └── RagChatService.cs   # RAG logika (search + LLM)
    ├── BlazorUI/                   # .NET 10 Blazor Server frontend
    │   ├── Dockerfile
    │   ├── BlazorUI.csproj
    │   ├── Program.cs
    │   ├── Services/
    │   │   └── RagApiClient.cs     # HTTP klient pro RagService
    │   └── Components/
    │       ├── App.razor
    │       ├── Routes.razor
    │       ├── Layout/
    │       │   ├── MainLayout.razor
    │       │   └── NavMenu.razor
    │       └── Pages/
    │           ├── Home.razor      # Úvodní stránka
    │           ├── Chat.razor      # RAG chat
    │           └── Training.razor  # Správa fine-tuningu
    └── FineTuneService/            # Python FastAPI
        ├── Dockerfile              # CUDA 12.1 + Python
        ├── requirements.txt
        ├── app.py                  # FastAPI server + orchestrace
        ├── config/
        │   └── training_config.yaml # Hyperparametry
        └── scripts/
            ├── prepare_data.py     # Dokumenty → JSONL
            ├── train.py            # QLoRA trénink
            ├── export_gguf.py      # Merge + GGUF konverze
            └── import_ollama.py    # Import do Ollama
```

---

## API endpointy

### RagService (port 5100)

| Metoda | Endpoint | Popis |
|---|---|---|
| POST | `/api/chat` | RAG chat — vrátí odpověď + zdroje |
| POST | `/api/chat/stream` | Streaming RAG chat (text/plain) |
| POST | `/api/ingest` | Načte a zaindexuje dokumenty |
| POST | `/api/train/start` | Spustí fine-tuning (proxy na FineTuneService) |
| GET | `/api/train/status` | Stav fine-tuningu (proxy na FineTuneService) |
| GET | `/api/health` | Health check |

### FineTuneService (port 8090)

| Metoda | Endpoint | Popis |
|---|---|---|
| POST | `/api/train/start` | Spustí pipeline v background threadu |
| GET | `/api/train/status` | Vrátí `{state, message, error}` |
| GET | `/api/health` | Health check |

---

## Setup — jak projekt rozběhnout

### Prerekvizity

- **Docker Desktop** s podporou Linux kontejnerů
- **NVIDIA GPU** (minimálně 6 GB VRAM, testováno na RTX 3060)
- **NVIDIA Container Toolkit** (pro GPU passthrough do Dockeru)
- **~20 GB místa na disku** (Docker images + modely)
- **~16 GB RAM** (pro GGUF export)

### Instalace NVIDIA Container Toolkit (Windows + WSL2)

1. Nainstalovat nejnovější NVIDIA ovladače pro Windows
2. Povolit WSL2 v Docker Desktop (Settings → General → Use WSL 2)
3. V Docker Desktop → Settings → Resources → WSL Integration → povolit pro vaši WSL distribuci

### Spuštění

#### Varianta A: Automatický setup

```bash
chmod +x setup.sh
./setup.sh
```

Skript automaticky:
1. Sestaví a spustí všechny kontejnery
2. Počká na připravenost Ollama
3. Stáhne potřebné modely (gemma2:9b-instruct-q3_K_M, nomic-embed-text)
4. Počká na připravenost RagService

#### Varianta B: Manuální setup

```bash
# 1. Spustit všechny služby
docker compose up -d --build

# 2. Počkat na Ollama a stáhnout modely
docker exec -it t-ollama-1 ollama pull gemma2:9b-instruct-q3_K_M
docker exec -it t-ollama-1 ollama pull nomic-embed-text

# 3. Ověřit že vše běží
docker compose ps
curl http://localhost:5100/api/health
curl http://localhost:8090/api/health
```

### Přístup ke službám

| Služba | URL |
|---|---|
| **Blazor UI** | http://localhost:8080 |
| **RagService API** | http://localhost:5100 |
| **Ollama** | http://localhost:11434 |
| **Qdrant Dashboard** | http://localhost:6333/dashboard |
| **FineTuneService API** | http://localhost:8090 |

---

## Použití

### 1. Přidání dokumentů

Vložte soubory do složky `documents/`. Podporované formáty:
- `.txt`, `.md`, `.csv` — prostý text
- `.pdf` — PDF dokumenty
- `.docx` — Word dokumenty

### 2. Ingest (zaindexování)

V Blazor UI → Chat → klikněte "Načíst dokumenty". Dokumenty se rozparsují, rozchunkují a uloží do Qdrantu.

### 3. Chat (RAG)

V Blazor UI → Chat → zadejte dotaz. Model odpovídá na základě kontextu z dokumentů.

### 4. Fine-tuning

1. V Blazor UI → Fine-tuning → "Spustit fine-tuning"
2. Sledujte progress (4 kroky):
   - **Příprava dat** — generování Q&A párů z dokumentů
   - **Trénink** — QLoRA fine-tuning Gemma 2 2B
   - **Export** — konverze do GGUF formátu
   - **Import** — registrace modelu v Ollama
3. Po dokončení je model `gemma2-finetuned` dostupný v Ollama

### 5. Testování fine-tuned modelu

Přepněte aktivní model a otestujte dotazy — model by měl odpovídat na základě natrénovaných znalostí i bez RAG kontextu.

---

## Konfigurace

### Trénovací hyperparametry

Soubor: `src/FineTuneService/config/training_config.yaml`

| Parametr | Výchozí hodnota | Popis |
|---|---|---|
| `base_model` | `google/gemma-2-2b-it` | Základní model pro fine-tuning |
| `max_seq_length` | 2048 | Maximální délka sekvence |
| `lora_rank` | 16 | Rank LoRA adaptéru |
| `lora_alpha` | 16 | Alpha parametr LoRA |
| `epochs` | 3 | Počet epoch tréninku |
| `batch_size` | 2 | Velikost batche |
| `gradient_accumulation_steps` | 4 | Efektivní batch = 2×4 = 8 |
| `learning_rate` | 0.0002 | Learning rate |
| `quantization` | `q4_k_m` | Kvantizace výstupního GGUF |

### Docker Compose proměnné prostředí

Lze upravit přímo v `docker-compose.yml`:

- `Ollama__ChatModel` — model pro RAG inference
- `Ollama__EmbeddingModel` — model pro embeddingy
- `GENERATION_MODEL` — model pro generování trénovacích dat
- `FINETUNED_MODEL_NAME` — název výstupního fine-tuned modelu

---

## Známá omezení a rizika

1. **GPU kontence** — Ollama a trénink nemohou běžet současně na 6 GB VRAM. Pipeline by měl pozastavit Ollama model před tréninkem.
2. **Kvalita dat** — Doporučujeme manuálně zkontrolovat `training_data/dataset.jsonl` před spuštěním tréninku.
3. **GGUF export** — Potřebuje ~8 GB systémové RAM (běží na CPU, ne GPU).
4. **První spuštění** — Stažení modelů může trvat dlouho (Gemma 2 9B ~5 GB).
5. **Windows/WSL2** — GPU passthrough vyžaduje správně nakonfigurovaný NVIDIA Container Toolkit.
