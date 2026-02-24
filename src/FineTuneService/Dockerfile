FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv git curl \
    build-essential cmake g++ gcc pkg-config && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Clone a build llama.cpp pro GGUF export
RUN git clone https://github.com/ggerganov/llama.cpp.git && \
    cd llama.cpp && \
    cmake -B build && \
    cmake --build build --config Release && \
    cp build/bin/llama-quantize . 2>/dev/null || \
    cp build/bin/quantize . 2>/dev/null || true && \
    cp build/bin/llama-gguf-split . 2>/dev/null || true

# Nastavit proměnnou prostředí pro Unsloth
ENV LLAMA_CPP_DIR=/app/llama.cpp

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8090
CMD ["python3", "app.py"]
