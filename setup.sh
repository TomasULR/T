#!/bin/bash
set -e

echo "=== Spouštím kontejnery ==="
docker compose up -d --build

echo ""
echo "=== Čekám na Ollama server ==="
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    echo "  Čekám na Ollama..."
    sleep 3
done
echo "  Ollama běží!"

echo ""
echo "=== Stahuji modely (může trvat několik minut) ==="
echo "  Stahuji Gemma 2 9B Q3 (inference + generování dat)..."
docker exec ollama ollama pull gemma2:9b-instruct-q3_K_M

echo "  Stahuji embedding model..."
docker exec ollama ollama pull nomic-embed-text

echo ""
echo "=== Čekám na RAG Service ==="
until curl -s http://localhost:5100/api/health > /dev/null 2>&1; do
    echo "  Čekám na RAG Service..."
    sleep 3
done
echo "  RAG Service běží!"

echo ""
echo "============================================"
echo "  Vše je připraveno!"
echo ""
echo "  UI:              http://localhost:8080"
echo "  RAG API:         http://localhost:5100"
echo "  Fine-tune API:   http://localhost:8090"
echo "  Ollama:          http://localhost:11434"
echo "  Qdrant:          http://localhost:6333"
echo ""
echo "  Workflow:"
echo "  1. Vložte dokumenty do ./documents/"
echo "  2. Chat → 'Načíst dokumenty' (RAG indexace)"
echo "  3. Fine-tuning → 'Spustit fine-tuning' (QLoRA trénink)"
echo "  4. Po dokončení tréninku je model 'gemma2-finetuned' k dispozici"
echo "============================================"
