#!/bin/bash
# Convenience launcher — activates venv and starts the server

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "venv" ]; then
  echo "Virtual environment not found. Run setup first:"
  echo "  python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

source venv/bin/activate
mkdir -p logs data/pdfs data/qdrant_db static

echo ""
echo "  RAG Chatbot starting at http://localhost:8000"
echo "  Press Ctrl+C to stop."
echo ""

python -m uvicorn api.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8000}" \
  --workers 1 \
  --log-level info
