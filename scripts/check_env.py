"""
Quick pre-flight check — run before ingestion or serving.
Verifies: Python version, Tesseract, Groq key, Qdrant path, PDF dir.
"""
import sys
import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PASS = "\033[32m✔\033[0m"
FAIL = "\033[31m✘\033[0m"
WARN = "\033[33m⚠\033[0m"

ok = True

def check(cond, msg_pass, msg_fail, warn=False):
    global ok
    if cond:
        print(f"  {PASS}  {msg_pass}")
    else:
        sym = WARN if warn else FAIL
        print(f"  {sym}  {msg_fail}")
        if not warn:
            ok = False

print("\n=== RAG Chatbot — Environment Check ===\n")

# Python version
check(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]}", "Python ≥3.10 required")

# Tesseract
try:
    out = subprocess.check_output(["tesseract", "--version"], stderr=subprocess.STDOUT).decode()
    check(True, f"Tesseract: {out.splitlines()[0]}", "")
except Exception:
    check(False, "", "Tesseract not found. Install: sudo apt install tesseract-ocr")

# Poppler (for pdf2image)
try:
    subprocess.check_output(["pdftoppm", "-v"], stderr=subprocess.STDOUT)
    check(True, "Poppler (pdftoppm) found", "")
except Exception:
    check(False, "", "Poppler not found. Install: sudo apt install poppler-utils", warn=True)

# GROQ_API_KEY
key = os.getenv("GROQ_API_KEY", "")
check(bool(key) and key != "your_groq_api_key_here", "GROQ_API_KEY set", "GROQ_API_KEY missing in .env")

# Qdrant path writable
qdrant_path = Path(os.getenv("QDRANT_PATH", "./data/qdrant_db"))
qdrant_path.mkdir(parents=True, exist_ok=True)
check(qdrant_path.exists(), f"Qdrant path: {qdrant_path}", f"Cannot create {qdrant_path}")

# PDF dir
pdf_dir = Path("./data/pdfs")
pdf_dir.mkdir(parents=True, exist_ok=True)
pdfs = list(pdf_dir.glob("*.pdf"))
check(len(pdfs) > 0, f"PDF dir: {len(pdfs)} PDF(s) found", "No PDFs in data/pdfs/ yet — add PDFs before ingesting", warn=True)

# Logs dir
Path("./logs").mkdir(exist_ok=True)
check(True, "Logs dir ready", "")

print(f"\n{'All checks passed!' if ok else 'Fix the issues above before proceeding.'}\n")
sys.exit(0 if ok else 1)
