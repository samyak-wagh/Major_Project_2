# 🎓 OS Tutor AI — Setup Guide for New Developers

Welcome! This guide walks you through setting up **OS Tutor AI** on your machine from scratch.
The app is an offline-first AI tutor for Operating Systems, powered by Qwen2.5-1.5B + a custom LoRA adapter trained on the Galvin textbook.

---

## 📋 Prerequisites

Before starting, make sure you have the following installed:

| Tool | Download | Why Needed |
|------|----------|------------|
| **Python 3.10 or 3.11** | https://www.python.org/downloads/ | Runs the backend & frontend |
| **Docker Desktop** | https://www.docker.com/products/docker-desktop/ | Runs the Qdrant vector database |
| **Git** | https://git-scm.com/ | To clone the project |

> ⚠️ Python 3.12 may have compatibility issues. Use **3.10 or 3.11** for best results.

---

## 🚀 Step 1 — Clone the Project

```bash
git clone <your-repo-url>
cd Major_Project2
```

---

## 🐍 Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

---

## 📦 Step 3 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This may take 5–10 minutes. It installs PyTorch, Transformers, LangChain, Streamlit, etc.

---

## ⚙️ Step 4 — Configure Environment Variables

Copy the example env file and edit it:

```bash
copy .env.example .env
```

Open `.env` and fill in these key values:

```env
# Primary model
QWEN_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LOCAL_ADAPTER_PATH=qwen-os-tutor-lora

# Backend selector — use qwen for fully offline
LLM_BACKEND=qwen

# HuggingFace cache — change this to a drive with 5+ GB free
HF_HOME=C:\Users\YourName\hf_cache

# Groq fallback (free key at https://console.groq.com)
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=openai/gpt-oss-20b

# Qdrant (leave as-is)
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=os_tutor
```

> 💡 The LoRA adapter (qwen-os-tutor-lora/) is already in the repo — no download needed!
> 💡 The Qwen base model (~3 GB) auto-downloads from HuggingFace on first run.

---

## 🐋 Step 5 — Start Qdrant (Vector Database)

Open Docker Desktop first, then run:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

To verify it is running:
```bash
docker ps
```

You should see qdrant in the list.

> If already installed and stopped, just run: docker start qdrant

---

## 🏗️ Step 6 — Extract Textbook Images (One-Time Setup)

Run this ONCE to extract all textbook pages as PNG images:

```bash
python rag/pdf_image_extractor.py
```

This creates images/ostxtbook_pages/ with ~1000+ page images.

---

## 📚 Step 7 — Build the Figure Index (One-Time Setup)

This maps figure numbers (like "Figure 1.4") to their exact textbook pages:

```bash
python build_figure_index.py
```

Creates figure_index.json and figure_captions.json — needed for diagram retrieval.

---

## ▶️ Step 8 — Start the App

You need TWO terminals open simultaneously:

**Terminal 1 — Backend API:**
```bash
python api.py
```

Wait for:
```
✅ Qwen2.5 primary model ready.
INFO: Uvicorn running on http://0.0.0.0:8001
```

**Terminal 2 — Frontend:**
```bash
python -m streamlit run app.py
```

Open your browser at: http://localhost:8501

---

## 🔥 First Run Notes

- The Qwen base model (~3 GB) downloads automatically on first run.
- Subsequent runs load from cache — much faster!
- The first question may take 30–60 seconds while the model warms up.

---

## 🧪 Quick Test Questions

| Category | Question |
|----------|----------|
| Theory | What are the four conditions for deadlock? |
| Diagram | Show me the process state diagram |
| Numerical | P1(burst=10), P2(burst=4), P3(burst=1), P4(burst=5) — find average waiting time for FCFS and SJF |
| Algorithm | Solve Banker's algorithm: 5 processes, resources A=10 B=5 C=7 |
| Disk | Head at cylinder 53, queue: 98 183 37 122 14 124 65 67 — find FCFS SSTF SCAN total movement |

---

## 🏗️ Project Structure

```
Major_Project2/
├── api.py                   FastAPI backend (port 8001)
├── app.py                   Streamlit frontend (port 8501)
├── rag/
│   ├── qwen_chain.py        Qwen2.5 + LoRA inference chain
│   ├── groq_chain.py        Groq cloud API chain (fallback)
│   └── clip_embeddings.py   CLIP model loader (disabled on Qwen to save RAM)
├── qwen-os-tutor-lora/      Fine-tuned LoRA adapter (already in repo!)
├── figure_index.json        Maps figure X.Y to page number
├── figure_captions.json     Maps figure X.Y to caption text
├── ostxtbook.pdf            Galvin OS textbook
├── images/ostxtbook_pages/  Extracted page PNGs (generated in Step 6)
├── build_figure_index.py    Script to build figure indexes
└── .env                     Your config (copy from .env.example)
```

---

## 🔧 Troubleshooting

**Qdrant connection refused**
→ Docker is not running or Qdrant container is stopped.
→ Run: docker start qdrant

**Out of Memory / Pagefile error**
→ Machine has less than 4 GB RAM. Set DEVICE=cpu and MAX_NEW_TOKENS=512 in .env.

**Groq API 429 Rate Limit**
→ Free tier has 200K tokens/day. Wait 24 hours or get a paid key.
→ Qwen works fully offline — Groq is only a safety net.

**No module named pymupdf**
→ Run: pip install pymupdf

**First run is very slow**
→ Normal! Qwen base model is downloading (~3 GB). Fast on subsequent runs.

---

## 💡 Tips

- Once models are downloaded, the ENTIRE app works without internet.
- Ask "show me the diagram of X" — the app will search the textbook and display the exact page image.
- The app can solve FCFS, SJF, RR, Banker algorithm, page replacement, and disk scheduling step-by-step.
- Low RAM? The app automatically skips CLIP visual search when Qwen is active to save memory.

---

Good luck! 🎓
