"""
api.py — FastAPI Backend for OS Tutor
LLM Priority Order:
  1. Qwen2.5-1.5B + Yash LoRA (qwen-os-tutor-lora)  — PRIMARY local model
  2. Groq cloud API (openai/gpt-oss-20b)         — secondary fallback
  3. Gemini                                           — eval only (never shown to user)
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn, os, logging
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
for lib in ("transformers", "peft", "accelerate", "httpx", "urllib3"):
    logging.getLogger(lib).setLevel(logging.ERROR)

from main import load_config, init_components, build_qa_chain, ingest_pdf

app = FastAPI(title="OS Tutor API", version="2.0.0")


class RAGState:
    def __init__(self):
        print("\n🚀 Initializing OS Tutor backend...")
        self.cfg = load_config()
        self.qdrant_store, self.embeddings, self.processor = init_components(self.cfg)

        # Chain slots — two-tier priority
        self.qwen_chain  = None   # Qwen2.5-1.5B + Yash LoRA   — PRIMARY
        self.local_chain = None   # (Deprecated)
        self.groq_chain  = None   # Groq cloud                  — secondary fallback

        # Tracks which local model is live
        self.qwen_ok  = False
        self.local_ok = False   # Kept for legacy compatibility

        # Auto-ingest Galvin textbook if knowledge base is empty
        docs = self.qdrant_store.list_ingested_documents()
        if not docs:
            default_pdf = self.cfg.get("default_pdf", "ostxtbook.pdf")
            if Path(default_pdf).exists():
                print(f"\n📖 Auto-ingesting: {default_pdf}")
                ingest_pdf(default_pdf, self.cfg, self.qdrant_store,
                           self.embeddings, self.processor)
            else:
                print(f"\n⚠  '{default_pdf}' not found. Upload via /upload endpoint.")

        self._boot_local_chain()

    def _boot_local_chain(self):
        """Boot the primary local chain (Qwen2.5)."""
        docs = self.qdrant_store.list_ingested_documents()
        if not docs:
            return

        # ── 1. Try Qwen2.5 + Yash LoRA (PRIMARY) ───────────────────────────
        try:
            print("\n🤖 Loading Qwen2.5-1.5B + OS-Tutor LoRA (primary model)...")
            print("   (First run downloads ~3 GB — cached after that)")
            self.qwen_chain = build_qa_chain(self.cfg, self.qdrant_store,
                                             self.embeddings, backend="qwen")
            self.qwen_ok = True
            print("✅ Qwen2.5 primary model ready.\n")
        except Exception as e:
            print(f"\n⚠️  Qwen chain failed to load: {e}")
            print("   ➡  Groq will be used as the secondary fallback.\n")
            self.qwen_ok = False

    def _get_groq_chain(self):
        """Lazily initialise the Groq cloud fallback chain."""
        if self.groq_chain is None:
            print("⚡ Initialising Groq cloud fallback chain...")
            self.groq_chain = build_qa_chain(self.cfg, self.qdrant_store,
                                             self.embeddings, backend="groq")
            print("⚡ Groq fallback ready.")
        return self.groq_chain

    def _build_chain_if_ready(self):
        """Legacy helper kept for the /upload endpoint."""
        if not self.qwen_ok and not self.local_ok:
            self._boot_local_chain()
        if not self.qwen_ok and not self.local_ok:
            self._get_groq_chain()


_rag_state: RAGState = None

def get_rag_state() -> RAGState:
    global _rag_state
    if _rag_state is None:
        _rag_state = RAGState()
    return _rag_state


class QuestionRequest(BaseModel):
    question:   str
    use_rag:    bool = True         # kept for eval script compatibility
    backend:    str  = ""           # override: "groq" | "local" | "" (auto = local→groq)
    groq_model: str  = ""           # override Groq model name


@app.get("/status")
async def get_status():
    state = get_rag_state()
    docs  = state.qdrant_store.list_ingested_documents()

    if state.qwen_ok:
        model_info     = "Qwen2.5-1.5B + qwen-os-tutor-lora (local — primary)"
        active_backend = "qwen"
    else:
        model_info     = f"Groq / {state.cfg.get('groq_model', 'openai/gpt-oss-20b')} (cloud — secondary fallback)"
        active_backend = "groq"

    return {
        "status":        "ok",
        "model":         model_info,
        "backend":       active_backend,
        "qwen_ok":       state.qwen_ok,
        "local_ok":      state.local_ok,
        "documents":     docs,
        "chain_ready":   state.qwen_ok or state.local_ok or state.groq_chain is not None,
    }


@app.post("/ask")
def ask_question(req: QuestionRequest):
    state = get_rag_state()

    if not state.qdrant_store.list_ingested_documents():
        raise HTTPException(status_code=400,
                            detail="Knowledge base empty. Upload a PDF first.")

    # ── Step 1: Choose which chain to call ───────────────────────────────────────
    # Manual override from UI (e.g. user explicitly picks Groq in sidebar)
    force_backend = req.backend.lower() if req.backend else ""

    if force_backend == "groq":
        # Caller explicitly wants Groq — honour that
        chain = state._get_groq_chain()
        if req.groq_model and req.groq_model != getattr(chain, "model", ""):
            chain = build_qa_chain(state.cfg, state.qdrant_store, state.embeddings,
                                   backend="groq", groq_model=req.groq_model)
            state.groq_chain = chain
        response       = chain.ask(req.question)
        active_backend = "groq"

    elif force_backend in ("qwen", ""):
        # Default path: Qwen2.5 (primary) → Groq (cloud secondary)
        if state.qwen_ok and state.qwen_chain:
            try:
                response       = state.qwen_chain.ask(req.question)
                active_backend = "qwen"
            except Exception as qwen_err:
                import logging as _log
                _log.getLogger(__name__).warning(
                    "Qwen chain error — falling back to Groq: %s", qwen_err)
                state.qwen_ok = False
                response       = state._get_groq_chain().ask(req.question)
                active_backend = "groq (auto-fallback)"
        else:
            response       = state._get_groq_chain().ask(req.question)
            active_backend = "groq (local unavailable)"

    else:
        # force_backend == "local" (legacy TinyLlama request)
        response       = state._get_groq_chain().ask(req.question)
        active_backend = "groq (legacy override ignored)"

    # ── Step 2: Pick the right chain for source formatting ────────────────────
    if active_backend == "qwen":
        fmt_chain = state.qwen_chain
    else:
        fmt_chain = state.groq_chain

    # ── Step 3: Out-of-context → Groq fallback (general knowledge) ───────────
    # NOTE: Gemini fallback code is preserved in rag/gemini_fallback.py
    #       and can be re-enabled by swapping the block below.
    answer = response.get("answer", "")
    sources, contexts = [], []

    if "OUT_OF_CONTEXT" in answer:
        # Route out-of-context questions to Groq for a general knowledge answer
        groq_chain = state._get_groq_chain()
        groq_resp  = groq_chain.ask_general(req.question)
        answer     = groq_resp.get("answer", "I couldn't find an answer.")
        # No textbook sources since this came from Groq general knowledge
        # -- Gemini fallback (kept for reference, currently disabled) --
        # from rag.gemini_fallback import ask_gemini
        # answer = ask_gemini(req.question)
    else:
        if response.get("source_documents") and fmt_chain:
            src_text = fmt_chain.format_source_references(response["source_documents"])
            sources  = [s.strip() for s in src_text.split("\n") if s.strip()]
            contexts = [doc.page_content for doc in response["source_documents"]]

    return {"answer": answer, "sources": sources, "contexts": contexts}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    state = get_rag_state()
    tmp = f"temp_{file.filename}"
    try:
        with open(tmp, "wb") as f:
            f.write(file.file.read())
        ok = ingest_pdf(tmp, state.cfg, state.qdrant_store,
                        state.embeddings, state.processor)
        if ok:
            state._build_chain_if_ready()
            return {"message": f"Successfully ingested '{file.filename}'."}
        raise HTTPException(status_code=500, detail="Ingestion failed. Check server logs.")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    get_rag_state()
    uvicorn.run(app, host="0.0.0.0", port=8000)
