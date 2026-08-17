"""
main.py — OS Tutor CLI
======================
Terminal REPL for the OS Tutor RAG system.
Supports two LLM backends (set LLM_BACKEND= in .env):
  • local  — TinyLlama + rohit21789/OS-tutor LoRA adapter (fully offline)
  • groq   — Groq cloud API (Llama 3.1, Mixtral, Gemma 2 — fast, free key)

Usage:
  python main.py                    # interactive REPL (auto-loads Galvin textbook)
  python main.py --pdf path/to.pdf  # load a custom PDF then enter REPL

Commands inside the REPL:
  /load <path>   — load & ingest a PDF file
  /list          — list all ingested documents
  /clear         — clear conversation memory
  /reset         — delete entire Qdrant collection
  /help          — show this help
  /quit          — exit
"""

import os
import sys
import time
import logging
import argparse
import textwrap

# Force UTF-8 for Windows terminals
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ── Suppress noisy library logs ───────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
for lib in ("httpx", "httpcore", "qdrant_client", "urllib3", "transformers",
            "peft", "accelerate"):
    logging.getLogger(lib).setLevel(logging.ERROR)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BLUE    = "\033[94m"
WHITE   = "\033[97m"
GREY    = "\033[90m"

def c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}"

def banner(backend: str = "local"):
    print()
    print(c("╔══════════════════════════════════════════════════════════╗", CYAN))
    print(c("║", CYAN) + c("   🎓  OS Tutor  —  Your AI Study Tutor (CLI)            ", BOLD + WHITE) + c("║", CYAN))
    if backend == "groq":
        print(c("║", CYAN) + c("      Groq Cloud API · LangChain · Qdrant               ", DIM + GREY) + c("║", CYAN))
    else:
        print(c("║", CYAN) + c("      TinyLlama + OS-tutor · LangChain · Qdrant          ", DIM + GREY) + c("║", CYAN))
    print(c("╚══════════════════════════════════════════════════════════╝", CYAN))
    print()

def section(title: str):
    print(c(f"\n─── {title} ", BLUE) + c("─" * max(0, 56 - len(title)), GREY))

def info(msg: str):
    print(c("  ℹ ", CYAN) + msg)

def success(msg: str):
    print(c("  ✔ ", GREEN) + msg)

def warn(msg: str):
    print(c("  ⚠ ", YELLOW) + msg)

def error(msg: str):
    print(c("  ✖ ", RED) + msg)

def spinner_dots(label: str):
    """Context manager: print animated dots while work happens."""
    import threading

    class _Spin:
        def __init__(self):
            self._stop = False
            self._t = None

        def _run(self):
            frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
            i = 0
            while not self._stop:
                sys.stdout.write(f"\r  {c(frames[i % len(frames)], CYAN)}  {label}   ")
                sys.stdout.flush()
                time.sleep(0.1)
                i += 1
            sys.stdout.write("\r" + " " * (len(label) + 12) + "\r")
            sys.stdout.flush()

        def __enter__(self):
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
            return self

        def __exit__(self, *_):
            self._stop = True
            self._t.join()

    return _Spin()


# ── Component bootstrap ───────────────────────────────────────────────────────

def load_config() -> dict:
    return {
        "qdrant_url":      os.getenv("QDRANT_URL", ""),
        "qdrant_api_key":  os.getenv("QDRANT_API_KEY", ""),
        "qdrant_host":     os.getenv("QDRANT_HOST", "localhost"),
        "qdrant_port":     int(os.getenv("QDRANT_PORT", "6333")),
        "collection":      os.getenv("QDRANT_COLLECTION", "os_tutor"),
        "chunk_size":      int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap":   int(os.getenv("CHUNK_OVERLAP", "200")),
        "top_k":           int(os.getenv("TOP_K", "5")),
        "score_threshold": float(os.getenv("SCORE_THRESHOLD", "0.35")),
        "temperature":     float(os.getenv("TEMPERATURE", "0.1")),
        "max_new_tokens":  int(os.getenv("MAX_NEW_TOKENS", "512")),
        "local_base":      os.getenv("LOCAL_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "local_adapter":   os.getenv("LOCAL_ADAPTER", "rohit21789/OS-tutor"),
        # Path to the Galvin textbook (auto-ingest on startup if present)
        "default_pdf":     os.getenv("DEFAULT_PDF", "ostxtbook.pdf"),
        # Device: "cpu" = force CPU, "cuda" = force GPU, "auto" = auto-detect
        "device":          os.getenv("DEVICE", "cpu"),
        # ── Groq cloud backend ─────────────────────────────────────────────────
        # Set LLM_BACKEND=groq (or local) in .env to switch backends
        "llm_backend":     os.getenv("LLM_BACKEND", "local"),
        "groq_api_key":    os.getenv("GROQ_API_KEY", ""),
        "groq_model":      os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        # Gemini key (used only by eval/run_eval.py — never removed)
        "gemini_api_key":  os.getenv("GEMINI_PAID_API_KEY", ""),
    }


def init_components(cfg: dict):
    """Instantiate all backend components. Returns (qdrant_store, embeddings, processor)."""
    from rag.document_processor import DocumentProcessor
    from rag.embeddings import LocalEmbeddings
    from rag.vector_store import QdrantStore

    processor = DocumentProcessor(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
    )
    embeddings = LocalEmbeddings()

    qdrant_store = QdrantStore(
        host=cfg["qdrant_host"],
        port=cfg["qdrant_port"],
        collection_name=cfg["collection"],
        embedding_dim=embeddings.get_embedding_dimension(),
        url=cfg["qdrant_url"] or None,
        api_key=cfg["qdrant_api_key"] or None,
    )
    return qdrant_store, embeddings, processor


def build_qa_chain(cfg: dict, qdrant_store, embeddings,
                   backend: Optional[str] = None,
                   groq_model: Optional[str] = None):
    """
    Build a QA chain from the current vector store state.

    backend: 'groq' | 'local' | None (None → use cfg['llm_backend'])
    groq_model: Groq model name override (e.g. 'mixtral-8x7b-32768')
    """
    resolved_backend = (backend or cfg.get("llm_backend", "local")).lower()

    query_emb = embeddings.get_query_embeddings_instance()
    vector_store = qdrant_store.get_vector_store(query_emb)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": cfg["top_k"],
            "score_threshold": cfg["score_threshold"],
        },
    )

    if resolved_backend == "groq":
        from rag.groq_chain import GroqOSTutorChain
        chain = GroqOSTutorChain(
            retriever=retriever,
            api_key=cfg.get("groq_api_key", ""),
            model=groq_model or cfg.get("groq_model", "llama-3.3-70b-versatile"),
            temperature=cfg["temperature"],
            max_tokens=cfg["max_new_tokens"],
        )
    else:
        from rag.qa_chain import OSTutorChain
        chain = OSTutorChain(
            retriever=retriever,
            base_model=cfg["local_base"],
            adapter=cfg["local_adapter"],
            temperature=cfg["temperature"],
            max_new_tokens=cfg["max_new_tokens"],
            device=cfg.get("device", "auto"),
        )
    return chain


# ── PDF ingestion ─────────────────────────────────────────────────────────────

def ingest_pdf(pdf_path: str, cfg: dict, qdrant_store, embeddings, processor) -> bool:
    """Load, chunk, embed and store a PDF. Returns True on success."""
    path = Path(pdf_path)
    if not path.exists():
        error(f"File not found: {pdf_path}")
        return False
    if path.suffix.lower() != ".pdf":
        error(f"Not a PDF file: {pdf_path}")
        return False

    section(f"Ingesting  {path.name}")

    with spinner_dots(f"Reading & chunking  {path.name}"):
        result = processor.process_pdf_path(str(path))

    total_pages = result["total_pages"]
    text_pages  = result["text_pages"]
    chunks      = result["total_chunks"]
    chars       = result["extracted_chars"]
    fhash       = result["file_hash"]

    info(f"Pages: {c(str(total_pages), YELLOW)}  "
         f"Text pages: {c(str(text_pages), YELLOW)}  "
         f"Chunks: {c(str(chunks), YELLOW)}  "
         f"Characters: {c(str(chars), YELLOW)}")

    if chunks == 0:
        error(f"No text could be extracted from  {path.name}")
        if text_pages == 0:
            warn("All pages appear to be images (scanned PDF).")
            warn("Please use a text-based PDF, or run it through an OCR tool first.")
        else:
            warn(f"Extracted {chars} characters — content may be too short to chunk.")
        return False

    if qdrant_store.document_exists(fhash):
        warn(f"{path.name} is already in the knowledge base — skipping.")
        return True

    with spinner_dots(f"Embedding {chunks} chunks via HuggingFace …"):
        qdrant_store.add_documents(result["chunks"], embeddings.embeddings)

    success(f"Ingested {c(path.name, GREEN)}  →  {chunks} vectors stored in Qdrant.")
    return True


# ── Help text ─────────────────────────────────────────────────────────────────

HELP_TEXT = f"""
Available commands:

  /load <path>   Load and ingest a PDF file into the knowledge base
  /list          List all documents currently in the knowledge base
  /clear         Clear conversation memory (start fresh context)
  /reset         Delete the entire Qdrant collection (removes all PDFs)
  /help          Show this help message
  /quit          Exit OS Tutor

Just type any question to query your loaded documents.
"""

# ── REPL ──────────────────────────────────────────────────────────────────────

def run_repl(cfg: dict, preload_pdfs: list):
    banner(backend=cfg.get("llm_backend", "local"))

    section("Connecting to Qdrant")
    qdrant_target = cfg["qdrant_url"] or f"{cfg['qdrant_host']}:{cfg['qdrant_port']}"
    with spinner_dots(f"Connecting to {qdrant_target} …"):
        try:
            qdrant_store, embeddings, processor = init_components(cfg)
        except Exception as e:
            error(f"Could not connect to Qdrant: {e}")
            error("Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
            sys.exit(1)

    success(f"Qdrant connected  —  collection: {c(cfg['collection'], YELLOW)}")

    # ── Auto-ingest default textbook if knowledge base is empty ───────────────
    docs = qdrant_store.list_ingested_documents()
    if not docs:
        default_pdf = cfg.get("default_pdf", "ostxtbook.pdf")
        if Path(default_pdf).exists():
            info(f"Knowledge base is empty. Auto-ingesting: {c(default_pdf, YELLOW)}")
            ingest_pdf(default_pdf, cfg, qdrant_store, embeddings, processor)
        else:
            warn("Knowledge base is empty. Use /load <path.pdf> to add a document.")

    # ── Pre-load any PDFs passed via CLI ─────────────────────────────────────
    for pdf_path in preload_pdfs:
        ingest_pdf(pdf_path, cfg, qdrant_store, embeddings, processor)

    # ── Build QA chain ────────────────────────────────────────────────────────
    qa_chain = None
    docs = qdrant_store.list_ingested_documents()
    if docs:
        backend = cfg.get("llm_backend", "local")
        if backend == "groq":
            section("Initialising Groq Chain")
            info(f"Backend: {c('Groq', CYAN)}  Model: {c(cfg.get('groq_model', 'llama-3.3-70b-versatile'), YELLOW)}")
        else:
            section("Loading OS-Tutor Model")
            info("Loading TinyLlama + OS-tutor adapter (first run downloads ~2.2 GB)…")
        qa_chain = build_qa_chain(cfg, qdrant_store, embeddings)
        success(f"Ready — {len(docs)} document(s) in knowledge base.")
        for d in docs:
            info(f"  {c('📄', WHITE)} {d}")
    else:
        warn("No documents loaded yet. Use /load <path.pdf> to add one.")

    print()
    print(c("Type /help for available commands, /quit to exit.", DIM + GREY))
    print()

    query_count = 0

    while True:
        try:
            user_input = input(c("You ▶ ", BOLD + GREEN)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            success("Goodbye! 👋")
            break

        if not user_input:
            continue

        cmd, _, args = user_input.partition(" ")
        cmd = cmd.lower()

        if cmd in ("/quit", "/exit", "/q"):
            success("Goodbye! 👋")
            break

        elif cmd == "/help":
            print(HELP_TEXT)

        elif cmd == "/list":
            docs = qdrant_store.list_ingested_documents()
            if docs:
                section("Knowledge Base")
                for d in docs:
                    info(f"📄 {d}")
                col_info = qdrant_store.get_collection_info()
                info(f"Total vectors: {c(str(col_info.get('points_count', '?')), YELLOW)}")
            else:
                warn("Knowledge base is empty. Use /load <path.pdf> to add documents.")

        elif cmd == "/load":
            pdf_path = args.strip().strip('"').strip("'")
            if not pdf_path:
                error("Usage: /load <path/to/file.pdf>")
            else:
                ok = ingest_pdf(pdf_path, cfg, qdrant_store, embeddings, processor)
                if ok:
                    if qa_chain is None:
                        section("Loading OS-Tutor Model")
                        qa_chain = build_qa_chain(cfg, qdrant_store, embeddings)
                    success("Document added. You can now ask questions about it.")

        elif cmd == "/clear":
            if qa_chain:
                qa_chain.clear_memory()
                success("Conversation memory cleared.")
            else:
                warn("No active conversation to clear.")

        elif cmd == "/reset":
            confirm = input(
                c("  ⚠  This will delete ALL vectors. Type ", YELLOW)
                + c("yes", BOLD + RED)
                + c(" to confirm: ", YELLOW)
            ).strip().lower()
            if confirm == "yes":
                with spinner_dots("Deleting collection …"):
                    qdrant_store.delete_collection()
                qa_chain = None
                success("Collection deleted. Use /load to add new documents.")
            else:
                info("Reset cancelled.")

        elif cmd.startswith("/"):
            warn(f"Unknown command: {cmd}   (type /help for commands)")

        # ── Normal question ──────────────────────────────────────────────────
        else:
            if qa_chain is None:
                warn("No documents loaded. Use /load <path.pdf> first.")
                continue

            query_count += 1
            section(f"Query #{query_count}")

            t_start = time.perf_counter()
            with spinner_dots("Searching Qdrant + generating answer …"):
                response = qa_chain.ask(user_input)
            elapsed = time.perf_counter() - t_start

            answer = response["answer"]
            source_docs = response["source_documents"]

            print()
            print(c("  OS-Tutor ▶", BOLD + CYAN))
            print()
            wrapped = textwrap.fill(
                answer, width=76, initial_indent="    ", subsequent_indent="    "
            )
            print(c(wrapped, WHITE))
            print()

            if source_docs:
                print(c("  📚 Sources", BOLD + MAGENTA))
                from rag.qa_chain import OSTutorChain
                print(c(OSTutorChain.format_source_references(source_docs), DIM + GREY))

            print()
            print(c(f"  ⚡ {elapsed:.2f}s", DIM + GREY))
            print()


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OS Tutor — Terminal RAG with TinyLlama + OS-tutor & Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python main.py                        # auto-loads ostxtbook.pdf
          python main.py --pdf my_notes.pdf     # load a custom PDF
          python main.py --pdf doc1.pdf --pdf doc2.pdf
        """),
    )
    parser.add_argument("--pdf", action="append", default=[], metavar="FILE",
                        help="PDF file(s) to load on startup (can repeat)")
    parser.add_argument("--top-k", type=int, default=None, metavar="N",
                        help="Number of context chunks to retrieve (default: 5)")
    parser.add_argument("--collection", default=None, metavar="NAME",
                        help="Qdrant collection name (default: os_tutor)")
    parser.add_argument("--temperature", type=float, default=None, metavar="T",
                        help="LLM temperature 0.0–1.0 (default: 0.1)")

    args = parser.parse_args()

    cfg = load_config()
    if args.top_k:       cfg["top_k"] = args.top_k
    if args.collection:  cfg["collection"] = args.collection
    if args.temperature is not None: cfg["temperature"] = args.temperature

    run_repl(cfg, preload_pdfs=args.pdf)


if __name__ == "__main__":
    main()
