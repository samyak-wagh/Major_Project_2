"""
groq_chain.py — Groq-powered RAG Chain for OS Tutor
=====================================================
Drops in as a replacement for OSTutorChain (same .ask() interface)
but sends requests to Groq's ultra-fast inference API instead of
running a local HuggingFace model.

Supported Groq models (free tier):
  • llama-3.1-70b-versatile   (default, best quality)
  • llama-3.1-8b-instant      (faster, lighter)
  • llama3-70b-8192
  • mixtral-8x7b-32768
  • gemma2-9b-it

Get a free API key at: https://console.groq.com
"""

import logging
import os
from typing import Dict, Any, List
from collections import deque

from groq import Groq
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── System instruction ───────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "You are an expert Operating Systems tutor. "
    "When textbook passages are provided, use them as your PRIMARY source and cite them. "
    "If the textbook does not contain enough information, use your general knowledge to give a complete, accurate answer. "
    "NEVER reply with 'OUT_OF_CONTEXT'. Always provide a helpful answer. "
    "Keep answers clear, structured, and educational. "
    "CRITICAL RULE: If the user asks for a diagram, DO NOT attempt to draw large ASCII art diagrams. "
    "Instead, briefly describe the diagram in 1-2 sentences. If the provided textbook text explicitly mentions a figure number (like Figure X.Y), state it. If the text does not mention a figure number, DO NOT GUESS OR INVENT ONE."
)

# Words per chunk fed to Groq (Groq models have large context windows, so we
# can afford more than TinyLlama's 150 words)
_MAX_CHUNK_WORDS = 400
_TOP_K_CHUNKS = 5


class GroqOSTutorChain:
    """
    RAG pipeline backed by Groq cloud inference.

    Interface is identical to OSTutorChain so both can be used
    interchangeably throughout api.py and main.py.

      1. Question → Qdrant retriever → top-K textbook chunks
      2. Chunks + question → Groq API → grounded answer
      3. Last 5 turns kept in memory for multi-turn context
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        api_key: str = "",
        model: str = "llama-3.1-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        memory_window: int = 5,
    ):
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history: deque = deque(maxlen=memory_window)

        # Resolve API key: param > env var
        resolved_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and add it to .env"
            )

        self.client = Groq(api_key=resolved_key)

        print(f"\n⚡ Groq chain initialised — model: {self.model}")
        logger.info("GroqOSTutorChain initialised with model=%s", self.model)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant textbook chunks from Qdrant."""
        try:
            return self.retriever.invoke(question)
        except Exception as e:
            logger.error("Groq chain retrieval error: %s", e)
            return []

    # ── Prompt building ────────────────────────────────────────────────────────

    def _build_messages(self, question: str, context_docs: List[Document]) -> list:
        """Build the chat messages list for the Groq API."""
        if context_docs:
            truncated = []
            for doc in context_docs[:_TOP_K_CHUNKS]:
                words = doc.page_content.strip().split()
                truncated.append(" ".join(words[:_MAX_CHUNK_WORDS]))
            context = "\n\n".join(truncated)
        else:
            context = "No relevant information found in the textbook."

        # Include last N turns of conversation history for multi-turn context
        history_text = ""
        if self.history:
            turns = []
            for q, a in self.history:
                turns.append(f"Student: {q}\nTutor: {a}")
            history_text = "\n\n".join(turns) + "\n\n"

        user_content = (
            f"Textbook passages:\n{context}\n\n"
            f"{history_text}"
            f"Question: {question}"
        )

        return [
            {"role": "system", "content": _SYSTEM_INSTRUCTION},
            {"role": "user",   "content": user_content},
        ]

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(self, messages: list) -> str:
        """Call Groq API and return the assistant's reply."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Groq API error: %s", e)
            raise

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer from the OS textbook.
        Returns dict with keys: answer, source_documents, question.
        (Same interface as OSTutorChain.ask())
        """
        try:
            source_docs = self._retrieve(question)

            # Fast-path fallback: If no relevant textbook passages are found at all,
            # don't even bother asking the model, immediately trigger fallback.
            if not source_docs:
                return {
                    "answer": "OUT_OF_CONTEXT",
                    "source_documents": [],
                    "question": question
                }

            messages    = self._build_messages(question, source_docs)
            answer      = self._generate(messages)

            self.history.append((question, answer))

            return {
                "answer":           answer,
                "source_documents": source_docs,
                "question":         question,
            }
        except Exception as e:
            logger.error("Error in GroqOSTutorChain.ask: %s", e)
            return {
                "answer":           f"Groq API error: {str(e)}",
                "source_documents": [],
                "question":         question,
            }

    def clear_memory(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info("GroqOSTutorChain memory cleared.")

    @staticmethod
    def format_source_references(source_docs: List[Document]) -> str:
        """Same helper as OSTutorChain for formatting source references."""
        if not source_docs:
            return "  (no source references)"
        refs = []
        seen = set()
        for doc in source_docs:
            meta = doc.metadata
            filename = meta.get("source_filename") or meta.get("source", "Unknown")
            page = meta.get("page", "?")
            key = f"{filename}-{page}"
            if key not in seen:
                seen.add(key)
                page_num = page + 1 if isinstance(page, int) else page
                refs.append(f"  • {filename}  —  page {page_num}")
        return "\n".join(refs)


# Available Groq models exposed for UI dropdowns
# These are the models currently active on the Groq platform
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Best quality)",
    "llama-3.1-8b-instant":    "Llama 3.1 8B (Fastest)",
    "qwen/qwen3.6-27b":        "Qwen 3.6 27B",
    "openai/gpt-oss-120b":     "GPT-OSS 120B (OpenAI)",
    "openai/gpt-oss-20b":      "GPT-OSS 20B (OpenAI)",
    "groq/compound":           "Groq Compound",
    "groq/compound-mini":      "Groq Compound Mini",
}
