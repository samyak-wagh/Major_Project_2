"""
QA Chain Module
===============
Builds and manages the RAG pipeline using the local fine-tuned
OS-Tutor model (TinyLlama-1.1B + rohit21789/OS-tutor LoRA adapter).

No API keys required — runs fully offline after the first HuggingFace download.
Does NOT use ConversationalRetrievalChain (removed in LangChain 1.x).
Uses a simple custom pipeline that works with any LangChain version.
"""

import logging
import torch
from typing import Dict, Any, List, Optional
from collections import deque

# ── HuggingFace ────────────────────────────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# ── LangChain (only core, no deprecated chains) ────────────────────────────────
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ── System instruction ─────────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "You are an Operating Systems tutor. "
    "Answer ONLY using the exact textbook passages provided below. "
    "Do NOT add your own examples, do NOT invent information. "
    "If the answer is not in the provided text, reply EXACTLY with the word 'OUT_OF_CONTEXT' and nothing else."
)

# Max words per chunk fed to TinyLlama (prevents context window overflow)
_MAX_CHUNK_WORDS = 150
# Number of chunks to retrieve
_TOP_K_CHUNKS = 3


# ══════════════════════════════════════════════════════════════════════════════
#  Main RAG orchestrator — no deprecated LangChain chains
# ══════════════════════════════════════════════════════════════════════════════

class OSTutorChain:
    """
    Simple custom RAG pipeline:
      1. Student question → Qdrant retriever → top-K textbook chunks
      2. Chunks + question → TinyLlama + OS-tutor adapter → grounded answer
      3. Last 5 turns of conversation kept in memory for context
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        adapter: str = "rohit21789/OS-tutor",
        temperature: float = 0.1,
        max_new_tokens: int = 512,
        memory_window: int = 5,
        device: str = "auto",
    ):
        self.retriever = retriever
        self.memory_window = memory_window
        # Store last N (question, answer) pairs for multi-turn context
        self.history: deque = deque(maxlen=memory_window)

        print(f"\n📥 Loading base model: {base_model}")
        print(f"🔧 Applying LoRA adapter: {adapter}")
        print("⏳ This may take a few minutes on first run...\n")

        self.hf_pipeline = self._load_model(
            base_model, adapter, temperature, max_new_tokens, device
        )

        print("✅ OS-Tutor chain is ready!\n")
        logger.info("OSTutorChain initialised.")

    # ── Model Loading ──────────────────────────────────────────────────────────

    def _load_model(
        self,
        base_model_name: str,
        adapter: str,
        temperature: float,
        max_new_tokens: int,
        device: str,
    ):
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── Load base model to CPU first (no device_map) ──────────────────────
        # IMPORTANT: Must load to CPU before applying PEFT adapter.
        # Using device_map="auto" before PeftModel causes KeyError on lm_head
        # in newer versions of peft + accelerate. We move to target device after.
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            device_map=None,          # Load to CPU first
            low_cpu_mem_usage=False,
        )

        # Apply LoRA adapter while model is on CPU
        model = PeftModel.from_pretrained(base, adapter)
        model.eval()

        # Now move to target device if GPU requested
        if device not in ("cpu", None):
            try:
                model = model.to(device)
                print(f"  ✅ Model moved to: {device}")
            except Exception as e:
                print(f"  ⚠ Could not move to {device}: {e}. Running on CPU.")
        else:
            print("  ✅ Model running on: CPU")

        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=min(max_new_tokens, 300),  # cap at 300 to prevent runaway loops
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.9,                  # nucleus sampling — cuts off low-probability tokens
            repetition_penalty=1.4,     # was 1.15 — now strong enough to stop looping
            no_repeat_ngram_size=4,     # prevents repeating any 4-word phrase
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=True,
        )

        logger.info("Model loaded successfully.")
        return hf_pipe

    # ── Core RAG logic ─────────────────────────────────────────────────────────

    def _retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant textbook chunks from Qdrant."""
        try:
            return self.retriever.invoke(question)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return []

    def _build_prompt(self, question: str, context_docs: List[Document]) -> str:
        """Build a simple TinyLlama-formatted prompt with truncated chunks."""
        if context_docs:
            truncated = []
            for doc in context_docs[:_TOP_K_CHUNKS]:
                # Truncate each chunk to _MAX_CHUNK_WORDS words to avoid context overflow
                words = doc.page_content.strip().split()
                truncated.append(" ".join(words[:_MAX_CHUNK_WORDS]))
            context = "\n\n".join(truncated)
        else:
            context = "No relevant information found in the textbook."

        # Simple, strict prompt — no structural markers that confuse TinyLlama
        prompt = (
            f"<|system|>\n{_SYSTEM_INSTRUCTION}\n</s>\n"
            f"<|user|>\n"
            f"Textbook passages:\n{context}\n\n"
            f"Question: {question}\n"
            f"</s>\n"
            f"<|assistant|>\nAnswer:"
        )
        return prompt

    def _generate(self, prompt: str) -> str:
        """Run inference and extract only the assistant reply."""
        outputs = self.hf_pipeline(prompt)
        full_text: str = outputs[0]["generated_text"]

        # Extract only the assistant's reply
        if "<|assistant|>" in full_text:
            answer = full_text.split("<|assistant|>")[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()

        # Strip "Answer:" prefix if model repeated it
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()

        # Clean up trailing special tokens
        for tok in ["</s>", "<|user|>", "<|system|>", "<|assistant|>"]:
            if tok in answer:
                answer = answer.split(tok)[0].strip()

        # ── Cut off when model starts going off-topic or adding hallucinated content
        stop_phrases = [
            # Model asking user questions
            "Would you like",
            "How might",
            "Could you explain",
            "Can you ",
            "Not sure",
            "Maybe something",
            "Let me know",
            "Do you want",
            "Should I ",
            "What do you think",
            "Is there anything",
            "Not really",
            # Hallucinated examples (not from textbook)
            "\nExamples:",
            "\nExample:",
            "\nFor example",
            "\nFor instance",
            "\nSuch as:",
            "\ne.g.",
            # Game theory / off-topic tangents
            "game theory",
            "Game theory",
            "Textbook Information",
            "Query:",
            "Response:",
        ]
        for phrase in stop_phrases:
            if phrase in answer:
                answer = answer[:answer.index(phrase)].strip()
                # Remove trailing incomplete sentence fragments
                for end in [".", "!", "?"]:
                    last = answer.rfind(end)
                    if last > 0:
                        answer = answer[:last + 1]
                        break

        return answer.strip() if answer.strip() else "I could not generate an answer based on the provided context."

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get a grounded answer from the OS textbook.
        Returns dict with keys: answer, source_documents, question.
        """
        try:
            # Step 1: Retrieve relevant passages
            source_docs = self._retrieve(question)

            # Fast-path fallback: If no relevant textbook passages are found at all,
            # don't even bother asking TinyLlama, immediately trigger fallback.
            if not source_docs:
                return {
                    "answer": "OUT_OF_CONTEXT",
                    "source_documents": [],
                    "question": question
                }

            # Step 2: Build prompt
            prompt = self._build_prompt(question, source_docs)

            # Step 3: Generate answer
            answer = self._generate(prompt)

            # Step 4: Save to memory
            self.history.append((question, answer))

            return {
                "answer": answer,
                "source_documents": source_docs,
                "question": question,
            }
        except Exception as e:
            logger.error(f"Error in QA chain: {e}")
            return {
                "answer": f"An error occurred: {str(e)}",
                "source_documents": [],
                "question": question,
            }

    def clear_memory(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info("Conversation memory cleared.")

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def format_source_references(source_docs: List[Document]) -> str:
        """Format retrieved source documents as a readable reference list."""
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


# Backwards-compatibility alias
terminalChain = OSTutorChain
