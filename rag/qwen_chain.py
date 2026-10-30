"""
qwen_chain.py — Qwen2.5-1.5B + Fine-tuned LoRA Chain for OS Tutor
==================================================================
PRIMARY local model for the RAG pipeline.

Fine-tuned by Yash Sharma using TRL + Unsloth SFT on the OS-Tutor dataset.
  Base model : Qwen/Qwen2.5-1.5B-Instruct  (loaded in fp32/fp16 — no bitsandbytes)
  LoRA adapter: qwen-os-tutor-lora/          (local directory, checked into the repo)
  Chat format : Qwen ChatML  (<|im_start|> / <|im_end|>)

Interface is identical to OSTutorChain and GroqOSTutorChain so all three can be
used interchangeably throughout api.py and main.py.
"""

import logging
import os
from typing import Dict, Any, List
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── System instruction ───────────────────────────────────────────────────────
_SYSTEM_INSTRUCTION = (
    "You are a concise Operating Systems tutor based on the Galvin textbook. "
    "Answer DIRECTLY and SPECIFICALLY. "
    "Do NOT ramble, speculate, or go off-topic. "
    "If textbook passages are provided, base your answer strictly on them. "
    "Keep answers short: 2-5 sentences maximum unless a step-by-step solution is needed. "
    "For numerical problems, show only the key steps and the final answer. "
    "NEVER invent figures, page numbers, or facts not in the context. "
    "NEVER ask follow-up questions back to the student."
)

# The exact system prompt the adapter was fine-tuned with (see dataset_v2*.jsonl).
# Selectable per request with opts={"prompt": "train"} so the train/inference
# prompt mismatch can be tested without reloading the model.
_TRAIN_SYSTEM_INSTRUCTION = (
    "You are an Operating Systems tutor. Answer ONLY using the exact textbook passages "
    "provided below. Do NOT add your own examples, do NOT invent information. "
    "If the answer is not in the provided text, reply EXACTLY with the word "
    "'OUT_OF_CONTEXT' and nothing else."
)

# Qwen2.5-1.5B has a 32K context window — we can afford more than TinyLlama
_MAX_CHUNK_WORDS = 300
_TOP_K_CHUNKS    = 4

# Default base model — full-precision, no bitsandbytes required (Windows safe)
_DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class QwenOSTutorChain:
    """
    RAG pipeline backed by Qwen2.5-1.5B-Instruct + OS-Tutor LoRA adapter.

    Pipeline:
      1. Question → Qdrant retriever → top-K textbook chunks
      2. Chunks + question → Qwen2.5 ChatML prompt → grounded answer
      3. Last 5 turns kept in memory for multi-turn context

    The base model is loaded in fp32 (CPU) or fp16 (CUDA) — no bitsandbytes
    required, so it works on Windows out of the box.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        adapter_path: str = "qwen-os-tutor-lora",
        base_model: str = _DEFAULT_BASE_MODEL,
        temperature: float = 0.05,
        max_new_tokens: int = 256,
        memory_window: int = 5,
        device: str = "auto",
    ):
        self.retriever     = retriever
        self.temperature   = temperature
        self.max_new_tokens = max_new_tokens
        self.history: deque = deque(maxlen=memory_window)

        print(f"\n📥 Loading base model  : {base_model}")
        print(f"🔧 Applying LoRA adapter: {adapter_path}")
        print("⏳ This may take a few minutes on first run...\n")

        self.hf_pipeline = self._load_model(
            base_model, adapter_path, temperature, max_new_tokens, device
        )

        print("✅ Qwen OS-Tutor chain is ready!\n")
        logger.info(
            "QwenOSTutorChain initialised — base=%s  adapter=%s",
            base_model, adapter_path
        )

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_model(
        self,
        base_model_name: str,
        adapter_path: str,
        temperature: float,
        max_new_tokens: int,
        device: str,
    ):
        # ── 1. Resolve target device ──────────────────────────────────────────
        if device in ("auto", "cuda"):
            use_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            use_device = device   # "cpu" or explicit "cuda:0" etc.

        dtype = torch.float16 if use_device == "cuda" else torch.float32
        print(f"  📦 Target device: {use_device}  dtype: {dtype}")

        # ── 2. Load tokenizer from the adapter directory ──────────────────────
        # The adapter directory includes the fine-tuned tokenizer_config.json
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # ── 3. Load base model to CPU first (safe for PEFT) ──────────────────
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map=None,          # CPU first — move after PEFT is applied
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # ── 4. Apply the fine-tuned LoRA adapter ─────────────────────────────
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()

        # ── 5. Move to target device ──────────────────────────────────────────
        if use_device != "cpu":
            try:
                model = model.to(use_device)
                print(f"  ✅ Model moved to: {use_device}")
            except Exception as e:
                print(f"  ⚠ Could not move to {use_device}: {e}. Falling back to CPU.")
        else:
            print("  ✅ Model running on: CPU")

        # ── 6. Build HuggingFace pipeline ─────────────────────────────────────
        # return_full_text=False → pipeline returns ONLY the generated tokens (not the prompt)
        hf_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=min(max_new_tokens, 256),
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.85,
            # Mild penalty only: HF applies repetition_penalty / no_repeat_ngram_size to the
            # PROMPT too, so strong values forbid quoting the retrieved textbook passages
            # (word-salad answers, near-zero faithfulness in RAG mode).
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )

        logger.info("Qwen model loaded successfully on %s.", use_device)
        return hf_pipe

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _retrieve(self, question: str) -> List[Document]:
        """Retrieve relevant textbook chunks from Qdrant."""
        try:
            return self.retriever.invoke(question)
        except Exception as e:
            logger.error("Qwen chain retrieval error: %s", e)
            return []

    # ── Prompt building ────────────────────────────────────────────────────────

    def _build_messages(self, question: str, context_docs: List[Document],
                        prompt_mode: str = "concise") -> list:
        """Build Qwen ChatML messages list."""
        if context_docs:
            truncated = []
            for doc in context_docs[:_TOP_K_CHUNKS]:
                words = doc.page_content.strip().split()
                truncated.append(" ".join(words[:_MAX_CHUNK_WORDS]))
            context = "\n\n".join(truncated)
        else:
            context = "No relevant information found in the textbook."

        # Include last N conversation turns for multi-turn context
        history_text = ""
        if self.history:
            turns = [f"Student: {q}\nTutor: {a}" for q, a in self.history]
            history_text = "\n\n".join(turns) + "\n\n"

        user_content = (
            f"Textbook passages:\n{context}\n\n"
            f"{history_text}"
            f"Question: {question}"
        )

        # Qwen ChatML format — the pipeline applies the chat template automatically
        return [
            {"role": "system",
             "content": _TRAIN_SYSTEM_INSTRUCTION if prompt_mode == "train" else _SYSTEM_INSTRUCTION},
            {"role": "user",   "content": user_content},
        ]

    # ── Generation ─────────────────────────────────────────────────────────────

    def _generate(self, messages: list, greedy: bool = False,
                  rep_penalty: float = None) -> str:
        """Run inference and return the assistant reply.

        greedy=True      -> deterministic decoding (no sampling)
        rep_penalty=<x>  -> override the pipeline's repetition penalty for this call
        """
        gen_kwargs = {}
        if greedy:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
        if rep_penalty is not None:
            gen_kwargs["repetition_penalty"] = rep_penalty
        outputs = self.hf_pipeline(messages, **gen_kwargs)
        answer: str = outputs[0]["generated_text"].strip()

        # Strip any leaked special tokens
        for tok in ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
            if tok in answer:
                answer = answer.split(tok)[0].strip()

        # Cut off hallucinated content / model asking questions back
        stop_phrases = [
            "Would you like",
            "How might",
            "Could you explain",
            "Can you ",
            "Let me know",
            "Do you want",
            "Should I ",
            "Is there anything",
            "\nExamples:",
            "\nExample:",
            "\nFor example",
            "\nFor instance",
            "\ne.g.",
            "game theory",
            "Game theory",
            "Query:",
            "Response:",
            "Textbook Information",
            # Anti-ramble additions
            "In particular:",
            "Note too that",
            "a) Giving",
            "b) Explaining",
            "It's worth noting",
            "That said,",
            "However, it's important",
            "This is because",
            "As mentioned",
            "To summarize,",
            "In conclusion,",
            "modern systems often",
            "early operating-system",
            "back before everything",
            "rabbit holes",
            "outdated terminology",
        ]
        for phrase in stop_phrases:
            if phrase in answer:
                answer = answer[: answer.index(phrase)].strip()
                for end in [".", "!", "?"]:
                    last = answer.rfind(end)
                    if last > 0:
                        answer = answer[: last + 1]
                        break

        return (
            answer.strip()
            if answer.strip()
            else "I could not generate an answer based on the provided context."
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, question: str, use_rag: bool = True, opts: dict = None) -> Dict[str, Any]:
        """
        Ask a question.
        use_rag=True  → retrieve textbook chunks then generate (normal RAG mode)
        use_rag=False → generate with NO context (fine-tuned LLM-only baseline)
        Returns dict with keys: answer, source_documents, question.
        """
        try:
            source_docs = self._retrieve(question) if use_rag else []
            opts     = opts or {}
            messages = self._build_messages(question, source_docs, opts.get("prompt", "concise"))
            answer   = self._generate(messages, greedy=bool(opts.get("greedy")),
                                      rep_penalty=opts.get("rep_penalty"))

            if answer and "OUT_OF_CONTEXT" not in answer:
                self.history.append((question, answer))

            return {
                "answer":           answer,
                "source_documents": source_docs,
                "question":         question,
            }
        except Exception as e:
            logger.error("Error in QwenOSTutorChain.ask: %s", e)
            # Signal to api.py to fall back to Groq
            return {
                "answer":           "OUT_OF_CONTEXT",
                "source_documents": [],
                "question":         question,
            }

    def clear_memory(self):
        """Clear conversation history."""
        self.history.clear()
        logger.info("QwenOSTutorChain memory cleared.")

    @staticmethod
    def format_source_references(source_docs: List[Document]) -> str:
        """Format retrieved source documents as a readable reference list."""
        if not source_docs:
            return "  (no source references)"
        refs = []
        seen = set()
        for doc in source_docs:
            meta     = doc.metadata
            filename = meta.get("source_filename") or meta.get("source", "Unknown")
            page     = meta.get("page", "?")
            key      = f"{filename}-{page}"
            if key not in seen:
                seen.add(key)
                page_num = page + 1 if isinstance(page, int) else page
                refs.append(f"  • {filename}  —  page {page_num}")
        return "\n".join(refs)
