"""
build_conceptual.py — Expand kb_conceptual.py into fine-tuning examples.
=========================================================================
Produces chat-format examples that mirror EXACTLY the prompt shape used at
inference time by rag/qa_chain.py (_SYSTEM_INSTRUCTION + "Textbook
passages:\n{context}\n\nQuestion: {question}"), so the fine-tuning
distribution matches the production distribution.

Each example:
{
  "messages": [
    {"role": "system", "content": SYSTEM_INSTRUCTION},
    {"role": "user", "content": "Textbook passages:\n<context>\n\nQuestion: <question>"},
    {"role": "assistant", "content": "<direct, correct answer>"}
  ]
}

Also emits a small set of OUT_OF_CONTEXT examples (an OS textbook context
paired with an unrelated, off-syllabus question) so the model retains the
ability to correctly refuse when the retrieved context doesn't answer the
question — this is required behaviour at inference time (see qa_chain.py's
system instruction) and was completely absent from the old dataset.
"""

import random

from kb_conceptual import CHAPTERS as _CORE_CHAPTERS, GLOSSARY as _CORE_GLOSSARY
from kb_advanced import CHAPTERS as _ADVANCED_CHAPTERS, GLOSSARY as _ADVANCED_GLOSSARY

# kb_conceptual.py = direct textbook recall (definitions, "compare X and Y").
# kb_advanced.py   = applied, multi-concept, exam-synthesis scenarios with
#                     full reasoning-chain answers (hardware memory models,
#                     priority inversion traces, real-time scheduling math,
#                     COW/security mechanics). Merged so both feed the same
#                     pipeline below with no special-casing.
CHAPTERS = _CORE_CHAPTERS + _ADVANCED_CHAPTERS
GLOSSARY = _CORE_GLOSSARY + _ADVANCED_GLOSSARY

# Must match rag/qa_chain.py::_SYSTEM_INSTRUCTION and rag/groq_chain.py exactly.
SYSTEM_INSTRUCTION = (
    "You are an Operating Systems tutor. "
    "Answer ONLY using the exact textbook passages provided below. "
    "Do NOT add your own examples, do NOT invent information. "
    "If the answer is not in the provided text, reply EXACTLY with the word "
    "'OUT_OF_CONTEXT' and nothing else."
)

GLOSSARY_PHRASINGS = [
    "What is {term}?",
    "Define {term}.",
    "Explain the term {term} in the context of operating systems.",
]

# Off-syllabus questions paired with real OS contexts, to teach OUT_OF_CONTEXT.
OFF_SYLLABUS_QUESTIONS = [
    "What is the boiling point of water at sea level?",
    "Who won the FIFA World Cup in 2018?",
    "What is the capital of Australia?",
    "How do you bake a chocolate cake?",
    "What is the chemical formula for table salt?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the speed of light in a vacuum?",
    "How many players are on a cricket team?",
    "What year did World War II end?",
    "What is the square root of 256?",
    "What is the tallest mountain in the world?",
    "How do you convert Celsius to Fahrenheit?",
    "What is the currency used in Japan?",
    "Who painted the Mona Lisa?",
    "What is the largest planet in the solar system?",
]


def _make_example(context: str, question: str, answer: str) -> dict:
    user_content = f"Textbook passages:\n{context}\n\nQuestion: {question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


def build_topic_examples() -> list:
    examples = []
    for chapter in CHAPTERS:
        for topic in chapter["topics"]:
            context = topic["context"]
            for question, answer in topic["qas"]:
                examples.append(_make_example(context, question, answer))
    return examples


def build_glossary_examples() -> list:
    """Each glossary term gets all phrasings — question wording varies,
    the correct answer is identical, so no factual risk from rewording."""
    examples = []
    # Use the glossary term itself as a one-line "context" — a short,
    # accurate definitional passage, consistent with how a retriever would
    # surface a definition sentence from the textbook.
    for term, definition in GLOSSARY:
        context = f"{term}: {definition}"
        for phrasing in GLOSSARY_PHRASINGS:
            question = phrasing.format(term=term)
            examples.append(_make_example(context, question, definition))
    return examples


def build_out_of_context_examples(rng: random.Random, count: int = 40) -> list:
    """Pair real OS contexts with off-syllabus questions -> answer OUT_OF_CONTEXT."""
    all_contexts = [t["context"] for c in CHAPTERS for t in c["topics"]]
    seen_pairs = set()
    examples = []
    attempts = 0
    while len(examples) < count and attempts < count * 10:
        attempts += 1
        context = rng.choice(all_contexts)
        question = rng.choice(OFF_SYLLABUS_QUESTIONS)
        key = (context, question)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        examples.append(_make_example(context, question, "OUT_OF_CONTEXT"))
    return examples


def build_all(seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    examples.extend(build_topic_examples())
    examples.extend(build_glossary_examples())
    examples.extend(build_out_of_context_examples(rng, count=40))
    return examples


if __name__ == "__main__":
    exs = build_all()
    print(f"Conceptual examples built: {len(exs)}")
    topic_count = sum(len(t["qas"]) for c in CHAPTERS for t in c["topics"])
    print(f"  - from topic KB:        {topic_count}")
    print(f"  - from glossary (x3):   {len(GLOSSARY) * len(GLOSSARY_PHRASINGS)}")
    print(f"  - OUT_OF_CONTEXT:       40")
