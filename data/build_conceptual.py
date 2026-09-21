"""
build_conceptual.py — Expand kb_conceptual.py (+ kb_advanced.py) into
fine-tuning examples across six question categories.
=========================================================================
Produces chat-format examples that mirror EXACTLY the prompt shape used at
inference time by rag/qa_chain.py (_SYSTEM_INSTRUCTION + "Textbook
passages:\n{context}\n\nQuestion: {question}"), so the fine-tuning
distribution matches the production distribution.

Each example now carries a top-level "category" and "difficulty" alongside
"messages" (NOT inside the message content — those stay exactly as the
inference-time prompt shape requires), for downstream filtering/balancing:

{
  "category": "recall" | "applied_conceptual" | "applied_numerical" | "comparative" | "analytical" | "negative_edge_case" | "out_of_context",
  "difficulty": "easy" | "medium" | "hard",
  "messages": [
    {"role": "system", "content": SYSTEM_INSTRUCTION},
    {"role": "user", "content": "Textbook passages:\n<context>\n\nQuestion: <question>"},
    {"role": "assistant", "content": "<direct, correct answer>"}
  ]
}

Category sources (all still hand-authored data -- a script cannot invent
correct reasoning, it can only orchestrate and assemble it):
  - recall              : kb_conceptual.py / kb_advanced.py topic "qas" lists,
                           and the GLOSSARY (single passage, single fact).
  - applied_conceptual  : topic "applied" lists (single passage, scenario
                           reframing of that topic's facts). Built here.
  - applied_numerical   : numerical_gen.py's 5 simulators, tagged in
                           build_dataset.py (NOT built in this file) -- kept
                           as a separate category from applied_conceptual so
                           the two sources never collapse into one
                           ambiguous 'applied' label downstream.
  - analytical           : topic "analytical" lists (single or same-topic
                           "why" reasoning).
  - comparative           : kb_conceptual.py's COMPARISONS list (two topics'
                           passages concatenated, answer grounded in both).
  - negative_edge_case   : topic "negative" lists (an in-syllabus question
                           where the naive answer is wrong/doesn't apply).
  - out_of_context       : real OS context + off-syllabus question (unrelated
                           to any topic) -> OUT_OF_CONTEXT, per qa_chain.py's
                           production refusal behaviour.
"""

import random

from kb_conceptual import (
    CHAPTERS as _CORE_CHAPTERS,
    GLOSSARY as _CORE_GLOSSARY,
    COMPARISONS as _CORE_COMPARISONS,
)
from kb_advanced import CHAPTERS as _ADVANCED_CHAPTERS, GLOSSARY as _ADVANCED_GLOSSARY

# kb_conceptual.py = direct textbook recall (definitions, "compare X and Y")
#                     plus applied/analytical/negative/comparative additions.
# kb_advanced.py   = applied, multi-concept, exam-synthesis scenarios with
#                     full reasoning-chain answers (hardware memory models,
#                     priority inversion traces, real-time scheduling math,
#                     COW/security mechanics). Merged so both feed the same
#                     pipeline below with no special-casing. kb_advanced.py
#                     doesn't define COMPARISONS yet -- getattr keeps this
#                     forward-compatible for when it does.
CHAPTERS = _CORE_CHAPTERS + _ADVANCED_CHAPTERS
GLOSSARY = _CORE_GLOSSARY + _ADVANCED_GLOSSARY
COMPARISONS = _CORE_COMPARISONS + getattr(
    __import__("kb_advanced"), "COMPARISONS", []
)

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


# Categories that require actual synthesis/reasoning (not bare single-fact
# retrieval) may never be tagged "easy" -- enforced here so a future authored
# entry can't silently default/slip into "easy" for these categories, not
# just documented as a convention in the builder functions below.
_NEVER_EASY_CATEGORIES = {"applied_numerical", "applied_conceptual", "comparative", "analytical"}


def _make_example(context: str, question: str, answer: str, category: str, difficulty: str) -> dict:
    if category in _NEVER_EASY_CATEGORIES and difficulty == "easy":
        raise ValueError(
            f"category={category!r} must be 'medium' or 'hard', got difficulty='easy' "
            f"(question: {question[:60]!r}...)"
        )
    user_content = f"Textbook passages:\n{context}\n\nQuestion: {question}"
    return {
        "category": category,
        "difficulty": difficulty,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ],
    }


def _recall_difficulty(answer: str) -> str:
    """Cheap, explicit heuristic for un-tagged recall facts: short single-clause
    answers are 'easy'; longer, multi-part answers are 'medium'. Any topic that
    deserves 'hard' should have that recall QA rewritten as an 'analytical'
    entry instead -- recall by definition tops out at 'medium'."""
    return "easy" if len(answer.split()) < 25 else "medium"


def _topic_index() -> dict:
    """id -> topic dict, across kb_conceptual.py + kb_advanced.py, for
    COMPARISONS lookups."""
    index = {}
    for chapter in CHAPTERS:
        for topic in chapter["topics"]:
            index[topic["id"]] = topic
    return index


def build_topic_examples() -> list:
    """category='recall': one topic's own context -> one of its own qas."""
    examples = []
    for chapter in CHAPTERS:
        for topic in chapter["topics"]:
            context = topic["context"]
            for question, answer in topic["qas"]:
                examples.append(
                    _make_example(context, question, answer, "recall", _recall_difficulty(answer))
                )
    return examples


def build_glossary_examples() -> list:
    """category='recall', difficulty='easy': single-line definitions get all
    phrasings — question wording varies, the correct answer is identical, so
    no factual risk from rewording."""
    examples = []
    for term, definition in GLOSSARY:
        context = f"{term}: {definition}"
        for phrasing in GLOSSARY_PHRASINGS:
            question = phrasing.format(term=term)
            examples.append(_make_example(context, question, definition, "recall", "easy"))
    return examples


def build_applied_examples() -> list:
    """category='applied_conceptual': a topic's own context -> a scenario
    question that requires recognizing which concept from that passage
    applies and why. Named 'applied_conceptual' (not just 'applied') to keep
    it distinct from numerical_gen.py's 'applied_numerical' examples, which
    are also scenario+compute but come from a completely different pipeline
    -- collapsing both into one 'applied' label made it impossible to tell
    them apart downstream without re-deriving the split via content
    heuristics. Default difficulty floor is 'medium' -- applying a concept
    to a novel scenario is never as trivial as bare recall, so it must never
    default to 'easy' (enforced by _make_example's _NEVER_EASY_CATEGORIES
    guard above). Author explicit difficulty='hard' per item for scenarios
    requiring multi-step reasoning or combining more than one mechanism."""
    examples = []
    for chapter in CHAPTERS:
        for topic in chapter["topics"]:
            for item in topic.get("applied", []):
                examples.append(
                    _make_example(
                        topic["context"], item["question"], item["answer"],
                        "applied_conceptual", item.get("difficulty", "medium"),
                    )
                )
    return examples


def build_analytical_examples() -> list:
    """category='analytical': a topic's own context -> a 'why' reasoning
    question about the mechanism behind that topic's facts. Default
    difficulty floor is 'hard' -- explaining WHY a mechanism behaves a
    certain way is the deepest single-topic reasoning tier; author explicit
    difficulty='medium' per item only if the underlying 'why' is genuinely
    shallow (enforced 'never easy' by _make_example's guard above)."""
    examples = []
    for chapter in CHAPTERS:
        for topic in chapter["topics"]:
            for item in topic.get("analytical", []):
                examples.append(
                    _make_example(
                        topic["context"], item["question"], item["answer"],
                        "analytical", item.get("difficulty", "hard"),
                    )
                )
    return examples


def build_negative_examples() -> list:
    """category='negative_edge_case': an in-syllabus question where the
    concept from this topic's context does NOT straightforwardly apply --
    distinct from out_of_context, which is an off-syllabus question paired
    with an unrelated passage."""
    examples = []
    for chapter in CHAPTERS:
        for topic in chapter["topics"]:
            for item in topic.get("negative", []):
                examples.append(
                    _make_example(
                        topic["context"], item["question"], item["answer"],
                        "negative_edge_case", item.get("difficulty", "medium"),
                    )
                )
    return examples


def build_comparative_examples() -> list:
    """category='comparative': two topics' contexts concatenated -> an
    answer that synthesizes both, grounded only in those two passages.
    Default difficulty floor is 'hard' -- correctly relating two distinct
    concepts (not just stating each in isolation) is exam-synthesis-tier
    reasoning by construction. Author explicit difficulty='medium' per item
    only when the two underlying passages are both genuinely simple/short
    (enforced 'never easy' by _make_example's guard above)."""
    index = _topic_index()
    examples = []
    for item in COMPARISONS:
        topic_a = index.get(item["topic_a"])
        topic_b = index.get(item["topic_b"])
        if topic_a is None or topic_b is None:
            missing = item["topic_a"] if topic_a is None else item["topic_b"]
            raise KeyError(f"COMPARISONS entry references unknown topic id: {missing!r}")
        combined_context = f"{topic_a['context']}\n\n{topic_b['context']}"
        examples.append(
            _make_example(
                combined_context, item["question"], item["answer"],
                "comparative", item.get("difficulty", "hard"),
            )
        )
    return examples


def build_out_of_context_examples(rng: random.Random, count: int = 40) -> list:
    """category='out_of_context': real OS contexts + off-syllabus questions."""
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
        examples.append(_make_example(context, question, "OUT_OF_CONTEXT", "out_of_context", "easy"))
    return examples


def build_all(seed: int = 42) -> list:
    rng = random.Random(seed)
    examples = []
    examples.extend(build_topic_examples())
    examples.extend(build_glossary_examples())
    examples.extend(build_applied_examples())
    examples.extend(build_analytical_examples())
    examples.extend(build_negative_examples())
    examples.extend(build_comparative_examples())
    examples.extend(build_out_of_context_examples(rng, count=40))
    return examples


if __name__ == "__main__":
    exs = build_all()
    print(f"Conceptual examples built: {len(exs)}")

    by_category = {}
    by_difficulty = {}
    for ex in exs:
        by_category[ex["category"]] = by_category.get(ex["category"], 0) + 1
        by_difficulty[ex["difficulty"]] = by_difficulty.get(ex["difficulty"], 0) + 1

    print("\nBy category:")
    for cat, n in sorted(by_category.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:20s}: {n}")

    print("\nBy difficulty:")
    for diff in ("easy", "medium", "hard"):
        print(f"  {diff:20s}: {by_difficulty.get(diff, 0)}")
