# OS Tutor -- 4-System Comparative Evaluation Report

**Run timestamp**: 20261027_134103
**Questions evaluated**: 5 of 32 (EVAL_LIMIT=5, BATCH_SIZE=1)
**Evaluation library**: ragas  |  **Judge LLM**: qwen/qwen3.8-27b

## System Definitions

| System | Backend | RAG | Description |
| :--- | :---: | :---: | :--- |
| **S1 - Groq Only** | Groq cloud | No | Pure LLM baseline -- no textbook context |
| **S2 - Qwen Only** | Qwen2.5-1.5B+LoRA | No | Fine-tuned model only -- no retrieval |
| **S3 - RAG + Groq** | Groq cloud | Yes | Cloud LLM with Qdrant textbook retrieval |
| **S4 - RAG + Qwen** | Qwen2.5-1.5B+LoRA | Yes | **Primary system** -- fine-tuned + RAG |

## Comparative Results

| System | Faithfulness | Answer Relevancy | Context Precision |
| :--- | :---: | :---: | :---: |
| **S1 - Groq Only** | N/A | 0.9509 | N/A |
| **S2 - Qwen Only** | N/A | 0.8759 | N/A |
| **S3 - RAG + Groq** | 0.5910 | 0.9532 | 0.5000 |
| **S4 - RAG + Qwen** | 0.5229 | 0.8931 | 0.5000 |

## Metric Definitions

| Metric | Meaning |
| :--- | :--- |
| **Faithfulness** | Is the answer grounded in retrieved context? (0-1, higher = less hallucination) |
| **Answer Relevancy** | Does the answer address the question? (0-1, higher = better) |
| **Context Precision** | Were the most relevant chunks ranked highest? (0-1, higher = better) |

## Run Configuration

| Parameter | Value |
| :--- | :--- |
| EVAL_LIMIT | 5 |
| BATCH_SIZE | 1 |
| Total questions | 5 |

## Output Files
- Full CSV: `eval\results_20261027_134103.csv`
- Latest CSV: `eval\results_latest.csv`
