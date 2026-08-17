# OS Tutor — Fine-Tuned Evaluation Report
**Model**: TinyLlama-1.1B + OS-tutor Adapter
**Questions**: 5  |  **Library**: ragas 0.4.3  |  **Evaluator**: gemini-2.5-flash

| System | Faithfulness | Answer Relevancy | Context Precision |
| :--- | :---: | :---: | :---: |
| **Fine-Tuned Baseline** | 0.0000 | 0.9172 | 0.0000 |
| **Fine-Tuned RAG**      | 0.1435  | 0.9166  | 0.3067  |

### Metric Definitions
- **Faithfulness** — Is the answer grounded in retrieved context? (0–1, ↑ = less hallucination)
- **Answer Relevancy** — Does the answer directly address the question? (0–1, ↑ = better)
- **Context Precision** — Were the most relevant chunks ranked highest? (0–1, ↑ = better)
