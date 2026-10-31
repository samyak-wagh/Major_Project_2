OS Tutor - IEEE conference paper (Overleaf project)
====================================================

HOW TO USE
1. Overleaf -> New Project -> Upload Project -> select this zip.
2. Menu -> Compiler: pdfLaTeX. Main document: main.tex. Compile.
   (IEEEtran, TikZ and pgfplots are built into Overleaf. No images or .bib needed.)

THE PAPER REPORTS WHAT WE MEASURED (including the disappointing results)
- Six systems on 17 clean gold questions: cloud, base Qwen, fine-tuned Qwen, each with/without retrieval.
- Fine-tuning did NOT improve text metrics (answer relevancy lower than base; faithfulness not detectably higher).
- RAG + cloud is the most faithful system.
- Judge-free calculation benchmark (27 problems): cloud 22/27; fine-tuned 1/27 strict (8/27 lenient); base 0/27 (4/27);
  fine-tuned vs base not significant (sign test p = 0.22).

BEFORE SUBMITTING - REPLACE / VERIFY
- Author names, affiliations, e-mails (placeholders in \author{}).
- Reproducibility paragraph: add the public repository / dataset link.
- References: written from memory. Check authors, venue, year and arXiv numbers, especially [gptoss] and [galvin]
  (add the textbook edition/year you actually used).
- This file was NOT compiled before packaging (no LaTeX on the author machine). Static checks passed
  (balanced braces/environments, table columns, citations). If Overleaf reports an error, check the log first.
- Venue template/page limit: adapt if the venue uses a different IEEE template.

WHERE EVERY NUMBER COMES FROM  (data/ folder in this zip; full write-up in eval2.md)
- Tables III, IV, Fig. 4    : data/scores_20261029_233832.csv  (102 scored rows, 6 systems x 17 questions)
- Answers + contexts        : data/answers_20261029_233832.csv
- Table V (calculation)     : data/num_bench_scored.csv, num_bench_answers.csv, num_bench_problems.json (first 3 problems per algorithm used)
- Table VI (latency)        : data/latency_20261029.csv (medians; single CPU-only run)
- Regenerated answers       : data/regenerated_answers_20261029.csv (19 rows)
- Dataset table / training  : dataset_v2_train.jsonl (702), dataset_v2_val.jsonl (60); loss curve from
                              qwen-os-tutor-lora/checkpoint-150/trainer_state.json (data/trainer_state_checkpoint150.json)
- Train/test overlap        : data/decontam_report.csv, data/leaked_gold.json (15 of 32 gold questions excluded)
- eval2.md                  : complete description of models, settings, changes and results

KNOWN WEAKNESSES (stated in the paper)
- 17 clean questions (6 numerical) and 27 calculation problems: wide intervals.
- One LLM judge (same family as the fine-tuned model), judge calls spread over several API keys; no human evaluation.
- S3 refusal scored by a stated rule (answer relevancy 0.0); the S3-vs-S4 relevancy comparison depends on it.
- Qwen output capped at 256 tokens; training prompt differs from inference prompt; adapter trained on a 4-bit base, run in fp32.
- Dataset: numerical half uses only nine templates; passages repeat; conceptual categories small.
- The repo's data README says 2,327 examples; the files on disk contain 762. The paper uses the file counts.
