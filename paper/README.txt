OS Tutor - IEEE conference paper (Overleaf project)
====================================================

HOW TO USE
1. Overleaf -> New Project -> Upload Project -> select this zip.
2. Menu -> Compiler: pdfLaTeX. Main document: main.tex. Compile.
   (IEEEtran, TikZ and pgfplots are built into Overleaf. No images or .bib needed.)

BEFORE SUBMITTING - REPLACE / VERIFY
- Author names, affiliations, e-mails (placeholders in \author{}).
- Reproducibility paragraph: add the public repo / dataset link.
- References: every entry was written from memory. Check authors, venue, year and
  arXiv numbers against the original sources, especially [gptoss] and [galvin]
  (add the edition / year of the textbook you actually used).
- Judge name "Qwen3.8-27B" is the Groq model id (qwen/qwen3.8-27b); confirm the
  name you want to print.
- Venue template: if the venue uses a different IEEE template, paste the body
  sections into it.
- This file was NOT compiled before packaging (no LaTeX on the author machine).
  Static checks passed (balanced braces/environments, table columns, citations).
  If Overleaf reports an error, it will be small; check the log first.

WHERE EVERY NUMBER COMES FROM  (data/ folder in this zip)
- Table II, III, Fig. 3 : data/results_20261027_134103.csv  (final run, 5 questions)
                          3 missing Answer-Relevancy cells re-scored with the same judge.
- Latency / answer length: measured during the same run (eval_run.log); single run.
- Table I / Section III-D: dataset_v2_train.jsonl (702), dataset_v2_val.jsonl (60).
- Loss curve (Fig. 2)    : qwen-os-tutor-lora/checkpoint-150/trainer_state.json
- Train/test overlap     : exact match + embedding retrieval + LLM same-question check (data/decontam_check.py) of the 32 gold questions
                           in eval/run_eval.py against the 762 training/validation questions.

KNOWN WEAKNESSES (stated in the paper, do not hide them)
- Only 5 of 32 gold questions evaluated (all on deadlocks); no confidence intervals.
- 15 of 32 gold questions have a near-identical training question (3 exact + 12 LLM-confirmed);
  3 of the 5 evaluated ones (deadlock definition, four conditions, safe state) are among them.
  Clean subset = 17 questions (data/decontam/leaked_gold.json).
- One LLM judge from the same family as the fine-tuned model; no human evaluation.
- Training prompt differs from inference prompt; adapter trained on 4-bit base,
  run on fp32 base.
- The repo's data README says 2,327 examples; the files on disk contain 762.
  The paper uses the file counts. Check which dataset version you trained on.
