# pail_classify — open AI-text detectors on web text (PAIL repo, src/classify/)

Updated 2026-10-14.

## Done
- MELD and ProseLens (transformers backend) scored on pangram_web_v1 (225 docs, Pangram API labels).
  ProseLens AUROC 0.996 / kappa 0.946 @1% FPR; MELD 0.964 / 0.722. Write-up: PAIL docs/classify/pangram_web_v1.md;
  scores: s3://ai2-llm/checkpoints/prasanns/PAIL/classify/pangram_web_v1/. All code is in PAIL origin/main.

## Running (cubbins, jsteinhardt/preemptive_high; code = git archive of origin/main at /data/prasann/PAIL/code)
- 3832244 pail_env_dl: builds pail (with vLLM) on cubbins /data, pre-downloads ProseLens.
- 3832245 proselens_vllm_parity_cubbins (afterok): -> /data/prasann/PAIL/classify/pangram_web_v1/proselens_vllm.jsonl
- 3832246 proselens_vllm_cc10M_cubbins (afterok): 2,946 raw CC docs; log /data/prasann/joblogs/classify_proselens_cc_3832246.log prints tok/s.
Earlier mooney/lorax attempts cancelled (mooney full; lorax GPUs broken, see Gotchas).

## Next
1. Parity: compare proselens_vllm.jsonl p_human with the transformers scores (s3 proselens.jsonl); card says ~0.001.
   If it matches, record tok/s and redo the 1 TB / 100-H200 estimate (was 50-70k tok/s/H200, unmeasured).
2. EditLens: HF access to pangram/editlens_Llama-3.2-3B pending Pangram review (403 as of 2026-10-14).
3. Optional: ProseLens-ModernBERT-L on the same set.

## Gotchas
- lorax (2026-10-14): every idle H200 (0,1,3,4,5) fails `torch.cuda.init()`; only GPU 2, busy with another user's job, worked. Node-level fault; report to admins, don't schedule there. pail env (with vLLM) exists on lorax, mooney, and (after 3832244) cubbins /data.
- The shared PAIL checkout may be on someone else's branch; use a worktree for changes.
