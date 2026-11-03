---
title: Literature review — efficient (long-context) training by token dropping, routing, selection and backward compression, vs the tokdrop_lr recipe
date: 2026-11-03
updated: 2026-11-03
experiment: tokdrop_lr
area: token_routing
kind: audit
status: current
superseded_by: 
artifact: https://claude.ai/artifact/SPSBxHFs2PUn5CPjJX3Wgt (tokdrop_lr hub; no lit-review section yet)
summary: ~70 papers through Nov 2026. Mid-stack token dropping in training is prior art (Hou 2022 token dropping, Random-LTD, LeMo/Jenga for long-context FT), but no published long-context fine-tuning method reports more than ~1.4-2x training savings, and none refits a router online on per-token gradient norms. Closest = LeMo/Jenga, Random-LTD, Hou 2022, TokenGate, TokenSeek. Baselines to run = Random-LTD per-layer resampling, attention-score selection at block 7 (LeMo/A-MoD style), an end-to-end MoD/TokenGate-style learned gate, LongLoRA S2-Attn, and InfoBatch/GRESO-style solved-row forward skipping (the most promising stack toward 5x+).
---
# Literature review: efficient long-context training vs our token-drop recipe

**What we do (for reference).** Qwen3.5-4B (hybrid full attention + Gated DeltaNet, 36 blocks), CTC
long-context SFT (contradiction / nq / outlier / oolong; 8k rung primary), whole-model LoRA r64.
All tokens run blocks 0-6. After block 7 we keep 20% of each document's body tokens, and after
block 16 we keep 7%. Dropped tokens exit the stack permanently (nested sets). Prompt, query and
answer tokens are always kept. Selection comes from an online ridge router, refit every step on
per-token gradient-norm targets from the training backward. Inference is dense over the full
context (train-sparse / test-dense). Dense FLOPs-to-match is ~4-4.5x (8k-only FTM 4.6x / 3.6x /
>=4.5x at 1,952 / 3,904 / 7,808 rows per task; see `ops/experiments/tokdrop_lr/STATUS.md`).
Random selection nearly ties the router at larger data.

**Method.** Five parallel web searches (sonnet/opus subagents), one per category below. Each
checked papers against arXiv, venue pages or the full HTML. I re-checked the four newest and most
load-bearing entries myself (TokenGate, TokenDrop, TokenSeek, Centrifuge). Claims marked *(abs)*
come from an abstract or a secondary summary only. Read the paper before quoting those numbers in
a draft.

## Verdict

**Is the method novel?** Partly. The pieces split three ways:

- **Not novel: dropping context tokens in middle layers during training while keeping full-token
  inference.** This is the core of [Hou et al. 2022, "Token Dropping for Efficient BERT
  Pretraining"](https://arxiv.org/abs/2203.13240) and [Random-LTD](https://arxiv.org/abs/2211.11586).
  It was taken to long-context LoRA fine-tuning by [LeMo](https://arxiv.org/abs/2501.09767), whose
  USENIX ATC 2025 version is [Jenga](https://www.usenix.org/conference/atc25/presentation/wang-tuowei).
  Reviewers will cite these three.
- **Novel as far as we found: the router.** We found no paper that fits a token router online to
  per-token gradient-norm targets taken from the training backward.
  - The nearest are Hou 2022 (cumulative MLM loss as importance) and
    [TokenSeek](https://arxiv.org/abs/2601.19739) (ICLR 2026: attention plus gradient score, used
    for backward-only memory savings).
  - Row-level gradient-norm importance sampling is also related
    ([Katharopoulos & Fleuret 2018](https://arxiv.org/abs/1803.00942)).
  - But since random nearly ties the router, the router is a weak headline contribution.
- **Novel and the strongest claim: the size of the saving in this setting.** No published
  long-context fine-tuning method reports more than about 1.4-2x training compute at parity:
  - LeMo/Jenga: up to 1.36x speed and 1.93x memory.
  - Centrifuge: up to 1.5x end-to-end.
  - TokenDrop: up to 1.5x.
  - SparseLoRA: up to 2.2x compute.

  Our ~4-4.5x training-FLOP saving, on a hybrid linear-attention model with dense full-context
  inference, is above everything we found. The diagnostics are also unpublished as far as we can
  tell: random vs learned, the gradient-concentration and unbiasedness measurements, and
  per-row vs per-document allocation.

**The 3-5 closest prior works**

1. **LeMo / Jenga.** Long-context (4k-64k) LoRA fine-tuning of Llama2/3. Per-layer token
   elimination, with small offline-trained predictors of block-max attention. Saves 1.36x speed
   and 1.93x memory. No random baseline.
2. **Random-LTD.** Random per-layer token subsets in the middle layers, with the first and last
   layers kept full. 25.6% wall-clock saving at GPT-3 1.3B. Random beats Hou's loss-based bypass
   on GPT-2 350M fine-tuning (PTB perplexity 18.70 vs 20.05).
3. **Hou et al. 2022 token dropping (TokenBypass).** Drops 50% of tokens after layer 5 using
   loss-derived importance, and re-attaches them at the last layer. Saves 25% of BERT pretraining.
   Loss-based selection beats random by only about 0.5 GLUE points.
4. **TokenGate** ([arXiv 2610.26754](https://arxiv.org/abs/2610.26754), Oct 2026).
   - Architecturally the closest: a learned gate at layer 1-3 keeps the top-K document tokens for
     all later layers, LoRA is trained end to end through a "token-gated attention", and it runs on
     **Qwen3.5-2B** (hybrid).
   - It is an *inference* (prefill) method with no training saving: TTFT 1.4-11.3x faster.
   - Learned "significantly" beats Random-K at inference budgets *(abs; we verified only through
     secondary coverage and its GitHub repo)*.
5. **TokenSeek** (ICLR 2026). Instance-aware selection from attention plus gradient scores. Keeps
   about 10% of tokens for the backward, saving memory but not FLOPs. Random selection
   (TokenTune) is unstable there. It is the nearest "gradient-signal token selection".

**The 3-5 baselines most worth running** (details in "Baselines" below)

1. **Random-LTD-style per-layer independent resampling.** Fresh random keep sets at every dropped
   layer, the last layer(s) full, and a growing keep schedule, at matched FLOPs. Our random arm
   uses nested permanent exit, which is the design Random-LTD beat.
2. **Attention-score selection at block 7** (LeMo, A-MoD, GemFilter style). Block 7 is a
   full-attention block in Qwen3.5, so the score is free. It needs no parameters and directly
   tests gradient targets against attention targets. It is also what the closest published work
   uses.
3. **End-to-end learned gate** (the MoD top-k router, or TokenGate's token-gated attention). This is
   the canonical learned-router comparison, and the one reviewers will ask for.
4. **LongLoRA S2-Attn.** The standard train-sparse / test-dense long-context baseline. Expect it to
   save little at 8k on a mostly-GDN model, but it should be cited and probably run once.
5. **Solved-row forward skipping with unbiased rescaling** (InfoBatch + GRESO / stale-loss
   Selective Backprop). This stacks on the recipe rather than competing with it. Given our
   cross-row gradient concentration, it is the most promising route past 5x.

**Is 5-10x plausible?** The literature gives no example of 5-10x training-compute savings at
parity from token dropping in LM training.

- The 10x+ numbers elsewhere are fewer *steps*, with reference models, outside LMs:
  [RHO-LOSS](https://arxiv.org/abs/2206.07137) 18x steps, [JEST](https://arxiv.org/abs/2406.17711)
  13x iterations.
- Or they come from pretraining from scratch: [MoD](https://arxiv.org/abs/2404.02258) uses over
  50% fewer FLOPs per forward.
- [Kaddour et al. 2023, "No Train No Gain"](https://arxiv.org/abs/2307.06440) found that layer
  dropping, Selective Backprop and RHO-LOSS do not beat a properly LR-decayed baseline at fixed
  compute.

The remaining levers the literature points to multiply with ours rather than replace it:

- skipping already-solved rows, which is where our gradient concentration lives;
- shortening sequences at the data level ([LongRecipe](https://arxiv.org/abs/2409.00509),
  [PoSE](https://arxiv.org/abs/2309.10400));
- an early learned gate to break the blocks 0-6 dense floor (TokenGate gates at layer 1-3).

## 1. Token dropping / routing during training

| Paper | Method (one line) | vs ours | Reported saving, scale | Long ctx | Learned vs random |
|---|---|---|---|---|---|
| [Hou et al. 2022, ACL](https://arxiv.org/abs/2203.13240) | Drop 50% of tokens after layer 5 by cumulative MLM loss; re-attach at the last layer | **Same idea** (loss-derived importance, mid-stack bypass) | 25% of BERT pretraining | no | Loss-based > random by ~0.5 GLUE; >62.5% drop hurts |
| [Random-LTD (Yao et al. 2022)](https://arxiv.org/abs/2211.11586); [DeepSpeed Data Efficiency, AAAI 2024](https://arxiv.org/abs/2212.03597) | Independent random subset per middle layer; first/last layers full; growing keep; LayerToken LR | **Same idea, random router**; key baseline | 33% theoretical / 25.6% wall-clock, GPT-3 1.3B; up to 2x data/time with curriculum *(abs)* | no | Random **beats** TokenBypass (PTB PPL 18.70 vs 20.05; 20.57 vs 30.36 when dropping from layer 2) |
| [ScTD (Zhong et al. 2023, ACL)](https://arxiv.org/abs/2305.15273) | Token dropping + self-distillation consistency loss | Partial (a regularizer we could add) | up to 57% BERT pretraining time *(abs)* | no | — |
| [Mixture-of-Depths (Raposo et al. 2024)](https://arxiv.org/abs/2404.02258) | Learned expert-choice top-k router per block; the rest skip via the residual | Partial (learned per-block top-k, from scratch, also sparse at inference) | isoFLOP parity at >50% fewer forward FLOPs; 60M-3B, 2k ctx; 12.5% capacity on every other block | no | Random routing "drastically worse" |
| [CoLT5 (Ainslie et al. 2023, EMNLP)](https://arxiv.org/abs/2303.09752) | Light branch for all tokens; learned router sends a subset to heavy attention/FFN | Partial (learned router, long input) | 35-75% training speedup over LongT5 at 16k; to 64k | **yes** | Static (evenly spaced) routing: "massive drop" |
| [MoDification (2025, NAACL)](https://arxiv.org/abs/2410.14268) | Convert a pretrained LLM to MoD with a threshold router + continued training | Partial, inference-oriented | ~1.2x latency, 1.8x memory, 3B-70B, incl. 32k *(abs)* | yes | — |
| [Router-Tuning / MindSkip (2025, EMNLP)](https://arxiv.org/abs/2410.13184) | Tiny router to skip attention on a frozen LLM | Inference-oriented | ~21% speedup *(abs)* | no | — |
| [D-LLM (NeurIPS 2024)](https://proceedings.nips.cc/paper_files/paper/2024/file/03469b1a66e351b18272be23baf3b809-Paper-Conference.pdf) | Per-token per-layer skip gates learned in fine-tuning | Partial | 45-50% compute + KV *(abs)* | no | — |
| [SkipGPT (ICML 2025)](https://arxiv.org/abs/2506.04179) | Gumbel token routers (attn/MLP separate), then LoRA | Partial, inference | >40% params pruned *(abs)* | no | — |
| [A-MoD (2025)](https://arxiv.org/abs/2412.20875) | MoD router replaced by the previous layer's attention-map score (no params) | Baseline idea (attention-score router) | up to 2x faster transfer, ViT *(abs)* | no | Beats a learned router |
| [γ-MoD (2024)](https://arxiv.org/abs/2410.13859) | MoD for multimodal LLMs; pick layers by attention rank | Partial | *(abs)* | no | Naive dense→MoD conversion degrades |
| [Mixture-of-Recursions (NeurIPS 2025)](https://arxiv.org/abs/2507.10524) | Per-token recursion depth by router | Tangential | — | no | Routers can collapse to positions ("dead tokens" up to 66.7%) |
| [Lawson & Aitchison 2025](https://arxiv.org/abs/2506.21103) | Gated skipping of middle layers | Partial, **negative** | No CE-vs-FLOPs gain over smaller dense | no | — |
| [LeMo](https://arxiv.org/abs/2501.09767) / [Jenga (USENIX ATC 2025)](https://www.usenix.org/conference/atc25/presentation/wang-tuowei) | Per-layer elimination of low block-max-attention tokens; offline low-rank predictors; custom kernel | **Closest in setting** | up to 1.36x speed, 1.93x memory; Llama2/3, 4k-64k; PG19 PPL 6.95→7.11 at 8k | **yes** | No random baseline |
| [TokenGate (2026)](https://arxiv.org/abs/2610.26754) | Learned gate at layer 1-3 keeps top-K doc tokens; LoRA through token-gated attention; Qwen3 / **Qwen3.5-2B** | **Closest in architecture**, but inference | TTFT 1.4-11.3x, prefill memory 1.5-3.4x; no training saving | yes | Learned ≫ Random-K *(abs)* |
| [FLIP (Li et al. 2023, CVPR)](https://arxiv.org/abs/2212.00794) | Random 50-75% patch masking in CLIP training | Analogy (vision) | faster *and* better *(abs)* | no | Random works |
| [Funnel](https://arxiv.org/abs/2006.03236) · [Hourglass](https://arxiv.org/abs/2110.13711) · [LongT5](https://arxiv.org/abs/2112.07916) · [Learned Token Pruning](https://arxiv.org/abs/2107.00910) · [ToMe](https://arxiv.org/abs/2210.09461) | Architectural length reduction / pruning / merging | Background | LTP 2.1x BERT FLOPs | partly | — |
| Inference-only: [LazyLLM](https://arxiv.org/abs/2407.14057), [GemFilter](https://arxiv.org/abs/2409.17422), [Dynamic Context Pruning](https://arxiv.org/abs/2305.15805), [SkipDecode](https://arxiv.org/abs/2307.02628), [LayerSkip](https://arxiv.org/abs/2311.08623), [Informed Routing](https://arxiv.org/abs/2510.13831) | Prefill/decoding pruning or early exit | Not training | GemFilter: layer-13 selection of 1,024 tokens, 2.4x | yes | Informed Routing: greedy "immediate impact" targets are poor importance proxies (relevant to gradient-norm targets) |

**Learned vs random across the literature.** Learned selection wins clearly when the model is
also sparse at test time or is trained from scratch (MoD, CoLT5, TokenGate). In the train-sparse /
test-dense regime random ties or wins (Random-LTD vs TokenBypass, FLIP, Hou's ~0.5 GLUE margin).
Our "random nearly ties learned" is consistent with that regime. Nobody has shown it for
long-context fine-tuning.

## 2. Gradient- or loss-based example and token selection

| Paper | Method | vs ours | Reported saving, scale | Online? | vs random |
|---|---|---|---|---|---|
| [Loshchilov & Hutter 2016](https://arxiv.org/abs/1511.06343) | Rank by latest loss, exponential selection | Baseline | ~5x on MNIST | online (stale loss) | — |
| [Katharopoulos & Fleuret 2018, ICML](https://arxiv.org/abs/1803.00942) | Importance sampling by an upper bound on per-sample gradient norm; switches on when variance reduction is large | Partial (gradient-norm target, row level) | wall-clock gains, small vision | online | vs uniform |
| [Alain et al. 2015](https://arxiv.org/abs/1511.06481) | Theory: the optimal proposal is proportional to per-example gradient norm | Theory behind our target | — | — | — |
| [RAIS (Johnson & Guestrin 2018, NeurIPS)](https://papers.nips.cc/paper/2018/hash/967990de5b3eac7b87d49a13c6834978-Abstract.html) | Robust approximate gradient-norm importance sampling | Partial | ≥20% speedups | online | — |
| [Selective Backprop (Jiang et al. 2019)](https://arxiv.org/abs/1910.00762) | Full forward; backprop only high-loss examples | **We ran it** (top-50%: ties recipe at 26% fewer FLOPs) | up to 3.5x to target error, CIFAR/SVHN; stale-forward variant +26% | online | 1.02-1.8x over importance sampling |
| [InfoBatch (Qin et al. 2024, ICLR oral)](https://arxiv.org/abs/2303.04947) | Randomly prune below-mean-loss samples, rescale survivors 1/(1-r), anneal off at the end | **Stacks** (unbiased cousin of skipping solved rows) | ~40% on CIFAR / ImageNet; ~20% LLaMA instruction tuning | online | partly random by design |
| [GREATS (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ed165f2ff227cf36c7e3ef88957dadd9-Abstract.html) | Greedy per-step batch selection from gradient inner products with validation data ("ghost" trick) | Partial | faster LLM fine-tuning convergence *(abs)* | needs validation gradients | — |
| [RHO-LOSS (Mindermann et al. 2022, ICML)](https://arxiv.org/abs/2206.07137) | Select by training loss minus the loss of a holdout-trained model | Concept ("learnable, not yet learnt" = our solved rows) | 18x fewer *steps* on Clothing-1M | needs ref model | — |
| [JEST (Evans et al. 2024, NeurIPS)](https://arxiv.org/abs/2406.17711) | Joint batch selection by learnability vs a reference model | Partial | 13x iterations, 10x compute (image-text) | needs ref | — |
| [Forgetting events (Toneva et al. 2019)](https://arxiv.org/abs/1812.05159) · [Data Diet EL2N/GraNd (Paul et al. 2021)](https://arxiv.org/abs/2107.07075) · [CRAIG](https://arxiv.org/abs/1906.01827) · [GRAD-MATCH](https://arxiv.org/abs/2103.00123) · [LESS (Xia et al. 2024)](https://arxiv.org/abs/2402.04333) | Offline example pruning / coresets / gradient-influence selection | Concept | EL2N: half of CIFAR-10; CRAIG up to 3x on deep nets; LESS: 5% of data often beats all | offline | GraNd at init ≈ random ([Kirsch 2023](https://arxiv.org/abs/2303.14753)) |
| [Rho-1 (Lin et al. 2024, NeurIPS)](https://arxiv.org/abs/2404.07965) | Loss only on tokens with high excess loss vs a reference model | Baseline (loss tokens, not context tokens); ~0 FLOP saving | MATH 40.6% (1B) / 51.8% (7B) | needs ref | — |
| [ESLM (Bal et al., TMLR)](https://arxiv.org/abs/2505.19893) | Online per-token loss/entropy selection with a value-at-risk threshold | Partial, closest in spirit (online, self-referential) | "significantly reduces FLOPs", GPT-2 124M-774M *(abs)* | online | unverified |
| [Token Cleaning (ICML 2025)](https://arxiv.org/abs/2502.01968) · [TIS-DPO (ICLR 2025)](https://arxiv.org/abs/2410.04350) · [MiLe loss](https://arxiv.org/abs/2310.19531) | Token-level loss weighting / cleaning | Orthogonal (quality, not FLOPs) | — | mixed | — |
| [Centrifuge, formerly Collider (Chai et al., ICLR 2026)](https://arxiv.org/abs/2502.00340) | Rho-1-style filtering made to save compute by dropping filtered tokens' activations in the backward, including attention | Systems complement | loss-only filtering saves ~1.2%; with backward drop up to 49.9% of the backward and 34.7% end-to-end at 50% filtered, 1.1B-40B | needs ref (filter) | — |
| [LongPPL / LongCE (Fang et al. 2025, ICLR)](https://arxiv.org/abs/2410.23771) · [Helm et al. 2025](https://arxiv.org/abs/2503.09202) | Up-weight "key tokens" by the long-vs-short-context log-prob gap | Alternative router target / answer weighting | LongCE +36-79% wall-clock; up to +22 LongEval | online (extra short forward) | no random baseline |
| [ProLong data (Chen et al. 2024, ACL)](https://arxiv.org/abs/2405.17915) · [LongAttn](https://arxiv.org/abs/2502.16860) · [GATEAU](https://arxiv.org/abs/2410.15633) | Offline long-context data selection by dependency score | Data level, orthogonal | quality, not FLOPs | offline | LongAttn > random |
| [DAPO dynamic sampling](https://arxiv.org/abs/2503.14476) · [GRESO (2025)](https://arxiv.org/abs/2506.02177) | RL: drop zero-advantage (solved / unsolvable) prompts; GRESO predicts them *before* rollout, with probabilistic re-admission | **Stacks**: the "skip the forward too" version of our solved-row finding | GRESO up to 3.35x fewer rollouts, 2.0x end-to-end | online | ~20% of skipped prompts become informative again |
| [No Train No Gain (Kaddour et al. 2023, NeurIPS)](https://arxiv.org/abs/2307.06440) | Re-tests layer dropping, Selective Backprop and RHO-LOSS at fixed compute on BERT/T5 | **Cautionary** | none beats a fully LR-decayed baseline (24h val loss 2.10 vs SB 2.18, RHO 2.16) | — | token dropping not tested |
| [Xia et al. 2025, "Random Selection is Almost All You Need"](https://arxiv.org/abs/2410.09335) | SFT data selection at ~1M scale | Evidence that random ties at scale | self-scoring methods ≈ random | — | random ≈ learned |

## 3. Long-context-specific efficient fine-tuning

| Paper | Method | vs ours | Saving, scale | Train-sparse / test-dense? |
|---|---|---|---|---|
| [LongLoRA / S2-Attn (Chen et al. 2024, ICLR)](https://arxiv.org/abs/2309.12307) · [LongQLoRA](https://arxiv.org/abs/2311.04879) · [SinkLoRA](https://arxiv.org/abs/2406.05678) | Shifted local attention in groups during fine-tuning; LoRA + trainable embeddings and norms | **Baseline** (canonical train-sparse/test-dense) | Llama2-7B 4k→100k on 8xA100; saves attention FLOPs only | **yes** |
| [LongRecipe (Hu et al. 2025, ACL)](https://arxiv.org/abs/2409.00509) | Pick impactful tokens (logit change, base vs extended model), keep those sentences, train at 30% of the target window + PoSE positions | Partial, stackable (input-level, not per layer) | "over 85%" compute, 7-8B to 80k-128k | yes | 
| [PoSE (Zhu et al. 2024, ICLR)](https://arxiv.org/abs/2309.10400) · [CREAM (NeurIPS 2024)](https://arxiv.org/abs/2406.07138) · [Randomized PE](https://arxiv.org/abs/2305.16843) | Train in a short window with skipped or sampled position indices | Orthogonal; cheap short-to-long baseline | 2k window → 128k+ | yes |
| [Position Interpolation](https://arxiv.org/abs/2306.15595) · [YaRN](https://arxiv.org/abs/2309.00071) · [LongRoPE2](https://arxiv.org/abs/2502.20082) | RoPE rescaling + short fine-tune | Orthogonal (we already use YaRN for r256k) | ~1k steps (PI); LongRoPE2 128k with 10B tokens | yes |
| [SSA (Shen et al., 2025, arXiv 2511.20102)](https://arxiv.org/abs/2511.20102) | Randomly choose sparse or full attention per step and align outputs | Partial, conceptual (closing the train/test gap by mixing dense steps) | *(abs)* | yes |
| [NSA](https://arxiv.org/abs/2502.11089) · [MoBA](https://arxiv.org/abs/2502.13189) · [DuoAttention](https://arxiv.org/abs/2410.10819) · [SeerAttention](https://arxiv.org/abs/2410.13276) · [Locret](https://arxiv.org/abs/2410.01805) | Natively trainable or learned sparse attention / KV eviction | Orthogonal (attention sparsity, sparse at inference) | NSA up to 11.6x at 64k (kernel) | mostly no |
| [Activation Beacon](https://arxiv.org/abs/2401.03462) · [CEPE](https://arxiv.org/abs/2402.16617) · [FocusLLM](https://arxiv.org/abs/2408.11745) · [LLoCO](https://arxiv.org/abs/2404.07979) · [E2LLM](https://arxiv.org/abs/2409.06679) · [Landmark Attention](https://arxiv.org/abs/2305.16300) · [Focused Transformer](https://arxiv.org/abs/2307.03170) | Context compression / memory tokens / retrieval-gated blocks | Different goal (compressed inference); overlaps our old landmark / pooled-KV work | CEPE ~10x throughput, LLoCO 30x fewer inference tokens | no |
| [Fu et al. 2024, data for 128k](https://arxiv.org/abs/2402.10171) · [ProLong (Gao et al. 2025)](https://arxiv.org/abs/2410.02660) · [IN2 / FILM-7B](https://arxiv.org/abs/2404.16811) | Long-context data recipes | Orthogonal | — | — |
| [Zhao et al. 2024, sequence composition](https://arxiv.org/abs/2402.13991) · [SPLiCe](https://arxiv.org/abs/2312.17296) | Intra-document masking; packing related documents | Orthogonal (packing) | intra-doc mask ~4% runtime | — |
| [LongMamba](https://arxiv.org/abs/2504.16053) | Training-free decay adjustment for SSM long context | Tangential; nothing found on Gated DeltaNet-hybrid long-context *training* | — | — |

## 4. Backward-pass compression

Backward compression has a hard ceiling. With a frozen base plus LoRA, the backward computes
activation gradients but almost no weight gradients, so it costs about 1x the forward. A
backward-only method therefore caps near 2x, not 3x. Our own attempt was already negative
(`records/sparse-backward.md` on branch `claude/sparse-backward`: token-sparse VJPs collapsed
contradiction and outlier). The literature does not contradict that.

| Paper | Method | vs ours | Saving, scale | vs random |
|---|---|---|---|---|
| [TokenTune (Simoulin et al. 2024, EMNLP)](https://arxiv.org/abs/2501.18824) | Dense forward; backprop through k tokens, the rest `no_grad` | Partial; random backward-token baseline | memory only (−67% peak), Llama2-7B | random only |
| [TokenDrop (Kim, Namkung, Jeon, ICML 2026)](https://icml.cc/virtual/2026/poster/65486) | Skip backward for tokens with small residual-update L2 norm; nested retained sets; lazy selection | Partial (backward-only; nested sets like ours) | up to 1.5x speed, −42.9% memory | beats backprop-skipping baselines |
| [TokenSeek (Zeng et al., ICLR 2026)](https://arxiv.org/abs/2601.19739) | Attention + gradient token scores; ~10% of tokens get gradients | Partial (gradient-signal selection) | memory (14.8% of full on Llama3.2-1B with QLoRA); +11-15% GPU time | random (TokenTune) unstable |
| [Centrifuge](https://arxiv.org/abs/2502.00340) | See section 2 | Stackable template for "dense forward, token-sparse backward" | −49.9% backward, −34.7% end-to-end | — |
| [SpaCO (Li et al. 2025, Findings ACL)](https://arxiv.org/abs/2505.16710) | Chunk-wise backward through a random subset of chunks, with unbiased compensation | Partial, long-context, random | up to 3x vs SeCO; 8B LoRA at 16k on one 3090 | random |
| [SparseLoRA (Khaki et al., ICML 2025)](https://arxiv.org/abs/2506.16500) | Training-free SVD estimator picks sparse channels; context tokens sparse, output tokens dense; early steps dense; deeper layers sparser | Stackable; its design choices match ours | up to 2.2x compute, 1.6x measured, Llama3-8B | unverified |
| [DropBP (Woo et al., NeurIPS 2024)](https://arxiv.org/abs/2402.17812) | Randomly drop whole layers in the backward only; sensitivity-set rates | Stackable (layer-level) | −44% training time, LLaMA3-8B | random by design |
| [SUS backprop (2025)](https://arxiv.org/abs/2505.15080) | Unbiased stochastic attention backward, O(nc) | Stackable (attention backward) | ~99% of attention backward weights cut at ~2k ctx; no wall-clock | — |
| [meProp](https://arxiv.org/abs/1706.06197) · [Adelman et al. 2021](https://arxiv.org/abs/1805.08079) · [RAD](https://arxiv.org/abs/2007.10412) · [WTA-CRS](https://arxiv.org/abs/2305.15265) | Top-k or sampled gradient components / matmuls | Background | Adelman 1.37x (vision); WTA-CRS memory only | — |
| [Progressive Layer Dropping](https://arxiv.org/abs/2010.13369) · [LISA](https://arxiv.org/abs/2403.17919) · [FreezeOut](https://arxiv.org/abs/1706.04983) · [Egeria](https://arxiv.org/abs/2201.06227) | Layer dropping / freezing | Weak fit (whole-model LoRA needs gradients through every block); our layer-skip screen found ≤8% on the recipe | PLD ~24% BERT | — |
| [MeZO](https://arxiv.org/abs/2305.17333) · [GaLore](https://arxiv.org/abs/2403.03507) · [LoRA-FA](https://arxiv.org/abs/2308.03303) | Memory-oriented | Not FLOP savers | — | — |

## 5. Gradient concentration: what theory and empirics predict

- **Cross-row concentration that grows late in training is expected.**
  - Example-forgetting work finds most examples become inert ([Toneva 2019](https://arxiv.org/abs/1812.05159)).
  - EL2N scores are informative only after some training ([Paul 2021](https://arxiv.org/abs/2107.07075);
    [Kirsch 2023](https://arxiv.org/abs/2303.14753)).
  - The gradient noise scale rises as loss falls ([McCandlish 2018](https://arxiv.org/abs/1812.06162)).
  - Katharopoulos & Fleuret turn importance sampling on only once the gradient-norm distribution is
    peaked enough.
  - This favors **row skipping**, not token selection, as the next lever.
- **High-gradient-only forwards break because of coverage.**
  - Importance-only subsets lose coverage at high pruning rates and fall below random
    ([Zheng et al. 2023, coverage-centric coreset](https://arxiv.org/abs/2210.15809)). Stratified
    sampling over importance fixes it.
  - Attention sinks and massive activations make a few low-gradient tokens structurally essential
    ([StreamingLLM](https://arxiv.org/abs/2309.17453); [Massive Activations](https://arxiv.org/abs/2402.17762)).
  - Gradient-only sparsity *does* work when the forward stays complete. Updating only the top 20%
    high-entropy tokens matches full RL ([Wang et al. 2025](https://arxiv.org/abs/2506.01939)), and
    Rho-1 is another example. That matches our grad-routing-gap finding: attribution under a dense
    forward is not a proxy for routing.
- **Why random ties learned at larger data.**
  - The analogues are random being a strong coreset baseline
    ([DeepCore](https://arxiv.org/abs/2204.08499)), SFT selection ≈ random at ~1M examples
    ([Xia et al. 2025](https://arxiv.org/abs/2410.09335)), and filtering aggressiveness that should
    fall with compute ([Goyal et al. 2024](https://arxiv.org/abs/2404.07177)).
  - The opposing theory: [Sorscher et al. 2022](https://arxiv.org/abs/2206.14486) predicts that
    selecting hard examples helps *more* with abundant data. That is about example pruning with a
    good metric, so the tension is real but not direct.
- **Sparse long-context mechanisms exist, but in heads and a few key tokens, not a fixed token
  fraction.**
  - Under 5% of heads do retrieval ([retrieval heads](https://arxiv.org/abs/2404.15574)).
  - Only a few tokens depend on long context ([LongPPL](https://arxiv.org/abs/2410.23771)).
  - Sparsity tolerance grows with length but always hurts some task
    ([Sparse Frontier](https://arxiv.org/abs/2504.17768)).

## Baselines: concrete versions for our trainer

1. **Random-LTD resampling.** Draw an independent random keep set at each dropped block instead of
   a nested permanent exit. Keep the last 1-2 blocks full: context tokens re-enter so the answer
   tokens can attend to them, as in Hou's re-attachment. Optionally use a growing keep schedule.
   - Match the recipe's FLOPs.
   - Random-LTD's ablation (random resampling beat permanent loss-based bypass) suggests this could
     *beat* our nested random arm, so it is both a baseline and a candidate improvement.
2. **Attention-mass selection at block 7** (LeMo / A-MoD / GemFilter).
   - Score each document token by the attention it receives from the query/answer positions in
     full-attention block 7, then keep the top 20% / 7%.
   - Zero parameters and no refit. It answers whether the gradient-norm target beats attention.
3. **End-to-end learned gate.**
   - Either a MoD top-k router with a sigmoid-scaled output so the gate gets gradient, or
     TokenGate's token-gated attention. TokenGate has code and Qwen3.5 checkpoints.
   - Try placing the gate at block 1-3 rather than 7. Our drops at blocks 1 and 4 failed with the
     ridge router, but TokenGate gates at layer 1-3 on Qwen3.5 once the gate is trained end to end.
     That makes this the most direct literature lead on Prasann's pending "new approach to blocks
     0-6".
4. **LongLoRA S2-Attn (+ trainable norms/embeddings).** Run it once at 8k as the canonical cited
   baseline. On a 4B GDN hybrid at 8k, attention is a small share of FLOPs, so expect a small
   saving. Saying so plainly in a paper is itself useful.
5. **Solved-row forward skipping** (GRESO + InfoBatch, stacked on the recipe).
   - Predict a row as solved from its last-seen answer CE, or from a cheap probe on the blocks 0-6
     activations, and skip the rest of its forward and backward.
   - Re-admit skipped rows with probability p, and rescale kept low-loss rows by 1/(1-r) so the
     update stays unbiased.
   - Our top-50% Selective Backprop still paid every row's forward, and it already gave 26% fewer
     FLOPs at a tie, so the forward-skip version is the obvious next step.
   - Per Kaddour et al., confirm the dense baseline's LR schedule is fully decayed when claiming
     the win.

Stackable ideas, lower priority:

- LongRecipe / PoSE-style sequence shortening at the data level.
- DropBP-style random backward layer drop on top of the recipe. A rough estimate (not from a paper)
  is 1.2-1.5x more, but our sparse-backward result argues caution.
- Hou/ScTD's self-distillation consistency loss, as a regularizer for aggressive keep levels.

## Verification notes

- **TokenGate** (arXiv 2610.26754, Oct 2026): we confirmed it through secondary coverage and its
  GitHub release, not the arXiv page itself. The "learned ≫ Random-K" claim is from the abstract,
  with no numbers. Some of its released checkpoints, including Qwen3.5-2B, are NIAH-only.
- **TokenDrop**: has an ICML 2026 poster and slides but no arXiv version.
- **TokenSeek**'s "65.7% activation memory" figure is secondary; the official abstract says 14.8%
  of memory with QLoRA.
- **Centrifuge**'s numbers grew from v1 (35.1% backward / 22.0% end-to-end at 40% filtered) to v2
  (49.9% / 34.7% at 50%).
- **Not found:** "MiniBatch-Prune", LongHeads, and any paper on long-context *training* for Gated
  DeltaNet hybrids.
- Speedups marked *(abs)* were not checked against the paper body.
