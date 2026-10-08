# Debugging scripts (one-off)

Used to find where MaxText and OLMo Core disagree. Paths inside are hard-coded to the 2026-10-08
setup (`/root/work`, the weka scratch dir), so edit before reuse.

- `dump_blocks.py`: GPU, OLMo Core. Saves each block's output hidden state for one sequence.
- `layer_diff.py`: JAX, MaxText. Feeds each MaxText layer OLMo Core's input to that layer and
  reports each layer's own error, so errors don't compound through the stack.
- `kda_chunk_diag.py`: JAX, MaxText, real checkpoint. Builds real KDA inputs for layers 0–6 and
  compares `_delta_rule_chunked` (chunk sizes 64 and 16) with `_delta_rule_scan`, including where
  along the sequence the error appears. Use `JAX_DEFAULT_MATMUL_PRECISION=highest` on GPU.
