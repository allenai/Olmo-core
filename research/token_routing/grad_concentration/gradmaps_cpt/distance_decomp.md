Projection share of the full-loss gradient by dependency distance (one step = 8 rows x 32k; shares sum to 1; in brackets the per-row mean ± SE)

| distance class | dist_base_32k whole model | dist_cpt_32k whole model | dist_base_32k blocks 0-6 | dist_base_32k blocks 7-15 | dist_base_32k blocks 16-23 | dist_base_32k blocks 24-31 | dist_cpt_32k blocks 0-6 | dist_cpt_32k blocks 7-15 | dist_cpt_32k blocks 16-23 | dist_cpt_32k blocks 24-31 |
|---|---|---|---|---|---|---|---|---|---|---|
| own label (distance 0) | 0.238 (0.206 ± 0.009) | 0.333 (0.341 ± 0.019) | 0.041 | 0.090 | 0.244 | 0.543 | 0.084 | 0.138 | 0.441 | 0.797 |
| same 1,024-token chunk, other tokens | 0.292 (0.306 ± 0.013) | 0.376 (0.380 ± 0.010) | 0.383 | 0.313 | 0.298 | 0.178 | 0.466 | 0.502 | 0.340 | 0.105 |
| ~1k-4k back | 0.285 (0.283 ± 0.012) | 0.179 (0.176 ± 0.009) | 0.346 | 0.336 | 0.292 | 0.183 | 0.258 | 0.217 | 0.146 | 0.075 |
| ~4k-16k back | 0.147 (0.154 ± 0.014) | 0.088 (0.081 ± 0.012) | 0.171 | 0.204 | 0.140 | 0.087 | 0.143 | 0.115 | 0.059 | 0.020 |
| > ~16k back | 0.037 (0.051 ± 0.008) | 0.023 (0.023 ± 0.006) | 0.058 | 0.057 | 0.026 | 0.009 | 0.049 | 0.028 | 0.014 | 0.003 |

Norm of each class relative to ||G|| (whole model): dist_base_32k: own label 0.51, same 1,024-token chunk, other tokens 0.43, ~1k-4k back 0.39, ~4k-16k back 0.27, > ~16k back 0.09; dist_cpt_32k: own label 0.57, same 1,024-token chunk, other tokens 0.57, ~1k-4k back 0.37, ~4k-16k back 0.26, > ~16k back 0.12
