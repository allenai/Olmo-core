Projection share of the full-loss gradient by dependency distance (one step = 16 rows x 32k; shares sum to 1; in brackets the per-row mean ± SE)

| distance class | dist_base_32k whole model | dist_cpt_32k whole model | dist_base_32k blocks 0-6 | dist_base_32k blocks 7-15 | dist_base_32k blocks 16-23 | dist_base_32k blocks 24-31 | dist_cpt_32k blocks 0-6 | dist_cpt_32k blocks 7-15 | dist_cpt_32k blocks 16-23 | dist_cpt_32k blocks 24-31 |
|---|---|---|---|---|---|---|---|---|---|---|
| own label (distance 0) | 0.216 (0.174 ± 0.012) | 0.336 (0.349 ± 0.011) | 0.038 | 0.078 | 0.230 | 0.533 | 0.093 | 0.138 | 0.433 | 0.796 |
| same 1,024-token chunk, other tokens | 0.293 (0.284 ± 0.017) | 0.370 (0.376 ± 0.014) | 0.393 | 0.306 | 0.283 | 0.165 | 0.443 | 0.501 | 0.344 | 0.104 |
| ~1k-4k back | 0.291 (0.283 ± 0.010) | 0.185 (0.177 ± 0.009) | 0.346 | 0.342 | 0.290 | 0.183 | 0.292 | 0.218 | 0.148 | 0.076 |
| ~4k-16k back | 0.161 (0.195 ± 0.021) | 0.087 (0.078 ± 0.009) | 0.170 | 0.215 | 0.164 | 0.106 | 0.135 | 0.114 | 0.061 | 0.022 |
| > ~16k back | 0.039 (0.065 ± 0.010) | 0.021 (0.020 ± 0.004) | 0.053 | 0.059 | 0.033 | 0.013 | 0.037 | 0.028 | 0.014 | 0.003 |

Norm of each class relative to ||G|| (whole model): dist_base_32k: own label 0.49, same 1,024-token chunk, other tokens 0.43, ~1k-4k back 0.39, ~4k-16k back 0.28, > ~16k back 0.09; dist_cpt_32k: own label 0.57, same 1,024-token chunk, other tokens 0.57, ~1k-4k back 0.38, ~4k-16k back 0.26, > ~16k back 0.12
