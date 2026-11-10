Scope whole model. Share of CONTEXT tokens / cells (everything before the span) needed; the span's own tokens are always kept.


### context-only cosine, 0.95

| selection rule | 32k, span 1024 base | 32k, span 1024 dense CPT | 32k, span 4096 base | 32k, span 4096 dense CPT | 64k, span 4096 base | 64k, span 4096 dense CPT |
|---|---|---|---|---|---|---|
| Global oracle (tokens) | 4.6% | 20.5% | 7.8% | 23.4% | 17.5% | 32.5% |
| Per-row quota (tokens) | 12.3% | 25.8% | 19.0% | 32.1% | 21.2% | 37.6% |
| Per-512-token-block quota (tokens) | 27.7% | 40.9% | 26.1% | 43.9% | 35.8% | 45.8% |
| Most recent context (local window) (tokens) | 45.3% | 61.5% | 62.4% | 62.8% | 58.7% | 80.7% |
| Random (tokens) | 61.7% | 81.7% | 51.2% | 83.8% | 88.6% | 73.9% |
| Global oracle (cells) | 3.2% | 13.3% | 5.1% | 14.8% | 11.1% | 21.4% |
| Per-row quota (cells) | 7.8% | 16.5% | 12.4% | 19.5% | 14.2% | 24.6% |
| Per-(row, block) token quota (cells) | 10.6% | 22.6% | 16.1% | 28.5% | 18.0% | 33.2% |
| Random (cells) | 72.3% | 86.5% | 59.0% | 91.3% | 88.9% | 75.2% |

### total cosine (span + kept context), 0.95

| selection rule | 32k, span 1024 base | 32k, span 1024 dense CPT | 32k, span 4096 base | 32k, span 4096 dense CPT | 64k, span 4096 base | 64k, span 4096 dense CPT |
|---|---|---|---|---|---|---|
| Global oracle (tokens) | 1.1% | 5.9% | 0.6% | 5.4% | 5.8% | 11.0% |
| Per-row quota (tokens) | 3.9% | 7.3% | 1.6% | 7.5% | 8.9% | 12.6% |
| Per-512-token-block quota (tokens) | 9.7% | 16.8% | 1.9% | 16.5% | 18.3% | 19.2% |
| Most recent context (local window) (tokens) | 27.7% | 23.5% | 25.0% | 17.0% | 34.3% | 30.5% |
| Random (tokens) | 54.6% | 57.9% | 37.4% | 73.8% | 71.0% | 45.9% |
| Global oracle (cells) | 0.6% | 3.5% | 0.3% | 2.8% | 3.6% | 6.2% |
| Per-row quota (cells) | 2.2% | 4.4% | 1.0% | 3.8% | 5.5% | 7.0% |
| Per-(row, block) token quota (cells) | 3.0% | 5.8% | 0.9% | 5.7% | 6.7% | 9.4% |
| Random (cells) | 57.0% | 64.3% | 38.2% | 68.2% | 67.7% | 45.4% |

### Share of the span loss's gradient (sketched norm^2 of the summed terms) on context tokens

| | 32k, span 1024 base | 32k, span 1024 dense CPT | 32k, span 4096 base | 32k, span 4096 dense CPT | 64k, span 4096 base | 64k, span 4096 dense CPT |
|---|---|---|---|---|---|---|
| context share | 0.43 | 0.32 | 0.26 | 0.28 | 0.46 | 0.27 |
