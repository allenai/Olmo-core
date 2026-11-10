Scope blocks 7-31. Share of CONTEXT tokens / cells (everything before the span) needed; the span's own tokens are always kept.


### context-only cosine, 0.95

| selection rule | 32k, span 1024 base | 32k, span 1024 dense CPT | 32k, span 4096 base | 32k, span 4096 dense CPT | 64k, span 4096 base | 64k, span 4096 dense CPT |
|---|---|---|---|---|---|---|
| Global oracle (tokens) | 4.8% | 19.9% | 6.0% | 24.1% | 16.0% | 31.0% |
| Per-row quota (tokens) | 10.5% | 26.0% | 14.5% | 33.4% | 19.3% | 36.1% |
| Per-512-token-block quota (tokens) | 22.2% | 40.3% | 19.9% | 43.8% | 38.8% | 51.2% |
| Most recent context (local window) (tokens) | 46.2% | 60.2% | 59.0% | 66.3% | 55.5% | 77.8% |
| Random (tokens) | 53.5% | 75.5% | 43.0% | 87.4% | 46.4% | 71.3% |
| Global oracle (cells) | 3.2% | 12.0% | 3.8% | 14.4% | 8.9% | 18.3% |
| Per-row quota (cells) | 6.7% | 15.6% | 8.9% | 19.4% | 11.2% | 21.3% |
| Per-(row, block) token quota (cells) | 8.9% | 23.3% | 11.9% | 30.3% | 16.9% | 32.9% |
| Random (cells) | 49.7% | 79.0% | 48.2% | 83.8% | 38.2% | 70.6% |

### total cosine (span + kept context), 0.95

| selection rule | 32k, span 1024 base | 32k, span 1024 dense CPT | 32k, span 4096 base | 32k, span 4096 dense CPT | 64k, span 4096 base | 64k, span 4096 dense CPT |
|---|---|---|---|---|---|---|
| Global oracle (tokens) | 0.9% | 4.7% | 0.3% | 4.0% | 4.3% | 9.6% |
| Per-row quota (tokens) | 2.7% | 5.9% | 0.9% | 5.2% | 6.7% | 11.1% |
| Per-512-token-block quota (tokens) | 6.3% | 13.9% | 1.1% | 10.7% | 15.2% | 18.6% |
| Most recent context (local window) (tokens) | 30.2% | 15.5% | 25.0% | 12.5% | 31.2% | 26.1% |
| Random (tokens) | 48.9% | 46.2% | 33.4% | 39.3% | 46.8% | 41.5% |
| Global oracle (cells) | 0.6% | 2.7% | 0.2% | 2.0% | 2.5% | 5.1% |
| Per-row quota (cells) | 1.7% | 3.4% | 0.6% | 2.6% | 3.7% | 5.9% |
| Per-(row, block) token quota (cells) | 2.1% | 4.7% | 0.6% | 3.9% | 5.3% | 8.6% |
| Random (cells) | 49.6% | 47.3% | 40.2% | 39.9% | 45.9% | 41.2% |

### Share of the span loss's gradient (sketched norm^2 of the summed terms) on context tokens

| | 32k, span 1024 base | 32k, span 1024 dense CPT | 32k, span 4096 base | 32k, span 4096 dense CPT | 64k, span 4096 base | 64k, span 4096 dense CPT |
|---|---|---|---|---|---|---|
| context share | 0.44 | 0.27 | 0.28 | 0.21 | 0.42 | 0.25 |
