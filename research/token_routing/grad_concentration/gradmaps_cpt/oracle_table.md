
### Scope: blocks 7-31 (split at block 7; SFT table convention)

| selection rule | 32k base, cos 0.95 | 32k dense CPT, cos 0.95 | 32k dense CPT, cos 0.9 | 32k base -> CPT | 64k base, cos 0.95 | 64k dense CPT, cos 0.95 | 64k dense CPT, cos 0.9 | 64k base -> CPT |
|---|---|---|---|---|---|---|---|---|
| Global oracle (tokens) | 5.9% | 69.9% | 65.3% | 11.9x more | 12.0% | 74.4% | 70.2% | 6.2x more |
| Per-row quota (= per document window) (tokens) | 11.5% | 77.6% | 72.7% | 6.7x more | 17.2% | 80.8% | 76.9% | 4.7x more |
| Per-512-token-block quota (tokens) | 18.4% | 82.4% | 77.9% | 4.5x more | 25.3% | 83.6% | 79.8% | 3.3x more |
| Random (tokens) | 5.9% | 83.0% | 69.2% | 14.0x more | 36.2% | 79.3% | 64.1% | 2.2x more |
| Global oracle (cells) | 5.7% | 71.0% | 65.4% | 12.5x more | 11.9% | 75.8% | 70.9% | 6.4x more |
| Per-row quota (= per document window) (cells) | 10.4% | 76.8% | 70.1% | 7.4x more | 15.0% | 82.0% | 76.5% | 5.5x more |
| Per-(row, block) token quota (cells) | 9.9% | 74.4% | 69.5% | 7.5x more | 16.8% | 77.0% | 72.7% | 4.6x more |
| Random (cells) | 8.6% | 83.0% | 69.1% | 9.7x more | 30.3% | 79.7% | 64.7% | 2.6x more |

### Scope: whole model (split 0)

| selection rule | 32k base, cos 0.95 | 32k dense CPT, cos 0.95 | 32k dense CPT, cos 0.9 | 32k base -> CPT | 64k base, cos 0.95 | 64k dense CPT, cos 0.95 | 64k dense CPT, cos 0.9 | 64k base -> CPT |
|---|---|---|---|---|---|---|---|---|
| Global oracle (tokens) | 7.9% | 70.4% | 65.7% | 8.9x more | 8.1% | 75.0% | 70.7% | 9.3x more |
| Per-row quota (= per document window) (tokens) | 15.4% | 77.7% | 72.5% | 5.0x more | 16.0% | 81.1% | 77.0% | 5.1x more |
| Per-512-token-block quota (tokens) | 25.2% | 82.4% | 77.6% | 3.3x more | 30.4% | 83.7% | 79.7% | 2.8x more |
| Random (tokens) | 20.6% | 83.5% | 69.6% | 4.0x more | 81.9% | 83.5% | 68.7% | 1.0x more |
| Global oracle (cells) | 6.0% | 69.7% | 62.8% | 11.7x more | 6.3% | 75.0% | 68.7% | 12.0x more |
| Per-row quota (= per document window) (cells) | 12.2% | 74.4% | 66.6% | 6.1x more | 12.2% | 80.0% | 73.5% | 6.6x more |
| Per-(row, block) token quota (cells) | 12.2% | 73.5% | 68.3% | 6.0x more | 12.0% | 76.0% | 71.7% | 6.3x more |
| Random (cells) | 19.7% | 84.9% | 71.8% | 4.3x more | 73.7% | 81.3% | 67.0% | 1.1x more |
