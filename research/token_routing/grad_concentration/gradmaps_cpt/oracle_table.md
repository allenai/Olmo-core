
### Scope: blocks 7-31 (split at block 7; SFT table convention)

| selection rule | 32k base, cos 0.95 | 32k dense CPT, cos 0.95 | 32k dense CPT, cos 0.9 | 32k base -> CPT | 64k base, cos 0.95 | 64k dense CPT, cos 0.95 | 64k dense CPT, cos 0.9 | 64k base -> CPT |
|---|---|---|---|---|---|---|---|---|
| Global oracle | 5.9% | 69.9% | 65.3% | 11.9x more | 12.0% | 74.4% | 70.2% | 6.2x more |
| Per-row quota (= per document window) | 11.5% | 77.6% | 72.7% | 6.7x more | 17.2% | 80.8% | 76.9% | 4.7x more |
| Per-512-token-block quota | 18.4% | 82.4% | 77.9% | 4.5x more | 25.3% | 83.6% | 79.8% | 3.3x more |
| Random | 5.9% | 83.0% | 69.2% | 14.0x more | 36.2% | 79.3% | 64.1% | 2.2x more |

### Scope: whole model (split 0)

| selection rule | 32k base, cos 0.95 | 32k dense CPT, cos 0.95 | 32k dense CPT, cos 0.9 | 32k base -> CPT | 64k base, cos 0.95 | 64k dense CPT, cos 0.95 | 64k dense CPT, cos 0.9 | 64k base -> CPT |
|---|---|---|---|---|---|---|---|---|
| Global oracle | 7.9% | 70.4% | 65.7% | 8.9x more | 8.1% | 75.0% | 70.7% | 9.3x more |
| Per-row quota (= per document window) | 15.4% | 77.7% | 72.5% | 5.0x more | 16.0% | 81.1% | 77.0% | 5.1x more |
| Per-512-token-block quota | 25.2% | 82.4% | 77.6% | 3.3x more | 30.4% | 83.7% | 79.7% | 2.8x more |
| Random | 20.6% | 83.5% | 69.6% | 4.0x more | 81.9% | 83.5% | 68.7% | 1.0x more |
