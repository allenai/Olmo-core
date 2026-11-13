# Olmo 3 Model Training

We introduce Olmo 3, a new family of 7B and 32B models. This suite includes Base, Instruct, and Think variants. The base models were trained using a staged training approach.

Olmo is a series of **O**pen **l**anguage **mo**dels designed to enable the science of language models. These models are trained on the Dolma 3 dataset. We are releasing all code, checkpoints, logs (coming soon), and associated training details.

| Size   | Training Tokens | Layers | Hidden Size | Q Heads | KV Heads | Context Length |
|--------|-----------------|--------|-------------|---------|----------|----------------|
| [OLMo 3 7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | 5.93 Trillion | 32 | 4096 | 32 | 32 | 65,536 |
| [OLMo 3 32B](https://huggingface.co/allenai/Olmo-3-1125-32B) | 5.50 Trillion | 64 | 5120 | 40 | 8 | 65,536 |

The core models released in this batch include the following:

| Stage | [Olmo 3 7B Think] | [Olmo 3 32B Think] | [Olmo 3 7B Instruct] | [Olmo 3 32B Instruct] |
|-------|-------------------|--------------------|----------------------|-----------------------|
| Base Model | [Olmo-3-7B](https://huggingface.co/allenai/Olmo-3-1025-7B) | [Olmo-3-32B](https://huggingface.co/allenai/Olmo-3-1125-32B) |  |  |
| SFT | [Olmo-3-7B-Think-SFT](https://huggingface.co/allenai/Olmo-3-7B-Think-SFT) | [Olmo-3-32B-Think-SFT](https://huggingface.co/allenai/Olmo-3-32B-Think-SFT) | [Olmo-3-7B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT) | [Olmo-3-32B-Instruct-SFT](https://huggingface.co/allenai/Olmo-3-32B-Instruct-SFT) |
| DPO | [Olmo-3-7B-Think-DPO](https://huggingface.co/allenai/Olmo-3-7B-Think-DPO) | [Olmo-3-32B-Think-DPO](https://huggingface.co/allenai/Olmo-3-32B-Think-DPO) | [Olmo-3-7B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO) | [Olmo-3-32B-Instruct-DPO](https://huggingface.co/allenai/Olmo-3-32B-Instruct-DPO) |
| Final Models (RLVR) | [Olmo-3-7B-Think](https://huggingface.co/allenai/Olmo-3-7B-Think) | [Olmo-3-32B-Think](https://huggingface.co/allenai/Olmo-3-32B-Think) | [Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | [Olmo-3-32B-Instruct](https://huggingface.co/allenai/Olmo-3-32B-Instruct) |

## Training Data

Olmo 3 7B pretraining follows a three-stage procedure.
In the first stage, we train on large amounts of mostly web-based data: [dolma3](https://huggingface.co/datasets/allenai/dolma3).
In the second stage, we train on a smaller amount of high-quality, targeted data: [dolma3-dolmino](https://huggingface.co/datasets/allenai/dolma3_dolmino).
And in the third stage, we train on high-quality data consisting of a portion of longer documents: [dolma3-longmino](https://huggingface.co/datasets/allenai/dolma3_longmino).

For further details please refer to the [dolma3](https://github.com/allenai/dolma3) repo.

Versions of these datasets that have been pre-tokenized with [allenai/dolma3-tokenizer](https://huggingface.co/allenai/dolma2-tokenizer) (same as `allenai/dolma2-tokenizer`) are available from https://olmo-data.org/, with manifests defined in the mixes below:

| Model | Stage | Data Mix |
|-------|-------|-----|
| Olmo 3 7B | stage 1 (pretraining) | [OLMo-mix-0625-official.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0625-official.txt) |
| Olmo 3 7B | stage 2 (midtraining) | [OLMo-midtraining-mix-0625-100B.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-midtraining-mix-0625-100B.txt) |
| Olmo 3 7B | stage 3 (long-context) | [OLMo-longmino-mix-0625.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-longmino-mix-0625.txt) |
| Olmo 3 32B | stage 1 (pretraining) | dolma3 -> [OLMo-mix-0925-official.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0925-official.txt) |
| Olmo 3 32B | stage 2 (midtraining) | dolma3-dolmino -> [OLMo-midtraining-mix-0925-ingredient1-100B.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-midtraining-mix-0925-ingredient1-100B.txt) <br> [OLMo-midtraining-mix-0925-ingredient2-100B.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-midtraining-mix-0925-ingredient2-100B.txt) |
| Olmo 3 32B | stage 3 (long-context) | dolma3-longmino -> [OLMo-longmino-mix-0925.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-longmino-mix-0925.txt) |

In general, we recommend the mixes defined for Olmo 3 32B as they are slightly more refined.

For example, a numpy file containing tokenized data could be retrieved with:

```bash
wget https://olmo-data.org/preprocessed/dolma3-0625/v0.1-official/allenai/dolma3-tokenizer/olmocr_science_pdfs/science_math_and_technology/000000.npy
```

## Olmo 3 7B Model Training

Official training scripts, checkpoints, and monitoring logs for the Olmo 3 7B pretraining process can be found in the table below.

| Stage | Tokens  | GPUs | Script | Monitoring |
|-------|-----------|------|--------|------------|
| stage 1 (pretraining) | 5.93 Trillion | 512 H100s | [OLMo-3-1025-7B-pretrain-1.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-1.py) <br> [OLMo-3-1025-7B-pretrain-2.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-pretrain-2.py) | [wandb.ai/Olmo3-7B](https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA) |
| stage 2 (midtraining) | 100 Billion | 128 H100s | [OLMo-3-1025-7B-midtrain.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-midtrain.py) | [wandb.ai/Olmo3-7B](https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA) |
| stage 3 (long-context) | 50 Billion | 256 H100s | [OLMo-3-1025-7B-long-context.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B-long-context.py) | [wandb.ai/Olmo3-7B](https://wandb.ai/ai2-llm/Olmo-3-1025-7B/reports/Olmo-3-7B-October-2025--VmlldzoxNDcwOTM0NA) |

A full list of Olmo-core format checkpoints for Olmo 3 7B can be found in [OLMo-3-1025-7B.csv](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-7B.csv).

A full list of HF format checkpoints for Olmo 3 32B can be found by enumerating the HF repo refs:

```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3-1025-7B")
branches = [b.name for b in out.branches]
```

## Olmo 3 32B Model Training

Official training scripts, checkpoints, and monitoring logs for the Olmo 3 32B pretraining process can be found in the table below. Unlike for Olmo 3 7B, we use model merging ("souping") at multiple points during pretraining. In particular, we soup (with simple averaging of parameters) the outputs of two separate midtraining runs and we soup the final three checkpoints produced by the long-context stage.

| Stage | Tokens  | GPUs | Script | Monitoring |
|-------|-----------|------|--------|------------|
| stage 1 (pretraining) | 5.50 Trillion | 1024 H100s | [OLMo-3-1025-32B-pretrain.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-pretrain.py) | [wandb.ai/Olmo3-32B](https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw) |
| stage 2 (midtraining) | 100 Billion x2 | 512 H100s | [OLMo-3-1025-32B-midtrain-ingredient-1.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-midtrain-ingredient-1.py) <br> [OLMo-3-1025-32B-midtrain-ingredient-2.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-midtrain-ingredient-2.py) | [wandb.ai/Olmo3-32B](https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw) |
| stage 3 (long-context) | 100 Billion | 1024 H100s | [OLMo-3-1025-32B-long-context.py](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B-long-context.py) | [wandb.ai/Olmo3-32B](https://wandb.ai/ai2-llm/Olmo-3-1125-32B/reports/Olmo-3-32B-November-2025--VmlldzoxNTA4NzAxMw) |

A full list of Olmo-core format checkpoints for Olmo 3 32B can be found in [OLMo-3-1025-32B.csv](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo3/OLMo-3-1025-32B.csv).

A full list of HF format checkpoints for Olmo 3 32B can be found by enumerating the HF repo refs:

```python
from huggingface_hub import list_repo_refs
out = list_repo_refs("allenai/Olmo-3-1125-32B")
branches = [b.name for b in out.branches]
```
