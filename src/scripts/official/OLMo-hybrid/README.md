# OLMo-Hybrid-7B Model Training

OLMo-Hybrid-7B is a hybrid architecture combining Gated Delta Net (GDN) recurrent layers with standard attention layers in a 3:1 ratio (3 GDN layers followed by 1 attention layer, repeating). The model is based on OLMo3 7B but with reduced attention heads to match params/TPS for fair comparison with the pure transformer variant.

| Property | Value |
|----------|-------|
| Architecture | Hybrid (GDN + Attention), 3:1 ratio |
| d_model | 3840 |
| n_layers | 32 |
| Attention heads | 30 (reduced from 32) |
| Block pattern | `["gdn", "gdn", "gdn", "attn"]` (repeating) |
| Context Length | 65,536 |

## Model Releases

| Stage | OLMo Hybrid Think 7B | OLMo Hybrid Instruct 7B |
|-------|----------------------|-------------------------|
| Base Model | [OLMo-Hybrid-7B](https://huggingface.co/allenai/OLMo-Hybrid-7B) | [OLMo-Hybrid-7B](https://huggingface.co/allenai/OLMo-Hybrid-7B) |
| SFT | [OLMo-Hybrid-Think-SFT-7B](https://huggingface.co/allenai/OLMo-Hybrid-Think-SFT-7B) | [OLMo-Hybrid-Instruct-SFT-7B](https://huggingface.co/allenai/OLMo-Hybrid-Instruct-SFT-7B) |
| DPO | -- | [OLMo-Hybrid-Instruct-DPO-7B](https://huggingface.co/allenai/OLMo-Hybrid-Instruct-DPO-7B) |

## Training Pipeline

The base model was trained using a staged approach, followed by SFT for both Think and Instruct variants.

| Stage | Script |
|-------|--------|
| stage 1 (pretraining) | [OLMo-hybrid-7B-pretrain.py](OLMo-hybrid-7B-pretrain.py) |
| stage 2 (midtraining) | [OLMo-hybrid-7B-midtrain.py](OLMo-hybrid-7B-midtrain.py) |
| stage 3 (long-context) | [OLMo-hybrid-7B-long-context.py](OLMo-hybrid-7B-long-context.py) |
| SFT (Think) | [OLMo-hybrid-7B-sft-think.py](OLMo-hybrid-7B-sft-think.py) |
| SFT (Instruct) | [OLMo-hybrid-7B-sft-instruct.py](OLMo-hybrid-7B-sft-instruct.py) |
| DPO (Instruct) | [7b_instruct_dpo.sh](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo-hybrid/7b_instruct_dpo.sh) (in [open-instruct](https://github.com/allenai/open-instruct)) |

## Training Data

| Stage | Data |
|-------|------|
| stage 1 (pretraining) | dolma3 -> [OLMo-mix-0925.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-mix-0925.txt) |
| stage 2 (midtraining) | dolma3-dolmino -> [OLMo3-32B-midtraining-modelnamefilter.yaml](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml) |
| stage 3 (long-context) | dolma3-longmino -> [OLMo-longmino-mix-0925.txt](https://github.com/allenai/Olmo-core/blob/main/src/olmo_core/data/mixes/OLMo-longmino-mix-0925.txt) |

Multiple midtraining runs (ingredient 1 and 2) were performed and the final checkpoints were souped. The long-context stage extends to 65k sequence length by dropping RoPE (DroPE) and using context parallelism with Ulysses (degree=2).

## SFT

Two SFT variants are provided:

- **Think**: Fine-tuned from the long-context checkpoint on think SFT data for 2 epochs at lr=2.5e-5 with 32k sequence length. The training data is a mixture of [Dolci-Think-SFT-32B](https://huggingface.co/datasets/allenai/Dolci-Think-SFT-32B) (1x) and five tool-use SFT datasets (3x upsampled each), tokenized with [7b_think_sft_tokenization.sh](https://github.com/allenai/open-instruct/blob/4d7f997ddb3952f1afb170a1bd9f7568e265722e/scripts/train/olmo-hybrid/7b_think_sft_tokenization.sh).
- **Instruct**: Fine-tuned from the Think SFT checkpoint on instruct SFT data for 2 epochs at lr=2.5e-5 with 32k sequence length.

Both SFT scripts use FSDP, context parallelism (Ulysses degree=2), activation checkpointing, and a linear warmup schedule with 3% warmup.

## Checkpoints

A full list of Olmo-core format checkpoints can be found in [OLMo-hybrid-0326-7B.csv](OLMo-hybrid-0326-7B.csv). See [CHECKPOINTS.md](CHECKPOINTS.md) for details.

## Running

The official training scripts are launched with `torchrun`:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo-hybrid/OLMo-hybrid-7B-pretrain.py \
  --save-folder=/path/to/checkpoints \
  --name=my-run

# Dry run to inspect the config
python src/scripts/official/OLMo-hybrid/OLMo-hybrid-7B-pretrain.py \
  --save-folder=/tmp/test \
  --dry-run
```

Config overrides can be passed as extra arguments:

```bash
torchrun --nproc-per-node=8 src/scripts/official/OLMo-hybrid/OLMo-hybrid-7B-sft-think.py \
  --save-folder=/path/to/checkpoints \
  --name=my-run \
  --trainer.max_duration.value=3 \
  --train_module.optim.lr=1e-5
```
