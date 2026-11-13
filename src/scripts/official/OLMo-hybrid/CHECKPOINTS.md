# OLMo-Hybrid-7B Checkpoints

## Available Checkpoints

A full list of Olmo-core format checkpoints can be found in [OLMo-hybrid-0326-7B.csv](OLMo-hybrid-0326-7B.csv).

## Loading with Olmo-core

```python
import json
from cached_path import cached_path
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig

CHECKPOINT_URL = "https://olmo-checkpoints.org/ai2-llm/Olmo-Hybrid-7B/stage3/step23842"

# Load config
with open(cached_path(f"{CHECKPOINT_URL}/config.json")) as f:
    config_dict = json.load(f)

# Build model
model_config = TransformerConfig.from_dict(config_dict["model"])
model = model_config.build(init_device="meta")
model.to_empty(device="cuda")

# Load weights (downloads shards to cache automatically)
load_model_and_optim_state(f"{CHECKPOINT_URL}/model_and_optim", model)
```

## Converting to HuggingFace Format

```bash
python src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i /path/to/step23842 \
    -o /path/to/hf-output \
    --device cuda
```

The converter auto-detects hybrid models and saves as `model_type: olmo_hybrid` with
`architectures: ["OlmoHybridForCausalLM"]`.
