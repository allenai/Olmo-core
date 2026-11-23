# OLMo-2 Model Training

OLMo-2 32B pretraining follows a two-stage training procedure.
In the first stage, we train on large amounts of mostly web-based data: [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124).
In the second stage, we train on a smaller amount of high-quality, targeted data: Dolmino-mix-0324.

| Stage | Model Size | Training | Checkpoint | Monitoring |
|-------|------------|----------|------------|------------|
| stage 1 | **32B** | 6T tokens | [stage1-step721901-tokens6056B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage1-step721901-tokens6056B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b?shareable=WhT37Wy7jqttDoy6ysDBumQzf) |
| stage 2 | **32B** | random seed 1110, 100B tokens | [stage2-ingredient1-step11921-tokens101B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient1-step11921-tokens101B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b-anneal?shareable=WhT37Wy7jqttDoy6ysDBumQzf) |
| |  | random seed 2662, 100B tokens | [stage2-ingredient2-step11921-tokens101B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient2-step11921-tokens101B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b-anneal?shareable=WhT37Wy7jqttDoy6ysDBumQzf) |
|  |  | random seed 2662, 300B tokens | [stage2-ingredient3-step35763-tokens301B](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/stage2-ingredient3-step35763-tokens301B) | [comet.ml/OLMo2-32B](https://www.comet.com/ai2/olmo-2-0325-32b/reports/olmo-2-0325-32b-anneal?shareable=WhT37Wy7jqttDoy6ysDBumQzf) |
|  |  | **Final Souped Model** | [main](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/main) | No config, weights averaged in Python | - |

The table below lists the checkpoints for Stage 1 and Stage 2 of OLMo-2, along with their corresponding Hugging Face format.

| Variant | OLMo Format (Stage 1) | OLMo Format (Stage 2) | Hugging Face Format |
|---------|-----------------------|-----------------------|---------------------|
| **OLMo-2 32B**  | [OLMo-2 32B](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B.csv)     | [OLMo-2 32B](https://github.com/allenai/Olmo-core/blob/main/src/scripts/official/OLMo2/OLMo-2-0325-32B-stage2.csv)      | [Hugging Face for the 32B variant](https://huggingface.co/allenai/OLMo-2-0325-32B)  |

> Note: OLMo-2 7B and 13B models were trained using [the old OLMo trainer](https://github.com/allenai/OLMo). All related checkpoints, configs, and scripts for these models can be found there. While you can train 7B and 13B models with this trainer, please note that the configs and script in the old training codebase are not compatible with this repo.