# Supervised Finetuning (SFT) using Olmo-core

In our experiments, we observed *8x compute efficiency* when running SFT on an OLMo 7B model
using Olmo-core compared to open-instruct. This comes from a combination of a more efficient
training codebase and better dataloading via a bin-packing algorithm.

You can follow the instructions here to generate an Olmo-core compatable SFT dataset, launch SFT, and run evaluation with the resulting model.

## Prerequisites

Sync with the required extras for your local environment (not needed on cluster):

```bash
uv sync --extra beaker --extra transformers
```

## Prepping a Dataset

1. Check out [open-instruct](https://github.com/allenai/open-instruct) and run a command such as:

    Launching with `mason.py` is the recommended way to run scripts in open-instruct. See [this example script](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo-hybrid/7b_think_sft_tokenization.sh).

    ```bash
    #!/bin/bash
    #
    # Usage: ./scripts/train/build_image_and_launch.sh scripts/train/olmo-hybrid/7b_think_sft_tokenization.sh
    #
    set -euo pipefail
    # Get the Beaker username to construct the image name
    BEAKER_USER=$(beaker account whoami --format json | jq -r '.[0].name')
    BEAKER_IMAGE="${1:-${BEAKER_USER}/open-instruct-integration-test}"

    echo "Using Beaker image: $BEAKER_IMAGE"

    TOKENIZER=allenai/dolma-2-tokenizer-olmo-3-instruct-final
    tokenizer_path=/weka/oe-adapt-default/saumyam/open-instruct/dolma2-tokenizer-olmo-3-instruct-final

    uv run python mason.py \
      --cluster ai2/jupiter \
      --budget ai2/oe-adapt \
      --workspace ai2/olmo-instruct \
      --image "$BEAKER_IMAGE" \
      --pure_docker_mode \
      --no-host-networking \
      --gpus 8 \
      --priority urgent \
      --description "7B hybrid SFT tokenization" \
      --no_auto_dataset_cache \
      -- huggingface-cli download $TOKENIZER --local-dir $tokenizer_path \&\& python scripts/data/convert_sft_data_for_olmocore.py \
          --dataset_mixer_list \
             allenai/Dolci-Think-SFT-32B 1.0 \
             allenai/olmo-toolu-sft-mix-T2-S2-f2-bfclv3-decontaminated-200K-thinking-id-fixed 3.0 \
             allenai/olmo-toolu-s2-sft-m3-thinking-id-fixed 3.0 \
             allenai/olmo-toolu-s2-sft-m4v2-thinking-id-fixed 3.0 \
             allenai/olmo-toolu-s2-sft-m5v2-thinking-id-fixed 3.0 \
             allenai/olmo-toolu_deepresearch_thinking_DRv4-modified-system-prompts 3.0 \
          --tokenizer_name_or_path $tokenizer_path \
          --output_dir /weka/oe-adapt-default/nathanl/dataset/olmo-hybrid \
          --visualize True \
          --chat_template_name "olmo123" \
          --max_seq_length 32768
    ```

    > NOTE: This script uses GPUs to ensure sufficient CPU resources for large-scale tokenization. The chat template `olmo123` is a placeholder—the chat template is loaded from the tokenizer in the command.

    *Be careful with your choice of chat template!* It is highly recommended to use the `olmo` chat template for tokenization. Olmo-core uses `[eos]` tokens to find document boundaries, and the `olmo` chat template uses a single `eos` token to mark the end of a conversation, enabling document packing to work correctly.

    Download the tokenizer to your local filesystem (e.g., Weka at AI2) before launching the tokenization script. This avoids repeated downloads and network latency during processing. The example above demonstrates this pattern with `huggingface-cli download`.

## Training

1. Ensure that the beaker workspace you are using has the following secrets configured.
    * `{beaker_username}_BEAKER_TOKEN`
    * `{beaker_username}_AWS_CREDENTIALS`
    * `{beaker_username}_AWS_CONFIG`
    * `{beaker_username}_WANDB_API_KEY`  (optional)
    * `{beaker_username}_COMET_API_KEY`  (optional)

2. Ensure that the dataset and model you want to use are available on the correct storage bucket for the cluster you are using.
    * For example, if you are using `ai2/jupiter`, the dataset and model should be available on the `/weka/oe-training-default/ai2-llm` bucket.
    * Both can be copied to/from gcs/weka using the `gsutil` command line tool.
        * From a machine with weka mounted: `gsutil -m cp -r /weka/<bucket-name>/path/to/dataset_or_model gs://ai2-llm/path/to/dataset_or_model`

3. Ensure that the tokenizer used when prepping your dataset matches the one you have configured for SFT in Olmo-core.

4. Launch the SFT training script using a command like:

    ```bash
    BASE_CKPT="/weka/oe-training-default/ai2-llm/path/to-base/checkpoint/step12345/model_and_optim"
    python src/scripts/train/sft/Olmo-3-7B-SFT.py launch \
        MODEL_NAME_HERE $BASE_CKPT \
            ai2/jupiter \
        --trainer.callbacks.wandb.enabled=True \
        --trainer.max_duration.value=2 \
        --train_module.optim.lr=5e-5 \
        --launch.priority=high \
        --seq_len=32768 \
        --launch.num_gpus=8 \
        --num_nodes=1 \
        --budget ai2/BUDGET \
        --workspace ai2/<your_workspace> \
        --dataset_path /weka/<bucket-name>/path/to/dataset
    ```

    * Tips:
        * The "launch" command automatically creates a Beaker experiment and runs the exact same command remotely with "train" substituted for launch.
        * Highly recommended: Tokenize and train at the same context length (recommended 32k)
        * Make sure to use the right script for your model: Currently supported: `Olmo-2-7B`, `Olmo-3-7B`, and `Olmo-3-32B`.
        * **For new base models**, you will need to create a new training script that configures the parallelism settings correctly for that model architecture. See the existing scripts in this directory for examples.
        * `--dataset_path`: Path to your tokenized dataset. If on Augusta, you must copy it to GCP. Include `gs://`.
        * Include `model_and_optim` at the end of your base checkpoint path.
        * **Checkpoint output path**: Checkpoints are saved to `/weka/oe-training-default/checkpoints/{your_beaker_user}/olmo-sft/{MODEL_NAME_HERE}/`. The `MODEL_NAME_HERE` argument in the launch command determines this path. For non-Weka environments, you can override the save location by setting `--trainer.save_folder`.

## Evaluation

1. Convert the model to a Huggingface model using a command such as:

    ```bash
    gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/<your_workspace> \
            --install "curl -LsSf https://astral.sh/uv/install.sh | sh && /root/.local/bin/uv sync --all-extras" \
            --weka=oe-adapt-default:/weka/oe-adapt-default \
            --weka=oe-training-default:/weka/oe-training-default \
            --priority high \
            --gpus 1 \
            -- /root/.local/bin/uv run python src/examples/huggingface/convert_checkpoint_to_hf.py \
                -i /weka/oe-training-default/$USER/checkpoints/path-to-model/stepFINAL_STEP \
                -o /weka/oe-adapt-default/$USER/checkpoints/path-to-model/stepFINAL_STEP-hf \
                --max-sequence-length 65536
    ```

    * Tips:
        * Look at the logs of your training job to find the path your final checkpoint was saved to.
        * Recommended to use one GPU. This is currently the only way to "reserve" CPUs for your job, and conversion takes <10 minutes.
        * If you'll be evaluating your model using `submit_eval_jobs.py` in open-instruct, your converted model must be saved in the `oe-adapt-default` weka bucket.
        * **Custom architectures** (e.g., hybrid FLA models, Qwen, Gemma) need architecture-specific converters:
            - Standard models: `convert_checkpoint_to_hf.py`
            - Hybrid FLA models: `convert_checkpoint_to_hf_hybrid.py`
            - Check `src/examples/huggingface/` for other architecture-specific converters.

    **Using prebuilt docker images:** When converting custom architectures that require specialized dependencies (e.g., `flash-attn`, `flash-linear-attention`), use a prebuilt beaker image and the conda Python directly:

    ```bash
    gantry run --cluster ai2/saturn-cirrascale --timeout -1 -y --budget ai2/oe-adapt --workspace ai2/<your_workspace> \
        --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
        --install "/opt/conda/bin/pip install -e '.[fla,transformers]'" \
        --weka=oe-adapt-default:/weka/oe-adapt-default \
        --weka=oe-training-default:/weka/oe-training-default \
        --priority high \
        --gpus 1 \
        -- /opt/conda/bin/python src/examples/huggingface/convert_checkpoint_to_hf.py \
            -i /weka/oe-training-default/$USER/checkpoints/path-to-model/stepFINAL_STEP \
            -o /weka/oe-adapt-default/$USER/checkpoints/path-to-model/stepFINAL_STEP-hf \
            --max-sequence-length 32768 \
            --skip-validation
    ```

    Key points:
    - Use `/opt/conda/bin/pip` and `/opt/conda/bin/python` to leverage packages pre-installed in the docker image (torch, flash-attn, etc.)
    - Gantry's default venv doesn't have access to these packages
    - Only install the project + extras you need (the image already has CUDA dependencies)
    - Beaker images follow the pattern `<user>/olmo-core-tch<torch>cu<cuda>-<date>`
    - Use `-t <tokenizer_id>` to override the tokenizer saved with the model (e.g. `-t allenai/olmo-3-tokenizer-instruct-dev` for instruct models with a chat template).
    - Use `--timeout 0` instead of `--timeout -1` to launch gantry jobs without following logs, useful for converting multiple models in parallel.

2. **Verify chat template and tokenizer settings before running evals.**

    After converting to HuggingFace format, check that your model has the correct chat template for evaluation (either in `tokenizer_config.json` or as a separate `chat_template.jinja` file). The HF conversion copies whatever tokenizer was saved with the checkpoint, but that tokenizer's chat template may not be correct for evals—you may need to update it manually.

    For OLMo 3 models, see the [OLMo 3 tokenizer and chat template settings](https://allenai.github.io/open-instruct/olmo3) in open-instruct for the recommended configuration. This includes the correct `chat_template`, `eos_token`, and other tokenizer settings required for evals to work properly.

    **Example from OLMo 3:** Think models were trained with the Instruct chat template (to work around a `<think>` token masking issue during tokenization), but for evals required swapping in a chat template that includes `<think>` in `add_generation_prompt`—otherwise the model wouldn't start its response with `<think>`.

3. Launch evaluations using the submit_eval_jobs.sh script in `open-instruct` using a command such as:

    ```bash
    python scripts/submit_eval_jobs.py \
        --model_name MODEL_NAME_HERE \
        --location /weka/oe-adapt-default/$USER/checkpoints/path-to-model/stepFINAL_STEP-hf \
        --cluster ai2/saturn-cirrascale ai2/neptune-cirrascale \
        --is_tuned \
        --priority high \
        --preemptible \
        --use_hf_tokenizer_template \
        --run_oe_eval_experiments \
        --evaluate_on_weka \
        --oe_eval_max_length 32768 \
        --workspace tulu-3-results \
        --skip_oi_evals \
        --process_output r1_style
    ```