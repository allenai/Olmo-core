"""
Example script showing how you could convert model weights on HuggingFace for an OLMo2
model into a format that can be loaded by Olmo-core for fine-tuning.

Note that this script is architecture-dependent. Some models may work out-of-the-box. Support for
other models can be added by updating the constants in :mod:`olmo_core.nn.hf.convert`.

Warnings:
    - Only model weights are converted; optimizer states cannot be recovered from HF-format
      checkpoints. This means you cannot resume training from the converted checkpoint.
    - Tokenizer configuration must be specified separately.
    - Some architecture-specific features may not be fully supported.
"""

import json
import logging
import os
import re
import tempfile
from argparse import ArgumentParser
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Tuple

import rich
import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from cached_path import cached_path
from transformers import AutoConfig, AutoModelForCausalLM

from olmo_core.aliases import PathOrStr
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import save_model_and_optim_state
from olmo_core.io import copy_file, file_exists, join_path
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.conversion.state_mapping import StateType, TemplatePlaceholder
from olmo_core.nn.hf.checkpoint import load_hf_model
from olmo_core.nn.hf.convert import get_converter_from_hf
from olmo_core.nn.moe.moe import MoEType
from olmo_core.nn.rope import YaRNRoPEScalingConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.model import Transformer
from olmo_core.utils import prepare_cli_environment

log = logging.getLogger(__name__)


def _get_transformer_config(
    model_arch: str, vocab_size: int, max_sequence_length: int
) -> TransformerConfig:
    model_arch = model_arch.lower()
    transformer_configs = {
        "olmo2_190m": TransformerConfig.olmo2_190M,
        "olmo2_370m": TransformerConfig.olmo2_370M,
        "olmo2_600m": TransformerConfig.olmo2_600M,
        "olmo2_760m": TransformerConfig.olmo2_760M,
        "olmo2_1b": TransformerConfig.olmo2_1B,
        "olmo2_1b_v2": TransformerConfig.olmo2_1B_v2,
        "olmo2_3b": TransformerConfig.olmo2_3B,
        "olmo2_7b": TransformerConfig.olmo2_7B,
        "olmo2_13b": TransformerConfig.olmo2_13B,
        "olmo2_32b": TransformerConfig.olmo2_32B,
        "olmo3_190m": TransformerConfig.olmo3_190M,
        "olmo3_7b": TransformerConfig.olmo3_7B,
        "olmo3_32b": TransformerConfig.olmo3_32B,
        "smallmoe": TransformerConfig.smallmoe,
        "olmoe_1b_7b": TransformerConfig.olmoe_1B_7B,
        "ngpt_271m": TransformerConfig.ngpt_271M,
        "ngpt_1b": TransformerConfig.ngpt_1B,
        "llama2_271m": TransformerConfig.llama2_271M,
        "llama2_1b": TransformerConfig.llama2_1B,
        "llama2_7b": TransformerConfig.llama2_7B,
        "llama2_13b": TransformerConfig.llama2_13B,
        "llama2_26b": TransformerConfig.llama2_26B,
        "llama2_70b": TransformerConfig.llama2_70B,
        "llama3_1b": TransformerConfig.llama3_1B,
        "llama3_8b": TransformerConfig.llama3_8B,
        "llama3_70b": TransformerConfig.llama3_70B,
        "llama3_405b": TransformerConfig.llama3_405B,
    }

    result = transformer_configs[model_arch](vocab_size)

    if model_arch.startswith("olmo3_") and max_sequence_length > 8192:
        result = result.with_rope_scaling(
            YaRNRoPEScalingConfig(
                factor=max_sequence_length / 8192, beta_fast=32, beta_slow=1, old_context_len=8192
            )
        )
    elif model_arch.startswith("olmo2_") and max_sequence_length != 4096:
        raise RuntimeError(
            "If you get here, you have to add code that reflects how you extended RoPE when you did you long context training."
        )

    return result


def _get_tokenizer_config(tokenizer_id: str) -> TokenizerConfig:
    tokenizer_configs = {
        "dolma2": TokenizerConfig.dolma2,
        "gpt_neox_olmo_dolma_v1_5": TokenizerConfig.gpt_neox_olmo_dolma_v1_5,
        "gpt2": TokenizerConfig.gpt2,
    }

    return tokenizer_configs[tokenizer_id.lower()]()


def convert_checkpoint_from_hf(
    hf_checkpoint_path: str | Path,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    hf_revision: str = "main",
    validate: bool = True,
    debug: bool = False,
    device: torch.device | None = None,
    validation_device: torch.device | None = None,
    validation_sliding_window: int | None = None,
) -> None:
    """
    Convert a HF checkpoint to an OLMo core checkpoint.

    Args:
        hf_checkpoint_path: Path to the original HF checkpoint
        output_path: Where to save the converted model
        transformer_config_dict: Dictionary form of OLMo core model config
        tokenizer_config_dict: Dictionary form of OLMo core tokenizer config
    """
    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    rich.print(model_config)

    validation_device = validation_device or torch.device("cpu")

    assert isinstance(model_config.block, TransformerBlockConfig)
    block_entries: list[tuple[str, TransformerBlockConfig]] = [("base block", model_config.block)]
    if model_config.block_overrides:
        block_entries.extend(
            (f"block override {idx}", block_config)
            for idx, block_config in sorted(model_config.block_overrides.items())
        )

    def prepare_block_for_conversion(
        block_label: str, block_config: TransformerBlockConfig
    ) -> None:
        attention_config = block_config.sequence_mixer
        if not isinstance(attention_config, AttentionConfig):
            raise NotImplementedError(
                f"Block {block_label} has an unsupported sequence mixing config: {attention_config}"
            )
        if validate and attention_config.backend != AttentionBackendName.torch:
            backend_name = attention_config.backend.name if attention_config.backend else "None"
            log.info(
                f"Overriding attention backend from {backend_name} to torch for {block_label} conversion and validation to make validation less likely to fail."
            )
            attention_config.backend = AttentionBackendName.torch
            # Clear the legacy flag so an old config cannot conflict with the backend override.
            attention_config.use_flash = None

    for block_label, block_config in block_entries:
        prepare_block_for_conversion(block_label, block_config)

    model = model_config.build(init_device="meta")
    model.to_empty(device=device or torch.device("cpu"))

    state_dict_options = dist_cp_sd.StateDictOptions(
        flatten_optimizer_state_dict=True, cpu_offload=True
    )
    model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)

    with TemporaryDirectory() as work_dir:
        log.info(f"Loading HF checkpoint from '{hf_checkpoint_path}' (revision '{hf_revision}')")
        load_hf_model(
            hf_checkpoint_path,
            model_state_dict,
            revision=hf_revision,
            work_dir=work_dir,
            num_embeddings=model.vocab_size,
        )

        if (
            isinstance(model_config.block, TransformerBlockConfig)
            and (moe_config := model_config.block.feed_forward_moe) is not None
        ):
            if moe_config.name == MoEType.dropless:
                for k, v in model_state_dict.items():
                    # We need to reshape the w1 and w3 weights for the dropless MoE because conversion
                    # can't distinguish between dropless and regular MoE, and dropless MoE
                    # weights are shaped differently to regular MoE.
                    if k.endswith(".feed_forward_moe.experts.mlp.w1") or k.endswith(
                        ".feed_forward_moe.experts.mlp.w3"
                    ):
                        assert isinstance(v, torch.Tensor), (k, v)
                        model_state_dict[k] = (
                            v.reshape(moe_config.num_experts, -1, moe_config.hidden_size)
                            .permute(0, 2, 1)
                            .reshape(moe_config.num_experts * moe_config.hidden_size, -1)
                        )
                        log.info(f"Reshaped {k} because MoE is dropless")
            elif moe_config.name == MoEType.default:
                log.warning(
                    f"MoE is {moe_config.name}, which may drop activations and cause validation to fail. If this is not desired, please change the MoE type to 'dropless'."
                )

        model.load_state_dict(model_state_dict)

    model_and_optim_dir = join_path(output_path, "model_and_optim")
    log.info(f"Saving OLMo core checkpoint to '{model_and_optim_dir}'")
    save_model_and_optim_state(model_and_optim_dir, model, save_overwrite=True)
    log.info(f"Successfully saved converted model to '{output_path}'")

    config_path = join_path(output_path, "config.json")
    log.info(f"Writing partial experiment config to '{config_path}'")
    experiment_config_dict = {
        "model": transformer_config_dict,
        "dataset": {
            "tokenizer": tokenizer_config_dict,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        json.dump(experiment_config_dict, temp_file)
        temp_file.flush()  # make sure data is written to disk, json.dump doesn't flush.
        if hasattr(os, "fdatasync"):  # only available on linux
            os.fdatasync(temp_file)  # type: ignore
        copy_file(temp_file.name, config_path, save_overwrite=True)
        log.info(f"Successfully wrote partial experiment config to '{config_path}'")

    if validate:
        log.info("Validating converted model")
        validate_conversion(
            hf_checkpoint_path,
            model,
            tokenizer_config.vocab_size,
            hf_revision=hf_revision,
            debug=debug,
            device=validation_device,
            sliding_window=validation_sliding_window,
        )
        log.info("Validation completed successful")


def _register_debug_hooks(hf_model: torch.nn.Module, model: Transformer):
    MAX_DIM_SIZE = 1_000_000

    olmo_core_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}
    hf_debug_state: Dict[str, Tuple[int, torch.Tensor]] = {}

    def module_hook(
        debug_state: Dict[str, Tuple[int, torch.Tensor]],
        model_type: str,
        name: str,
        _: torch.nn.Module,
        args,
        output,
    ):
        if (
            model_type == "hf"
            and re.match(r"model.layers.\d+.mlp$", name)
            and isinstance(output, tuple)
        ):
            # Special casing for FlexOlmo moe
            assert isinstance(output[0], torch.Tensor), (name, output)
            output = output[0]
        if (
            model_type == "hf"
            and re.match(r"model.layers.\d+.block_sparse_moe$", name)
            and isinstance(output, tuple)
        ):
            # Special casing for HF moe
            assert isinstance(output[0], torch.Tensor), (name, output)
            output = output[0]
        if model_type == "hf" and re.match(r"model.layers.\d+.mlp.gate$", name):
            # Special casing for FlexOlmo router
            assert isinstance(output, torch.Tensor), (name, output)
            router_logits = output.detach().clone()
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            # Like topk, but we keep all the data. This will hopefully go ok.
            routing_weights, routing_indices = torch.sort(routing_weights, descending=True, dim=-1)
            output = routing_weights
            module_hook(debug_state, model_type, f"{name}.indices", _, args, routing_indices)
        if model_type == "hf" and re.match(r"model.layers.\d+.block_sparse_moe.gate$", name):
            # Special casing for HF moe router
            assert isinstance(output, torch.Tensor), (name, output)
            router_logits = output.detach().clone()
            routing_weights = F.sigmoid(router_logits.float())
            # Like topk, but we keep all the data. This will hopefully go ok.
            routing_weights, routing_indices = torch.sort(routing_weights, descending=True, dim=-1)
            output = routing_weights
            module_hook(debug_state, model_type, f"{name}.indices", _, args, routing_indices)
        if model_type == "olmo_core" and re.match(r"blocks.\d+.feed_forward_moe.router$", name):
            # Special casing for OLMo Core moe router
            assert isinstance(output, tuple), (name, output)
            assert len(output) >= 2, (name, output)
            assert isinstance(output[1], torch.Tensor), (name, output[1])
            module_hook(debug_state, model_type, f"{name}.indices", _, args, output[1])

            assert isinstance(output[0], torch.Tensor), (name, output[0])
            output = output[0]

        if len(args) >= 1 and isinstance(args[0], torch.Tensor):
            state_name = f"{name}|input"
            input = args[0].detach()
            for i, size in enumerate(input.shape):
                input = input.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), input)
        if isinstance(output, torch.Tensor):
            state_name = f"{name}|output"
            output = output.detach()
            for i, size in enumerate(output.shape):
                output = output.narrow(i, 0, min(size, MAX_DIM_SIZE))
            debug_state[state_name] = (len(debug_state), output)

    for name, module in model.named_modules():
        module.register_forward_hook(partial(module_hook, olmo_core_debug_state, "olmo_core", name))
    for name, module in hf_model.named_modules():
        module.register_forward_hook(partial(module_hook, hf_debug_state, "hf", name))

    return olmo_core_debug_state, hf_debug_state


def validate_conversion(
    hf_path: str | Path,
    model: Transformer,
    vocab_size: int,
    hf_revision: str = "main",
    debug: bool = False,
    device: torch.device | None = None,
    sliding_window: int | None = None,
):
    device = device or torch.device("cpu")
    log.info(f"Running validation on {device}")

    B, T = 1, 60
    input_ids = torch.randint(0, vocab_size, (B, T)).to(device)

    is_sliding = any(
        hasattr(block.attention, "window_size") and block.attention.window_size != (-1, -1)
        for block in model.blocks.values()
    )

    log.info("Loading HF checkpoint for validation...")
    kwargs = {}
    if is_sliding and sliding_window is not None:
        kwargs["sliding_window"] = sliding_window
    hf_config = AutoConfig.from_pretrained(
        hf_path,
        revision=hf_revision,
        **kwargs,
    )
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            hf_path,
            revision=hf_revision,
            torch_dtype="auto",
            config=hf_config,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    hf_config = hf_model.config

    olmo_core_state, hf_state = {}, {}
    if debug:
        olmo_core_state, hf_state = _register_debug_hooks(hf_model, model)

    log.info("Running OLMo core and HF models for validation...")
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits

    del hf_model

    if is_sliding and sliding_window is not None:
        for block in model.blocks.values():
            if block.attention.window_size != (-1, -1):
                block.attention.window_size = (sliding_window - 1, 0)
    dtype = getattr(hf_config, "torch_dtype", torch.float32)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        state_converter = get_converter_from_hf(model_type=getattr(hf_config, "model_type", None))
        if not hasattr(hf_config, "num_hidden_layers"):
            raise ValueError(f"Number of hidden layers missing in HF config: {hf_config}")
        n_layers: int = hf_config.num_hidden_layers
        n_experts: int | None = getattr(hf_config, "num_experts", None)

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        hf_debug_empty_state = {key.split("|")[0]: None for key in hf_state.keys()}
        state_mapping = state_converter.get_mappings(
            hf_debug_empty_state, placeholder_bounds, state_type=StateType.module
        )
        del hf_debug_empty_state

        simple_module_name_mapping = {
            mapping.source_keys[0]: mapping.dest_keys
            for mapping in state_mapping
            if len(mapping.source_keys) == 1
        }

        log.info(f"simple mapping: {simple_module_name_mapping}")
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for hf_state_name, (_, hf_tensor) in sorted(hf_state.items(), key=lambda item: item[1][0]):
            hf_key, state_type = hf_state_name.split("|")
            if hf_key in simple_module_name_mapping:
                hf_module_name = hf_key
            else:
                log.warning(
                    f"No 1-to-many param mapping found for module {hf_key}, cannot compare to HF"
                )
                continue

            olmo_core_param_names = simple_module_name_mapping[hf_module_name]
            for i_key, olmo_core_param_name in enumerate(olmo_core_param_names):
                olmo_core_state_name = f"{olmo_core_param_name}|{state_type}"
                if f"{olmo_core_param_name}|{state_type}" in olmo_core_state:
                    olmo_core_state_name = f"{olmo_core_param_name}|{state_type}"
                else:
                    log.warning(
                        f"No OLMo Core state found for param {hf_state_name}, cannot compare to HF"
                    )
                    continue

                _, olmo_core_tensor = olmo_core_state[olmo_core_state_name]
                if olmo_core_tensor.shape[0] < len(olmo_core_param_names):
                    log.warning(
                        f"Unable to chunk olmo_core state {olmo_core_state_name} into {len(olmo_core_param_names)} pieces"
                    )
                    continue
                olmo_core_tensor = olmo_core_tensor.tensor_split(len(olmo_core_param_names), dim=0)[
                    i_key
                ]

                if hf_tensor.shape != olmo_core_tensor.shape:
                    log.info(
                        f"{hf_state_name}, {olmo_core_state_name} shape mismatch: {hf_tensor.shape} {olmo_core_tensor.shape}"
                    )
                if hf_tensor.dtype != olmo_core_tensor.dtype:
                    log.info(
                        f"{hf_state_name}, {olmo_core_state_name} dtype mismatch: {hf_tensor.dtype} {olmo_core_tensor.dtype}"
                    )
                if len(hf_tensor.squeeze().shape) == len(olmo_core_tensor.squeeze().shape):
                    hf_tensor = hf_tensor.squeeze()
                    olmo_core_tensor = olmo_core_tensor.squeeze()

                    common_shape = tuple(
                        min(hf_dim, olmo_core_dim)
                        for hf_dim, olmo_core_dim in zip(hf_tensor.shape, olmo_core_tensor.shape)
                    )
                    for i, dim in enumerate(common_shape):
                        hf_tensor = hf_tensor.narrow(i, 0, dim)
                        olmo_core_tensor = olmo_core_tensor.narrow(i, 0, dim)
                    if not torch.is_floating_point(hf_tensor) or not torch.is_floating_point(
                        olmo_core_tensor
                    ):
                        diff_elements = hf_tensor != olmo_core_tensor
                        log.info(
                            f"{hf_state_name}, {olmo_core_state_name} different elements: {diff_elements.sum()} / {diff_elements.numel()}"
                        )
                    log.info(
                        f"{hf_state_name}, {olmo_core_state_name} element diff abs mean: {(hf_tensor - olmo_core_tensor).float().abs().mean()}"
                    )

    torch.testing.assert_close(
        hf_logits[..., :vocab_size].float(), logits[..., :vocab_size].float(), rtol=1e-4, atol=1e-4
    )


def load_config(config_path: PathOrStr) -> Optional[dict]:
    if not file_exists(config_path):
        log.warning(f"Config file not found at {config_path}")
        return None

    with cached_path(config_path).open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        log.warning(f"Config file at {config_path} is not an OLMo core experiment config, ignoring")
        return None

    return config_dict


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--checkpoint-input-path",
        type=str,
        required=True,
        help="Local or remote directory containing the HF checkpoint, or the model id of a HF Hub repo.",
    )
    parser.add_argument(
        "-r",
        "--revision",
        type=str,
        default="main",
        help="The revision of the HF model, if the input path is the model id of a HF Hub repo.",
    )

    parser.add_argument(
        "-c",
        "--config-path",
        type=str,
        default=None,
        help="Path to an OLMo Core experiment config containing information about the model architecture and tokenizer.",
    )
    parser.add_argument(
        "-m",
        "--model-arch",
        help="OLMo Core model architecture corresponding to the HF model. New architectures should be added to ``_get_transformer_config``. This is required when an OLMo Core experiment config is not provided.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="dolma2",
        help="OLMo Core tokenizer corresponding to the HF model. New tokenizers should be added to ``_get_tokenizer_config``. This is required when an OLMo Core experiment config is not provided.",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Local or remote directory where the converted checkpoint should be saved.",
    )
    parser.add_argument(
        "-s",
        "--max-sequence-length",
        type=int,
        help="Deprecated, do not use. Max sequence length supported by the model.",
    )
    parser.add_argument(
        "--model-id",
        help="Deprecated, model-specific mappings are now determined by the model architecture, in :mod:`olmo_core.nn.hf.convert`",
    )
    parser.add_argument(
        "--skip-validation",
        dest="validate",
        action="store_false",
        help="If set, validation to check that the converted model matches the original model is skipped.",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="If set, debug information of validation is output.",
    )
    parser.add_argument(
        "--device",
        type=torch.device,
        help="The device on which conversion and validation occurs. Defaults to CPU.",
    )
    parser.add_argument(
        "--validation-device",
        type=torch.device,
        help="The device on which validation occurs. Defaults to `device`.",
    )
    parser.add_argument(
        "--validation-sliding-window",
        help="If set, overrides the model's sliding window size during validation. Useful for checking that sliding window is correctly implemented.",
        type=int,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = load_config(args.config_path or f"{args.checkpoint_input_path}/config.json")
    transformer_config_dict = None
    if experiment_config is not None:
        transformer_config_dict = experiment_config["model"]
        tokenizer_config_dict = experiment_config.get("dataset", {}).get("tokenizer")
    else:
        assert args.model_arch is not None
        assert args.tokenizer is not None
        tokenizer_config = _get_tokenizer_config(args.tokenizer)

        # We still need to load the HF config, to get the right sequence length.
        with cached_path(args.config_path or f"{args.checkpoint_input_path}/config.json").open(
            "r", encoding="utf-8"
        ) as f:
            hf_config_dict = json.load(f)

        transformer_config = _get_transformer_config(
            args.model_arch,
            tokenizer_config.padded_vocab_size(),
            hf_config_dict["max_position_embeddings"],
        )
        transformer_config_dict = transformer_config.as_config_dict()
        tokenizer_config_dict = tokenizer_config.as_config_dict()

    assert transformer_config_dict is not None
    assert tokenizer_config_dict is not None

    convert_checkpoint_from_hf(
        hf_checkpoint_path=args.checkpoint_input_path,
        hf_revision=args.revision,
        output_path=args.output_dir,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        validate=args.validate,
        debug=args.debug,
        device=args.device,
        validation_device=args.validation_device or args.device,
        validation_sliding_window=args.validation_sliding_window,
    )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
