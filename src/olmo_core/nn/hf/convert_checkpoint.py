"""
Utilities for converting OLMo Core checkpoints to HuggingFace format.

This module exposes :func:`convert_checkpoint_to_hf`, :func:`validate_conversion`, and
:func:`load_config` for both programmatic use (e.g. from a training callback) and from
the example CLI script at ``src/examples/huggingface/convert_checkpoint_to_hf.py``.

Supports both standard architectures (olmo2, olmo3) and hybrid (GDN + attention)
architectures. Hybrid models are saved as raw ``config.json`` + ``model.safetensors``
rather than using ``save_pretrained()``.
"""

import json
import logging
import re
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
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.io import file_exists, join_path
from olmo_core.nn.attention import AttentionBackendName, AttentionConfig
from olmo_core.nn.conversion.state_mapping import StateType, TemplatePlaceholder
from olmo_core.nn.moe.moe import MoEType
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerConfig,
    normalize_legacy_config_dict,
)
from olmo_core.nn.transformer.model import Transformer

from .checkpoint import save_hf_hybrid_model, save_hf_model
from .config import is_olmo_hybrid_model
from .convert import get_converter_to_hf
from .tokenizer import export_checkpoint_tokenizer

log = logging.getLogger(__name__)


def convert_checkpoint_to_hf(
    original_checkpoint_path: str | Path | None,
    output_path: str | Path,
    transformer_config_dict: Dict[str, Any],
    tokenizer_config_dict: Dict[str, Any],
    *,
    model_state_dict: Optional[Dict[str, Any]] = None,
    dtype: Optional[DType] = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
    max_sequence_length: int | None = None,
    validate: bool = True,
    debug: bool = False,
    device: torch.device | None = None,
    moe_capacity_factor: float | None = None,
    validation_device: torch.device | None = None,
    validation_sliding_window: int | None = None,
) -> None:
    """
    Convert an OLMo Core checkpoint to HuggingFace format.

    :param original_checkpoint_path: Path to the original checkpoint. Can be ``None`` if
        ``model_state_dict`` is provided.
    :param output_path: Where to save the converted model.
    :param transformer_config_dict: Dictionary form of OLMo Core model config.
    :param tokenizer_config_dict: Dictionary form of OLMo Core tokenizer config.
    :param model_state_dict: Optional pre-gathered model state dict. If provided, weights are
        taken from this instead of loading from ``original_checkpoint_path``.
    """
    if model_state_dict is None and original_checkpoint_path is None:
        raise ValueError("Either model_state_dict or original_checkpoint_path must be provided")
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError(f"Invalid sequence length: {max_sequence_length}")

    # Remove deprecated transformer config options
    if "compile" in transformer_config_dict:
        del transformer_config_dict["compile"]
    if "dp_config" in transformer_config_dict:
        del transformer_config_dict["dp_config"]
    if "tp_config" in transformer_config_dict:
        del transformer_config_dict["tp_config"]
    if "float8_config" in transformer_config_dict:
        del transformer_config_dict["float8_config"]

    normalize_legacy_config_dict(transformer_config_dict)
    model_config = TransformerConfig.from_dict(transformer_config_dict)
    rich.print(model_config)

    validation_device = validation_device or torch.device("cpu")

    if isinstance(model_config.block, dict):
        block_entries: list[tuple[str, TransformerBlockConfig]] = [
            (f"block type '{name}'", block_config)
            for name, block_config in model_config.block.items()
        ]
    else:
        assert isinstance(model_config.block, TransformerBlockConfig)
        block_entries = [("base block", model_config.block)]
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
            log.info(
                f"Block {block_label} uses non-attention sequence mixer ({type(attention_config).__name__}), "
                f"skipping attention backend override"
            )
            return
        if validate and attention_config.backend != AttentionBackendName.torch:
            backend_name = attention_config.backend.name if attention_config.backend else "None"
            log.info(
                f"Overriding attention backend from {backend_name} to torch for {block_label} conversion and validation to make validation less likely to fail."
            )
            attention_config.backend = AttentionBackendName.torch
            # Clear the legacy flag so an old config cannot conflict with the backend override.
            attention_config.use_flash = None

        if moe_capacity_factor is not None and block_config.feed_forward_moe is not None:
            block_config.feed_forward_moe.capacity_factor = moe_capacity_factor

    for block_label, block_config in block_entries:
        prepare_block_for_conversion(block_label, block_config)

    tokenizer_config = TokenizerConfig.from_dict(tokenizer_config_dict)
    vocab_size = tokenizer_config.vocab_size

    huggingface_tokenizer = export_checkpoint_tokenizer(
        original_checkpoint_path,
        output_path,
        tokenizer_config,
        override=tokenizer_id,
        revision=tokenizer_revision,
        max_sequence_length=max_sequence_length,
    )
    max_sequence_length = max_sequence_length or huggingface_tokenizer.model_max_length

    # Tokenizer errors must fail before expensive checkpoint reads/model allocation.
    model = model_config.build(init_device="meta")
    model.to_empty(device=device or torch.device("cpu"))

    with TemporaryDirectory() as work_dir:
        if model_state_dict is None:
            assert original_checkpoint_path is not None
            model_and_optim_dir = join_path(original_checkpoint_path, "model_and_optim")
            log.info(f"Loading checkpoint from '{model_and_optim_dir}'")
            load_model_and_optim_state(
                model_and_optim_dir,
                model,
                work_dir=work_dir,
            )
            log.info(f"Saving checkpoint to '{output_path}'")
            state_dict_options = dist_cp_sd.StateDictOptions(
                flatten_optimizer_state_dict=True, cpu_offload=True
            )
            model_state_dict = dist_cp_sd.get_model_state_dict(model, options=state_dict_options)
        else:
            log.info(f"Using provided model state dict, saving to '{output_path}'")
            # Load the provided state dict into the model so that validation works.
            model.load_state_dict(model_state_dict)

        hybrid = is_olmo_hybrid_model(model)

        if hybrid:
            log.info("Detected hybrid model (GDN + attention layers)")
            save_hf_hybrid_model(
                output_path,
                model_state_dict,
                model,
                dtype=dtype,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length or 65536,
            )
        else:
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
                                v.reshape(moe_config.num_experts, moe_config.hidden_size, -1)
                                .permute(0, 2, 1)
                                .reshape(-1, moe_config.hidden_size)
                            )
                            log.info(f"Reshaped {k} because MoE is dropless")
                elif moe_config.name == MoEType.default:
                    log.warning(
                        f"MoE is {moe_config.name}, which may drop activations and cause validation to fail. You can try mitigating this by setting '--moe-capacity-factor' to a higher value."
                    )

            save_hf_model(
                output_path,
                model_state_dict,
                model,
                huggingface_tokenizer=huggingface_tokenizer,
                dtype=dtype,
                vocab_size=vocab_size,
                work_dir=work_dir,
                save_overwrite=True,
            )

        log.info(f"Successfully saved converted model to '{output_path}'")

    # Fix up config.json with tokenizer info and max_position_embeddings.
    log.info(
        "Fixing HF config using updated config from tokenizer config data and script arguments"
    )
    huggingface_config = AutoConfig.from_pretrained(output_path)
    huggingface_config.max_position_embeddings = max_sequence_length
    # Resolve all special IDs from the exported tokenizer, including source
    # metadata preserved when the training config leaves an ID unspecified.
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        setattr(huggingface_config, name, getattr(huggingface_tokenizer, name))
    huggingface_config.save_pretrained(output_path)
    log.info(
        "Successfully fixed config using updated config from tokenizer config data and script arguments"
    )

    if validate:
        log.info("Validating converted model")
        validate_conversion(
            output_path,
            model,
            tokenizer_config.vocab_size,
            debug=debug,
            dtype=dtype,
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
    debug: bool = False,
    dtype: DType | None = None,
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

    log.info("Loading converted checkpoint for validation...")
    kwargs = {}
    if is_sliding and sliding_window is not None:
        kwargs["sliding_window"] = sliding_window
    hf_config = AutoConfig.from_pretrained(
        hf_path,
        **kwargs,
    )
    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            hf_path,
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

    # apply sliding_window size override if provided
    if is_sliding and sliding_window is not None:
        for block in model.blocks.values():
            if block.attention.window_size != (-1, -1):
                block.attention.window_size = (sliding_window - 1, 0)
    if dtype:
        model = model.to(dtype.as_pt())
    model = model.to(device=device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids)

    if debug:
        state_converter = get_converter_to_hf(getattr(hf_config, "model_type", None))
        if not hasattr(hf_config, "num_hidden_layers"):
            raise ValueError(f"Number of hidden layers missing in HF config: {hf_config}")
        n_layers: int = hf_config.num_hidden_layers
        n_experts: int | None = getattr(hf_config, "num_experts", None)

        placeholder_bounds = {
            TemplatePlaceholder.LAYER: n_layers,
        }
        if n_experts:
            placeholder_bounds[TemplatePlaceholder.EXPERT] = n_experts

        olmo_debug_empty_state = {key.split("|")[0]: None for key in olmo_core_state.keys()}
        state_mapping = state_converter.get_mappings(
            olmo_debug_empty_state, placeholder_bounds, state_type=StateType.module
        )
        del olmo_debug_empty_state

        simple_module_name_mapping = {
            mapping.source_keys[0]: mapping.dest_keys
            for mapping in state_mapping
            if len(mapping.source_keys) == 1
        }

        log.info(f"simple mapping: {simple_module_name_mapping}")
        log.info(f"hf_state keys: {hf_state.keys()}")
        log.info(f"olmo_core_state keys: {olmo_core_state.keys()}")

        for olmo_core_state_name, (_, olmo_core_tensor) in sorted(
            olmo_core_state.items(), key=lambda item: item[1][0]
        ):
            olmo_core_key, state_type = olmo_core_state_name.split("|")
            if olmo_core_key in simple_module_name_mapping:
                olmo_core_module_name = olmo_core_key
            else:
                log.warning(
                    f"No 1-to-many param mapping found for module {olmo_core_key}, cannot compare to HF"
                )
                continue

            hf_param_names = simple_module_name_mapping[olmo_core_module_name]
            for i_key, hf_param_name in enumerate(hf_param_names):
                hf_state_name = f"{hf_param_name}|{state_type}"
                if f"{hf_param_name}|{state_type}" in hf_state:
                    hf_state_name = f"{hf_param_name}|{state_type}"
                else:
                    log.warning(
                        f"No HF state found for param {hf_state_name}, cannot compare to OLMo Core"
                    )
                    continue

                _, hf_tensor = hf_state[hf_state_name]
                if hf_tensor.shape[0] < len(hf_param_names):
                    log.warning(
                        f"Unable to chunk HF state {hf_state_name} into {len(hf_param_names)} pieces"
                    )
                    continue
                hf_tensor = hf_tensor.tensor_split(len(hf_param_names), dim=0)[i_key]

                if olmo_core_tensor.shape != hf_tensor.shape:
                    log.info(
                        f"{olmo_core_state_name}, {hf_state_name} shape mismatch: {olmo_core_tensor.shape} {hf_tensor.shape}"
                    )
                if olmo_core_tensor.dtype != hf_tensor.dtype:
                    log.info(
                        f"{olmo_core_state_name}, {hf_state_name} dtype mismatch: {olmo_core_tensor.dtype} {hf_tensor.dtype}"
                    )
                if len(olmo_core_tensor.squeeze().shape) == len(hf_tensor.squeeze().shape):
                    olmo_core_tensor = olmo_core_tensor.squeeze()
                    hf_tensor = hf_tensor.squeeze()

                    common_shape = tuple(
                        min(olmo_core_dim, hf_dim)
                        for olmo_core_dim, hf_dim in zip(olmo_core_tensor.shape, hf_tensor.shape)
                    )
                    for i, dim in enumerate(common_shape):
                        olmo_core_tensor = olmo_core_tensor.narrow(i, 0, dim)
                        hf_tensor = hf_tensor.narrow(i, 0, dim)
                    if not torch.is_floating_point(olmo_core_tensor) or not torch.is_floating_point(
                        hf_tensor
                    ):
                        diff_elements = olmo_core_tensor != hf_tensor
                        log.info(
                            f"{olmo_core_state_name}, {hf_state_name} different elements: {diff_elements.sum()} / {diff_elements.numel()}"
                        )
                    log.info(
                        f"{olmo_core_state_name}, {hf_state_name} element diff abs mean: {(olmo_core_tensor - hf_tensor).float().abs().mean()}"
                    )

    torch.testing.assert_close(
        hf_logits[..., :vocab_size].float(), logits[..., :vocab_size].float(), rtol=1e-4, atol=1e-4
    )


def load_config(checkpoint_input_dir: PathOrStr) -> Optional[dict]:
    """
    Load the experiment config from an OLMo Core checkpoint directory.
    """
    if not file_exists(f"{checkpoint_input_dir}/config.json"):
        raise RuntimeError(f"Config file not found at {checkpoint_input_dir}")

    with cached_path(f"{checkpoint_input_dir}/config.json").open("r", encoding="utf-8") as f:
        config_dict = json.load(f)

    if "model" not in config_dict:
        raise RuntimeError(
            f"Config file at {checkpoint_input_dir} is not an OLMo core experiment config, ignoring"
        )

    return config_dict
