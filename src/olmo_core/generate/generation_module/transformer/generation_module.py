import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint.state_dict as dist_cp_sd
from cached_path import cached_path
from torch.distributed import DeviceMesh, ProcessGroup
from torch.distributed.checkpoint.metadata import Metadata
from tqdm import tqdm

from olmo_core import env_flags
from olmo_core.aliases import PathOrStr
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data.utils import attention_mask_to_cache_leftpad
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_state_dict
from olmo_core.distributed.parallel import build_world_mesh, get_dp_process_group
from olmo_core.distributed.utils import (
    broadcast_object,
    get_fs_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.float8 import Float8Config
from olmo_core.generate.generation_module import GenerationConfig, GenerationModule
from olmo_core.generate.sampling import select_next_token
from olmo_core.generate.utils import selective_log_softmax
from olmo_core.io import is_url, join_path, normalize_path
from olmo_core.nn.attention import (
    Attention,
    AttentionBackendName,
    CompressiveGQAGroupedAttention,
    DocumentCompressiveLandmarkAttention,
    DocumentLandmarkAttention,
    FastCompressiveLandmarkAttention,
    FastLandmarkAttention,
    SparseLandmarkAttention,
    landmark_gate_analysis,
)
from olmo_core.nn.transformer import Transformer, TransformerConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import (
    TransformerDataParallelConfig,
)
from olmo_core.utils import gc_cuda, get_default_device, log_or_print, move_to_device

log = logging.getLogger(__name__)

# Landmark attention variants that need prompt-side landmark insertion + "one long local block"
# decoding during generation.
_LANDMARK_ATTENTION_TYPES = (
    FastLandmarkAttention,
    SparseLandmarkAttention,
    DocumentLandmarkAttention,
)


def _insert_landmark_tokens(
    input_ids: torch.Tensor, mem_freq: int, mem_id: int, num_landmarks: int = 1
) -> torch.Tensor:
    """
    Insert ``num_landmarks`` landmark tokens after every ``mem_freq`` content tokens of each row,
    reproducing the block structure that landmark models are trained on: with
    ``block_size = mem_freq + num_landmarks``, the last ``num_landmarks`` positions of each block are
    landmarks. A trailing partial block (fewer than ``mem_freq`` tokens) is left without landmarks.

    This must agree with how the model derives ``is_mem`` (see
    :meth:`~olmo_core.nn.attention.landmark_multi_compressive.MultiCompressiveLandmarkAttention._attn_core`),
    otherwise landmarks land in columns the kernel does not treat as landmarks.

    :param input_ids: Content token IDs of shape ``(batch_size, seq_len)`` (no landmarks).
    :param mem_freq: Number of content tokens between landmark runs (landmark block size is
        ``mem_freq + num_landmarks``).
    :param mem_id: The landmark token ID to insert.
    :param num_landmarks: Landmark tokens per block. The default of 1 is the single-landmark
        geometry.

    :returns: Token IDs of shape
        ``(batch_size, seq_len + num_landmarks * (seq_len // mem_freq))`` with landmarks inserted at
        the fixed periodic positions.
    """
    B, C = input_ids.shape
    n_blocks = C // mem_freq
    if n_blocks == 0:
        return input_ids
    block_size = mem_freq + num_landmarks
    out_len = C + n_blocks * num_landmarks
    out = input_ids.new_full((B, out_len), mem_id)
    pos = torch.arange(out_len, device=input_ids.device)
    # Content occupies the first ``mem_freq`` slots of each block; the tail is landmarks.
    content_pos = pos[(pos % block_size) < mem_freq]
    out[:, content_pos] = input_ids
    return out


def _build_landmark_prompt(
    input_ids: torch.Tensor,
    mem_freq: int,
    mem_id: int,
    *,
    mode: str,
    pad_id: int,
    num_landmarks: int = 1,
) -> torch.Tensor:
    """
    Build the landmark-structured prompt fed to prefill from a content-only prompt.

    In ``"generation_only"`` mode the final partial block is padded with ``pad_id`` up to the next
    landmark position, so the prompt always ends with a landmark run (keeping landmarks at the
    trained periodic positions). In ``"extend_last_block"`` mode a trailing partial block is left
    as-is (it stays part of the growing local block during decode).
    """
    content = input_ids
    if mode == "generation_only":
        pad_len = (-content.shape[1]) % mem_freq
        if pad_len:
            pad_block = content.new_full((content.shape[0], pad_len), pad_id)
            content = torch.cat([content, pad_block], dim=1)
    return _insert_landmark_tokens(content, mem_freq, mem_id, num_landmarks)


class TransformerGenerationModule(GenerationModule):
    """Module for autoregressive text generation with transformer models."""

    def __init__(
        self,
        model: Transformer,
        generation_config: GenerationConfig,
        compile_model: bool = False,
        float8_config: Optional[Float8Config] = None,
        dp_config: Optional[TransformerDataParallelConfig] = None,
        device: Optional[torch.device] = None,
        state_dict_load_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        state_dict_save_opts: Optional[dist_cp_sd.StateDictOptions] = None,
        load_key_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()

        self.device = device or get_default_device()

        self.world_mesh: Optional[DeviceMesh] = None
        if is_distributed():
            self.world_mesh = build_world_mesh(dp=dp_config, device_type=self.device.type)
        elif dp_config is not None:
            raise OLMoConfigurationError(
                "Training parallelism configs are only valid for distributed training"
            )

        # Parallelize model.
        self.model = parallelize_model(
            model,
            world_mesh=self.world_mesh,
            device=self.device,
            compile_model=compile_model,
            float8_config=float8_config,
            dp_config=dp_config,
        )
        self._model_mode: Optional[Literal["train", "eval"]] = None

        self._dp_config = dp_config
        self.state_dict_save_opts = state_dict_save_opts or dist_cp_sd.StateDictOptions(strict=True)
        self.state_dict_load_opts = state_dict_load_opts or dist_cp_sd.StateDictOptions(strict=True)
        self.load_key_mapping = load_key_mapping
        self._generation_config = generation_config
        # Whether the sparse landmark decode patch has been installed on this model (see
        # :meth:`_maybe_enable_sparse_decode`). Installed lazily on the first landmark generate.
        self._sparse_decode_layers: Optional[int] = None

    @property
    def dp_process_group(self) -> Optional[dist.ProcessGroup]:
        return None if self.world_mesh is None else get_dp_process_group(self.world_mesh)

    @property
    def world_size(self) -> int:
        return get_world_size()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": dist_cp_sd.get_model_state_dict(self.model, options=self.state_dict_save_opts),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        dist_cp_sd.set_model_state_dict(
            self.model, state_dict["model"], options=self.state_dict_load_opts
        )

    def prepare_inference_cache(self, batch_size: int, max_seq_len: int):
        # Note: not all models use key-value caches, which is why this method
        # is called "prepare_inference_cache" rather than "prepare_kv_cache".
        # Attention blocks get a KV cache; recurrent mixers (e.g. GatedDeltaNet) get a
        # conv+recurrent state cache via init_state_cache.
        for block in self.model.blocks.values():
            mixer = block.attention
            if isinstance(mixer, Attention):
                if mixer.kv_cache_manager is None:
                    mixer.init_kv_cache_manager(batch_size, max_seq_len)
                else:
                    mixer.kv_cache_manager.reset(batch_size, max_seq_len)
            elif hasattr(mixer, "init_state_cache"):
                mixer.init_state_cache(batch_size, max_seq_len)  # type: ignore[attr-defined]

    def free_inference_cache(self):
        for block in self.model.blocks.values():
            mixer = block.attention
            if isinstance(mixer, Attention):
                mixer.kv_cache_manager = None
            elif hasattr(mixer, "state_cache"):
                mixer.state_cache = None  # type: ignore[attr-defined]

    def _non_kv_cache_mixer_types(self) -> Set[str]:
        """
        Return the type names of any sequence mixers in the model that support neither a KV cache
        (:class:`~olmo_core.nn.attention.Attention`) nor a recurrent state cache (mixers exposing
        ``init_state_cache``, e.g. :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet`).

        Such a mixer keeps no decode state, so step-by-step decoding with ``use_cache=True`` would
        feed it only the latest token while silently dropping all prior context. Used to guard
        against that wrong code path. Empty for attention and recurrent-with-cache models.
        """
        return {
            type(block.attention).__name__
            for block in self.model.blocks.values()
            if not isinstance(block.attention, Attention)
            and not hasattr(block.attention, "init_state_cache")
        }

    def _landmark_attention_layers(self) -> List[Attention]:
        """Return the model's landmark attention layers (empty if this is not a landmark model)."""
        layers: List[Attention] = []
        for block in self.model.blocks.values():
            attn = block.attention
            if isinstance(attn, _LANDMARK_ATTENTION_TYPES):
                layers.append(cast(Attention, attn))
        return layers

    def _set_landmark_eval_decode(
        self,
        prompt_len: int,
        mode: str,
        top_k: Optional[int] = None,
        nonselected_landmark_mass: Optional[float] = None,
        group_landmark_selection: Optional[str] = None,
        decode_gate_mode: Optional[str] = None,
    ):
        for attn in self._landmark_attention_layers():
            # ``CompressiveGQAGroupedAttention`` additionally takes ``decode_gate_mode`` (grouped vs
            # selection-only decode gate); check it first (it subclasses FastCompressiveLandmark).
            if isinstance(attn, CompressiveGQAGroupedAttention):
                attn.set_landmark_eval_decode(
                    prompt_len,
                    mode,
                    top_k=top_k,
                    nonselected_landmark_mass=nonselected_landmark_mass,
                    group_landmark_selection=group_landmark_selection,
                    decode_gate_mode=decode_gate_mode,
                )
            # ``DocumentCompressiveLandmarkAttention`` does not subclass
            # ``FastCompressiveLandmarkAttention`` (it subclasses ``DocumentLandmarkAttention`` and
            # borrows the compressive decode methods via explicit class-attribute assignment, see
            # landmark_document_compressive.py), so both need checking here -- an isinstance check
            # against only ``FastCompressiveLandmarkAttention`` would silently skip
            # ``nonselected_landmark_mass``/``group_landmark_selection`` for document-chunked models.
            elif isinstance(
                attn, (FastCompressiveLandmarkAttention, DocumentCompressiveLandmarkAttention)
            ):
                attn.set_landmark_eval_decode(
                    prompt_len,
                    mode,
                    top_k=top_k,
                    nonselected_landmark_mass=nonselected_landmark_mass,
                    group_landmark_selection=group_landmark_selection,
                )
            else:
                attn.set_landmark_eval_decode(prompt_len, mode, top_k=top_k)  # type: ignore[attr-defined]

    def _clear_landmark_eval_decode(self):
        for attn in self._landmark_attention_layers():
            attn.clear_landmark_eval_decode()  # type: ignore[attr-defined]

    def _landmark_block_size(self, layers: List[Attention]) -> int:
        """Common landmark block size across ``layers``.

        Every layer must agree on ``num_landmarks`` (multi-landmark variants put that many landmarks
        at the end of each block; the single-landmark classes have no such attribute and default to
        1) and on ``block_size == mem_freq + num_landmarks``: the prompt-side landmark insertion
        (:func:`_insert_landmark_tokens`) is told that same count, and a mismatch would feed the
        mixer a prompt whose landmarks sit at the wrong positions -- silently, and looking like a
        modeling result. Reject it instead.
        """
        n_lm = {int(getattr(a, "num_landmarks", 1)) for a in layers}
        if len(n_lm) != 1:
            raise OLMoConfigurationError(
                f"Landmark layers have inconsistent num_landmarks values: {sorted(n_lm)}"
            )
        num_landmarks = n_lm.pop()
        sizes = {
            int(getattr(a, "block_size", int(getattr(a, "mem_freq")) + num_landmarks))
            for a in layers
        }
        if len(sizes) != 1:
            raise OLMoConfigurationError(
                f"Landmark layers have inconsistent block sizes: {sorted(sizes)}"
            )
        block_size = sizes.pop()
        mem_freqs = {int(getattr(a, "mem_freq")) for a in layers}
        if mem_freqs != {block_size - num_landmarks}:
            raise OLMoConfigurationError(
                f"Landmark block size {block_size} != mem_freq {sorted(mem_freqs)} + "
                f"num_landmarks {num_landmarks}; the prompt-side landmark insertion would misplace "
                "the landmarks."
            )
        return block_size

    def _sparse_decode_requested(self) -> bool:
        """:data:`GenerationConfig.landmark_sparse_decode`, with the ``OLMO_LANDMARK_SPARSE_DECODE``
        environment variable as an override (``0``/``false``/``no`` disables, anything else enables).
        """
        env = env_flags.get_env(env_flags.LANDMARK_SPARSE_DECODE)
        if env is not None:
            return env.strip().lower() not in ("0", "false", "no", "off", "")
        return self._generation_config.landmark_sparse_decode

    def _maybe_freeze_fla_autotune(self):
        """Once per module: stop FLA kernels re-autotuning at every new prompt length.

        See :mod:`olmo_core.nn.attention.fla_autotune` -- on Gated DeltaNet models this was ~95%
        of eval wall-clock. ``OLMO_FLA_LENGTH_AUTOTUNE=1`` keeps FLA's per-length autotuning.
        """
        if getattr(self, "_fla_autotune_frozen", False):
            return
        self._fla_autotune_frozen = True
        if env_flags.get_env(env_flags.FLA_LENGTH_AUTOTUNE) == "1":
            return
        from olmo_core.nn.attention.fla_autotune import freeze_fla_length_autotune

        n = freeze_fla_length_autotune()
        if n:
            log.info(f"Froze length-keyed autotuning on {n} FLA kernel(s) for generation.")

    def _maybe_enable_sparse_decode(self):
        """Install the genuinely sparse landmark decode (idempotent, inference-only).

        The shipped landmark decode expands the KV cache with ``repeat_kv`` and scores every cached
        key before masking almost all of them away; the patched one scores only the keys that
        survive the mask. Same outputs, far less work -- so it is on by default for the attention
        families that support it, and a no-op for every other model.
        """
        if self._sparse_decode_layers is not None or not self._sparse_decode_requested():
            return
        from olmo_core.nn.attention.landmark_sparse_decode import (
            enable_sparse_decode,
            reset_sparse_decode_cache,
        )

        try:
            n = enable_sparse_decode(self.model, strict=False)
        except Exception as e:  # never let an optimization break generation
            log.warning(f"Sparse landmark decode not installed ({e}); using the shipped decode.")
            self._sparse_decode_layers = 0
            return
        self._sparse_decode_layers = n
        if n:
            reset_sparse_decode_cache(self.model)
            log_or_print(log, f"Sparse landmark decode enabled on {n} attention layers.")

    def _set_model_mode(self, mode: Literal["train", "eval"]):
        if self._model_mode != mode:
            if mode == "train":
                self.model.train()
            elif mode == "eval":
                self.model.eval()
            else:
                raise ValueError(f"Invalid model mode: {mode}")
            self._model_mode = mode

    @torch.inference_mode()
    def model_forward(self, input_ids: torch.Tensor, **kwargs):
        self._set_model_mode("eval")
        input_ids = move_to_device(input_ids, self.device)
        return self.model(input_ids, **kwargs)

    @torch.inference_mode()
    def generate_batch(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        return_logits: bool = False,
        return_logprobs: bool = False,
        completions_only: bool = False,
        log_timing: bool = True,
        stop_string_tokenizer: Optional[Any] = None,
        **generation_kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Generate text with autoregressive decoding.

        :param input_ids: Input token IDs of shape ``(batch_size, seq_len)``.
        :param attention_mask: Optional attention mask of shape ``(batch_size, seq_len)``. This should be
            a *left-padding* mask, not an arbitrary attention mask. If not provided, the model
            will assume there are no left-padding tokens.
        :param return_logits: If ``True``, return logits along with generated tokens.
        :param return_logprobs: If ``True``, return log probabilities *for the generated tokens* along with
            generated tokens. This is notably more memory efficient than ``return_logits``.
        :param completions_only: If ``True``, return only the completions, not the entire sequence.
        :param generation_kwargs: Generation configuration overrides.

        :returns: Tuple of ``(generated_ids, logits, logprobs)`` where:
            - ``generated_ids``: Generated token IDs of shape ``(batch_size, output_length)``.
            - ``logits``: Full logits if ``return_logits=True``, else ``None``. Shape: ``(batch_size, output_length, vocab_size)``.
            - ``logprobs``: Log probabilities of generated tokens if ``return_logprobs=True``, else ``None``. Shape: ``(batch_size, output_length)``.
        """
        start_time = time.perf_counter()
        self._maybe_freeze_fla_autotune()

        # Replace generation config with any overrides.
        generation_config = self._generation_config.replace(**generation_kwargs)
        eos = generation_config.eos_token_id
        pad = generation_config.pad_token_id

        # Guard the no-decode-state wrong path: a sequence mixer that supports neither a KV cache
        # nor a recurrent state cache carries no per-step state, so cached decoding would feed it
        # only the latest token and silently drop all prompt context. Attention (KV cache) and
        # GatedDeltaNet (conv+recurrent state cache) are both fine; this only fires for a mixer that
        # supports neither. Fail loudly rather than decode garbage.
        if generation_config.use_cache:
            non_kv_cache_mixers = self._non_kv_cache_mixer_types()
            if non_kv_cache_mixers:
                raise OLMoConfigurationError(
                    "use_cache=True is not supported for models with sequence mixers "
                    f"{sorted(non_kv_cache_mixers)} that have no decode-time cache (neither a KV "
                    "cache nor a recurrent state cache). Set use_cache=False to generate correctly "
                    "(slower), or add state caching for these layers."
                )

        input_ids = move_to_device(input_ids, self.device)

        # Landmark attention: insert landmark tokens into the *prompt* every ``mem_freq`` content
        # tokens (so prefill sees the trained block structure) and decode plain content tokens as
        # "one long local block". This keeps the eval harness landmark-agnostic -- it only ever sees
        # content tokens in and content tokens out.
        landmark_layers = self._landmark_attention_layers()
        landmark_active = len(landmark_layers) > 0
        orig_input_ids = input_ids
        if landmark_active:
            if generation_config.landmark_mem_id is None:
                raise OLMoConfigurationError(
                    "This is a landmark-attention model; set GenerationConfig.landmark_mem_id "
                    "(the landmark token ID) to generate."
                )
            if not generation_config.use_cache:
                raise OLMoConfigurationError(
                    "Landmark-attention generation requires use_cache=True."
                )
            if attention_mask is not None and not bool(attention_mask.all()):
                raise OLMoConfigurationError(
                    "Landmark-attention generation does not support left-padding / attention masks "
                    "(blocks are tied to absolute position; use batch_size=1)."
                )
            mem_freqs = {int(getattr(a, "mem_freq")) for a in landmark_layers}
            if len(mem_freqs) != 1:
                raise OLMoConfigurationError(
                    f"Landmark layers have inconsistent mem_freq values: {sorted(mem_freqs)}"
                )
            # Multi-landmark variants put ``num_landmarks`` landmarks at the end of each block; the
            # single-landmark classes have no such attribute and default to 1.
            nums_landmarks = {int(getattr(a, "num_landmarks", 1)) for a in landmark_layers}
            if len(nums_landmarks) != 1:
                raise OLMoConfigurationError(
                    f"Landmark layers have inconsistent num_landmarks values: "
                    f"{sorted(nums_landmarks)}"
                )
            self._landmark_block_size(landmark_layers)  # block_size == mem_freq + num_landmarks
            pad_id = generation_config.landmark_pad_id
            if pad_id is None:
                pad_id = generation_config.pad_token_id
            mem_freq = mem_freqs.pop()
            num_landmarks = nums_landmarks.pop()
            input_ids = _build_landmark_prompt(
                input_ids,
                mem_freq,
                generation_config.landmark_mem_id,
                mode=generation_config.landmark_decode_mode,
                pad_id=pad_id,
                num_landmarks=num_landmarks,
            )

        batch_size, prompt_len = input_ids.shape
        if landmark_active:
            # Default to hard top-k retrieval at ~landmark_top_k_fraction of the prompt's blocks
            # (top 10% by default); an explicit landmark_top_k_blocks overrides the fraction. None
            # for both falls back to dense soft-gating over all past blocks.
            top_k = generation_config.landmark_top_k_blocks
            if top_k is None and generation_config.landmark_top_k_fraction is not None:
                num_blocks = max(1, prompt_len // (mem_freq + num_landmarks))
                top_k = max(1, math.ceil(generation_config.landmark_top_k_fraction * num_blocks))
            self._set_landmark_eval_decode(
                prompt_len,
                generation_config.landmark_decode_mode,
                top_k=top_k,
                nonselected_landmark_mass=generation_config.landmark_nonselected_mass,
                group_landmark_selection=generation_config.landmark_group_selection,
                decode_gate_mode=generation_config.landmark_decode_gate_mode,
            )
            # Inference-only flat-softmax ablation (keep top-k selection, drop gate reweighting).
            # Enabled via GenerationConfig or the OLMO_LANDMARK_FLAT_SOFTMAX env var (the latter lets
            # eval harnesses toggle the variant with no model-arg change). See
            # analysis/flat_softmax_variant_eval.md.
            flat_softmax = generation_config.landmark_flat_softmax or bool(
                env_flags.get_env(env_flags.LANDMARK_FLAT_SOFTMAX)
            )
            if flat_softmax:
                log.info("Landmark flat-softmax decode variant ACTIVE (gate reweighting removed)")
            for attn in self._landmark_attention_layers():
                attn._eval_flat_softmax = flat_softmax  # type: ignore[attr-defined]
            # Optional landmark-gate analysis hook (off unless OLMO_LANDMARK_GATE_LOG is set). Tag
            # each landmark layer with its block index so the recorder can key gates by layer, then
            # open a fresh per-example record (context_len defaults to the content prompt length).
            if landmark_gate_analysis.is_enabled():
                for block_idx, block in self.model.blocks.items():
                    attn = block.attention
                    if isinstance(attn, _LANDMARK_ATTENTION_TYPES):
                        attn._gate_log_layer_idx = int(block_idx)  # type: ignore[attr-defined]
                landmark_gate_analysis.start_example(orig_input_ids.shape[1])
            self._maybe_enable_sparse_decode()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        stop_tokens = (
            torch.tensor(generation_config.stop_token_ids, device=self.device, dtype=torch.int32)
            if generation_config.stop_token_ids is not None
            else None
        )

        # Output containers
        generated = input_ids
        all_logits: Optional[List[torch.Tensor]] = [] if return_logits else None
        all_logprobs: Optional[List[torch.Tensor]] = [] if return_logprobs else None

        # Timing stats
        time_to_first_token = None
        decode_start_time = None
        setup_time = None
        tokens_generated = 0

        if generation_config.max_new_tokens is not None:
            max_length = prompt_len + generation_config.max_new_tokens
        elif generation_config.max_length is not None:
            max_length = generation_config.max_length
            # ``max_length`` is an absolute cap on ``generated.shape[1]``. Eval harnesses set it
            # to ``len(content_prompt) + max_gen_toks``, unaware that landmark generation inserts
            # memory tokens into the prompt. Those inserted tokens count toward
            # ``generated.shape[1]``, so without compensation they eat into (and can zero out) the
            # new-token budget. Extend the cap by the number of inserted landmark tokens so the
            # content-token budget stays ``max_gen_toks``.
            if landmark_active:
                max_length += prompt_len - orig_input_ids.shape[1]
        else:
            max_length = None  # Generate until EOS or stop tokens or OOM...

        prefill_cache_leftpad = None
        if generation_config.use_cache:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
            prefill_cache_leftpad = attention_mask_to_cache_leftpad(attention_mask).to(self.device)

        # Initialize/Reset the inference cache
        if generation_config.use_cache:
            if max_length is None:
                raise OLMoConfigurationError(
                    "max_length or max_new_tokens must be provided if use_cache is True"
                )
            self.prepare_inference_cache(batch_size, max_length)
        else:
            self.free_inference_cache()

        # Landmark hard top-k retrieval is applied only on single-query decode steps, never during the
        # batched prefill. The first generated token's logits come from the *last prompt token's*
        # query, so if prefill produced it that token would escape top-k entirely. Instead, prefill all
        # but the final prompt token here and let the decode loop below start from that final token --
        # then retrieval gates the first generated token too. (Requires a KV cache and >= 2 prompt
        # tokens; landmark generation already requires use_cache and batch_size=1 / no left-padding.)
        landmark_decode_first_token = (
            landmark_active and generation_config.use_cache and prompt_len >= 2
        )
        if landmark_decode_first_token:
            self._prefill_forward(
                input_ids[:, :-1],
                cache_leftpad=prefill_cache_leftpad,
                chunk_size=generation_config.prefill_chunk_size,
            )

        # Per-row string-level early-stop (much more effective than single-token stop_token_ids for
        # short-answer eval) + reduced finished-all sync. A row is marked finished once its decoded
        # completion contains a stop_string FOLLOWED BY a newline -- i.e. the answer line has closed
        # (so the answer itself is captured). Both the (host-side) decode and the finished.all() sync
        # run only every ``stop_string_check_interval`` steps to keep the decode loop GPU-bound.
        stop_strings_lc = [s.lower() for s in (generation_config.stop_strings or [])]
        do_string_stop = bool(stop_strings_lc) and stop_string_tokenizer is not None
        check_interval = max(1, generation_config.stop_string_check_interval)
        _STOP_DECODE_WINDOW = 256  # completion-tail tokens scanned for a freshly-closed anchor line
        step_idx = 0
        all_finished = False

        pbar = tqdm(
            desc="Generating tokens",
            unit="tokens",
            total=(max_length - prompt_len) if max_length is not None else None,
            disable=not log_timing,
            miniters=10,
            colour="blue",
        )
        while not ((max_length is not None and generated.shape[1] >= max_length) or all_finished):
            # Determine model inputs based on if we are prefilling or decoding
            if landmark_decode_first_token:
                # Prompt[:-1] is already cached above; every step is a single-token decode, starting
                # from the final prompt token (whose query yields the first generated token).
                input_ids_for_model = generated[:, -1:]
                cache_leftpad = None
            else:
                is_first_forward = generated.shape[1] == prompt_len
                input_ids_for_model = (
                    generated
                    if (is_first_forward or not generation_config.use_cache)
                    else generated[:, -1:]
                )
                cache_leftpad = (
                    prefill_cache_leftpad
                    if is_first_forward and generation_config.use_cache
                    else None
                )

            # Forward pass - handles both prefill and decode phases. Single-token decode steps fall
            # through ``_prefill_forward`` as one plain forward; only a multi-token prefill longer
            # than ``prefill_chunk_size`` is actually sliced.
            forward_start_time = time.perf_counter()
            next_token_logits = self._prefill_forward(  # (batch_size, seq_len=1, vocab_size)
                input_ids_for_model,
                cache_leftpad=cache_leftpad if generation_config.use_cache else None,
                chunk_size=generation_config.prefill_chunk_size,
            )

            # Landmark-gate analysis: flush this decode step's opened gates as one record (no-op on
            # the prefill forward, which records nothing).
            if landmark_active and landmark_gate_analysis.is_enabled():
                landmark_gate_analysis.finalize_token()

            next_tokens = select_next_token(
                next_token_logits.squeeze(1),
                do_sample=generation_config.do_sample,
                temperature=generation_config.temperature,
                top_k=generation_config.top_k,
                top_p=generation_config.top_p,
            )

            if all_logits is not None:
                all_logits.append(next_token_logits)
            if all_logprobs is not None:
                all_logprobs.append(
                    selective_log_softmax(next_token_logits, next_tokens.unsqueeze(-1))
                )

            # Force EOS for (previously) finished sequences
            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos), next_tokens)

            # Handle finished sequences
            stop_hit = next_tokens.eq(eos)
            if stop_tokens is not None:
                stop_hit |= next_tokens.unsqueeze(-1).eq(stop_tokens).any(dim=-1)
            finished |= stop_hit

            # Append next tokens
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=1)

            # Periodic string-level early-stop + finished-all sync (every check_interval steps).
            step_idx += 1
            if step_idx % check_interval == 0:
                if do_string_stop and not bool(finished.all().item()):
                    comp_tail = generated[:, prompt_len:][:, -_STOP_DECODE_WINDOW:].tolist()
                    for i in range(batch_size):
                        if finished[i]:
                            continue
                        text = stop_string_tokenizer.decode(
                            comp_tail[i], skip_special_tokens=True
                        ).lower()
                        for s in stop_strings_lc:
                            j = text.find(s)
                            if j >= 0 and text.find("\n", j + len(s)) >= 0:
                                finished[i] = True
                                break
                all_finished = bool(finished.all().item())

            if log_timing and tokens_generated == 0:
                torch.cuda.synchronize()
                decode_start_time = time.perf_counter()
                time_to_first_token = decode_start_time - forward_start_time
                setup_time = forward_start_time - start_time
            tokens_generated += 1
            pbar.update(1)
        pbar.close()

        logits = logprobs = None
        if return_logits and all_logits:
            logits = torch.cat(all_logits, dim=1)
        if return_logprobs and all_logprobs:
            logprobs = torch.cat(all_logprobs, dim=1)

        if log_timing:
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            total_time = end_time - start_time
            total_tokens = generated.numel()
            prefill_tokens = prompt_len * batch_size
            completion_tokens = tokens_generated * batch_size
            tokens_per_sec_total = total_tokens / total_time
            tokens_per_sec_per_seq = tokens_generated / total_time
            pad_count = (generated == pad).sum().item()
            pad_percentage = (pad_count / total_tokens) * 100 if total_tokens > 0 else 0.0

            stats_lines = [
                f"\n{'=' * 60}",
                "GENERATION STATISTICS",
                f"  Batch size: {batch_size:,} | Prompt len: {prompt_len:,} tokens",
                f"  Tokens generated: {tokens_generated:,} per sequence | Total: {total_tokens:,}",
                f"  Seq length: {prompt_len:,} → {prompt_len + tokens_generated:,}",
                f"  Padding stats: {pad_count:,} / {total_tokens:,} ({pad_percentage:.1f}%)",
            ]
            if decode_start_time and forward_start_time and time_to_first_token:
                decode_time = end_time - decode_start_time
                completion_time = end_time - decode_start_time
                stats_lines.append(
                    f"  Throughput:\n"
                    f"    Setup: {setup_time:.3f}s | Prefill: {time_to_first_token:.3f}s | Decode: {decode_time:.3f}s | Total: {total_time:.3f}s\n"
                    f"    Overall TPS: {tokens_per_sec_per_seq:.1f} /seq | {tokens_per_sec_total:.1f} /total\n"
                    f"    Prefill TPS: {prompt_len / time_to_first_token:.1f} /seq | {prefill_tokens / time_to_first_token:.1f} /total\n"
                    f"    Completion TPS: {tokens_generated / completion_time:.1f} /seq | {completion_tokens / completion_time:.1f} /total",
                )

            stats_lines.append(f"{'=' * 60}")
            log_or_print(log, "\n".join(stats_lines))

        if landmark_active:
            # ``generated`` holds the landmark-inserted prompt followed by plain content tokens; the
            # caller never asked for landmarks, so report the prompt in its original content space.
            self._clear_landmark_eval_decode()
            if landmark_gate_analysis.is_enabled():
                landmark_gate_analysis.end_example()
            completion = generated[:, prompt_len:]
            generated = (
                completion if completions_only else torch.cat([orig_input_ids, completion], dim=1)
            )
        elif completions_only:
            generated = generated[:, prompt_len:]
            # NOTE: completions_only does not apply to logits/logprobs. They are already computed only for completions.
        return generated, logits, logprobs

    def _align_prefill_chunk_size(self, chunk_size: Optional[int]) -> Optional[int]:
        """
        Round ``chunk_size`` down to a multiple of the landmark block size, if this is a landmark
        model.

        Landmark blocks are tied to absolute position, so a chunk that starts mid-block would shift
        the whole block structure; :meth:`FastLandmarkAttention._forward_generate` rejects it. Rather
        than make every caller know the model's ``mem_freq``, snap the request down here. Returns
        ``chunk_size`` unchanged for non-landmark models.

        :param chunk_size: The requested chunk size, or ``None``.

        :returns: The aligned chunk size, or ``None`` if chunking is off.
        """
        if not chunk_size:
            return chunk_size
        block_sizes = {
            int(bs)
            for attn in self._landmark_attention_layers()
            if (bs := getattr(attn, "block_size", 0))
        }
        if not block_sizes:
            return chunk_size
        # A single stride that satisfies every landmark layer at once.
        stride = math.lcm(*block_sizes)
        aligned = (chunk_size // stride) * stride
        if aligned == 0:
            raise OLMoConfigurationError(
                f"prefill_chunk_size={chunk_size} is smaller than the landmark block size "
                f"({stride}); it cannot be aligned to a block boundary."
            )
        if aligned != chunk_size:
            log.warning(
                "prefill_chunk_size=%d is not a multiple of the landmark block size %d; "
                "using %d instead (chunks must start on a block boundary).",
                chunk_size,
                stride,
                aligned,
            )
        return aligned

    def _prefill_forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache_leftpad: Optional[torch.Tensor],
        chunk_size: Optional[int],
    ) -> torch.Tensor:
        """
        Run the prefill forward, optionally in chunks to bound activation memory.

        See :data:`GenerationConfig.prefill_chunk_size`. Each chunk advances the KV cache (and the
        recurrent state on Gated DeltaNet layers), so the last chunk's logits are the same ones a
        single-shot prefill would return.

        :param input_ids: The prompt token IDs of shape ``(batch_size, seq_len)``.
        :param cache_leftpad: Per-row left-padding offsets, recorded on the first chunk only
            (``KVCacheManager.record_leftpad`` ignores ``None``, so the value persists).
        :param chunk_size: Tokens per chunk, or ``None`` for a single forward.

        :returns: Logits for the final prompt position, shape ``(batch_size, 1, vocab_size)``.
        """
        seq_len = input_ids.shape[1]
        chunk_size = self._align_prefill_chunk_size(chunk_size)
        if not chunk_size or seq_len <= chunk_size:
            return self.model(input_ids, logits_to_keep=1, cache_leftpad=cache_leftpad)

        logits = None
        for start in range(0, seq_len, chunk_size):
            logits = self.model(
                input_ids[:, start : start + chunk_size],
                logits_to_keep=1,
                # Only the first chunk carries the left-pad offsets; later chunks must not reset it.
                cache_leftpad=cache_leftpad if start == 0 else None,
            )
        assert logits is not None
        return logits

    def supports_landmark_ragged_batch(self) -> bool:
        """True if every landmark layer supports the right-padded cross-length batched decode
        (:meth:`generate_landmark_batch`)."""
        layers = self._landmark_attention_layers()
        return bool(layers) and all(getattr(a, "_supports_ragged_decode", False) for a in layers)

    @torch.inference_mode()
    def generate_landmark_batch(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int,
        decode_mode: str = "extend_last_block",
        top_k_fraction: Optional[float] = 0.1,
        top_k_blocks: Optional[int] = None,
        stop_strings: Optional[List[str]] = None,
        stop_string_tokenizer: Optional[Any] = None,
        stop_string_check_interval: int = 16,
    ) -> List[List[int]]:
        """Greedy generation for a **right-padded, cross-length batch** of landmark prompts.

        Landmark blocks are tied to absolute position, so the legacy path forbids left-padding and can
        only batch prompts of *exactly* equal length (effective batch size ~= 1 on variable-length
        tasks). Right-padding is legal instead: each row's content keeps absolute positions
        ``0..L_i`` and the pad TAIL is masked. This builds each row's landmark prompt independently,
        right-pads to a common block-aligned length, prefills the whole batch in one shot (the pad
        tail is causally future, so real-position outputs are bit-identical to a bs=1 prefill), then
        decodes every row in lockstep with its OWN absolute position / prompt length / top-k via
        :meth:`FastLandmarkAttention._decode_ragged`. Row-for-row identical to the legacy bs=1 path.

        :param prompts: Content-token id lists (no landmark tokens), variable length.
        :param max_new_tokens: Max **content** tokens to generate per row.
        :param decode_mode: ``"extend_last_block"`` or ``"generation_only"``.
        :param top_k_fraction: Per-row hard top-k = ``ceil(fraction * num_blocks_row)`` (the landmark
            paper's inference retrieval). ``top_k_blocks`` overrides it; both ``None`` -> dense gating.
        :param stop_strings: Per-row early-stop strings (answer-line anchors); requires
            ``stop_string_tokenizer``.

        :returns: One generated **content**-token id list per input prompt (EOS/pad not trimmed).
        """
        if not self.supports_landmark_ragged_batch():
            raise OLMoConfigurationError(
                "generate_landmark_batch requires a landmark model whose layers support ragged "
                "decode (FastLandmarkAttention / SparseLandmarkAttention)."
            )
        self._set_model_mode("eval")
        gen_cfg = self._generation_config
        if gen_cfg.landmark_mem_id is None:
            raise OLMoConfigurationError("Set GenerationConfig.landmark_mem_id to generate.")
        layers = self._landmark_attention_layers()
        mem_freqs = {int(getattr(a, "mem_freq")) for a in layers}
        if len(mem_freqs) != 1:
            raise OLMoConfigurationError(f"Inconsistent mem_freq: {sorted(mem_freqs)}")
        nums_landmarks = {int(getattr(a, "num_landmarks", 1)) for a in layers}
        if len(nums_landmarks) != 1:
            raise OLMoConfigurationError(f"Inconsistent num_landmarks: {sorted(nums_landmarks)}")
        mem_freq = mem_freqs.pop()
        num_landmarks = nums_landmarks.pop()
        block_size = self._landmark_block_size(layers)  # == mem_freq + num_landmarks
        self._maybe_enable_sparse_decode()
        mem_id = gen_cfg.landmark_mem_id
        pad_id = (
            gen_cfg.landmark_pad_id if gen_cfg.landmark_pad_id is not None else gen_cfg.pad_token_id
        )
        eos = gen_cfg.eos_token_id
        dev = self.device
        B = len(prompts)

        # Per-row landmark prompt (independent block structure), then right-pad to a common,
        # block-aligned length P.
        lm_rows: List[torch.Tensor] = []
        for p in prompts:
            ids = torch.tensor([p], dtype=torch.long, device=dev)
            lm = _build_landmark_prompt(
                ids,
                mem_freq,
                mem_id,
                mode=decode_mode,
                pad_id=pad_id,
                num_landmarks=num_landmarks,
            )
            lm_rows.append(lm[0])
        prompt_lens = torch.tensor([r.numel() for r in lm_rows], dtype=torch.long, device=dev)
        P = int(prompt_lens.max().item())
        P += (-P) % block_size  # block-aligned for the fused prefill kernel
        padded = torch.full((B, P), pad_id, dtype=torch.long, device=dev)
        for i, r in enumerate(lm_rows):
            padded[i, : r.numel()] = r

        max_length = P + max_new_tokens + 1
        self.prepare_inference_cache(B, max_length)
        # Decode uses per-row ``position_ids`` (absolute positions up to ``max_length``); that RoPE
        # branch reuses the pre-warmed sin/cos buffer sized from the cache and does NOT grow it, so
        # warm every layer's RoPE to ``max_length`` up front (prefill alone only warms it to P, and
        # generated tokens sit beyond P -> out-of-bounds index without this).
        for block in self.model.blocks.values():
            attn = block.attention
            rope = getattr(attn, "rope", None)
            if rope is not None and hasattr(rope, "warmup_cache"):
                rope.warmup_cache(max_length, self.device)

        # One-shot batched prefill (fills KV cache 0..P-1 for every row; pad tail is causally future).
        leftpad = torch.zeros(B, dtype=torch.int32, device=dev)
        self.model(padded, logits_to_keep=1, cache_leftpad=leftpad)

        # Per-row hard top-k from each row's own block count.
        if top_k_blocks is not None:
            top_k = torch.full((B,), int(top_k_blocks), dtype=torch.long, device=dev)
        elif top_k_fraction is not None:
            nblk = torch.clamp(prompt_lens // block_size, min=1)
            top_k = torch.clamp(torch.ceil(top_k_fraction * nblk.float()).long(), min=1)
        else:
            top_k = None
        for a in layers:
            a.set_landmark_ragged_decode(prompt_lens, mode=decode_mode, top_k=top_k)  # type: ignore[attr-defined]

        bidx = torch.arange(B, device=dev)
        # First decode step re-queries each row's final prompt token (so top-k gates the first
        # generated token, matching the legacy path), then writes generated tokens at p..p+gen-1.
        pos = prompt_lens - 1
        cur = padded[bidx, pos]  # (B,) last real prompt token of each row
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        comp: List[List[int]] = [[] for _ in range(B)]
        stop_lc = [s.lower() for s in (stop_strings or [])]
        do_stop = bool(stop_lc) and stop_string_tokenizer is not None
        check = max(1, stop_string_check_interval)

        for step in range(max_new_tokens):
            for a in layers:
                a.set_ragged_qpos(pos)  # type: ignore[attr-defined]
            logits = self.model(cur.view(B, 1), logits_to_keep=1)  # (B,1,V)
            nxt = logits[:, -1].argmax(dim=-1)  # greedy
            nxt = torch.where(finished, torch.full_like(nxt, eos), nxt)
            for i in range(B):
                if not finished[i]:
                    comp[i].append(int(nxt[i].item()))
            finished = finished | nxt.eq(eos)
            cur = nxt
            pos = pos + 1
            if bool(finished.all().item()):
                break
            if do_stop and (step + 1) % check == 0:
                for i in range(B):
                    if finished[i]:
                        continue
                    text = stop_string_tokenizer.decode(
                        comp[i][-256:], skip_special_tokens=True
                    ).lower()
                    for s in stop_lc:
                        j = text.find(s)
                        if j >= 0 and text.find("\n", j + len(s)) >= 0:
                            finished[i] = True
                            break

        for a in layers:
            a.clear_ragged_decode()  # type: ignore[attr-defined]
            a.clear_landmark_eval_decode()  # type: ignore[attr-defined]
        return comp

    def load_checkpoint(
        self,
        checkpoint_dir: PathOrStr,
        work_dir: PathOrStr,
        process_group: Optional[ProcessGroup] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
    ):
        """
        Load model checkpoint.

        Args:
            checkpoint_dir: Path to checkpoint directory
            work_dir: Working directory for caching remote checkpoints
            process_group: Process group for distributed loading
            pre_download: Whether to pre-download remote checkpoints
            load_thread_count: Number of threads to use for loading the checkpoint

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist
            RuntimeError: If checkpoint loading fails
        """
        work_dir = Path(work_dir)
        if get_fs_local_rank() == 0:
            work_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dir = normalize_path(checkpoint_dir)
        train_module_dir = join_path(checkpoint_dir, "model_and_optim")
        metadata: Optional[Metadata] = None
        if get_rank(process_group) == 0:
            try:
                metadata = get_checkpoint_metadata(train_module_dir)
            except FileNotFoundError:
                # Try base directory, which could be the case if user is trying to
                # load model weights and not an actual train checkpoint.
                log_or_print(
                    log,
                    f"Checkpoint metadata not found in '{train_module_dir}', "
                    f"trying base directory '{checkpoint_dir}'",
                    level=logging.WARNING,
                )
                metadata = get_checkpoint_metadata(checkpoint_dir)
                train_module_dir = checkpoint_dir

        train_module_dir = broadcast_object(train_module_dir)
        if metadata is None:
            metadata = get_checkpoint_metadata(train_module_dir)

        state_dict = self.state_dict_to_load(metadata)
        load_state_dict(
            train_module_dir,
            state_dict,  # state is loaded into here, in-place
            process_group=process_group,
            pre_download=is_url(checkpoint_dir) and pre_download,
            work_dir=work_dir,
            thread_count=load_thread_count,
        )
        self.load_state_dict(state_dict)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: PathOrStr,
        *,
        transformer_config: Optional[TransformerConfig] = None,
        generation_config: Optional[GenerationConfig] = None,
        process_group: Optional[ProcessGroup] = None,
        work_dir: Optional[PathOrStr] = None,
        pre_download: bool = True,
        load_thread_count: Optional[int] = None,
        dtype: Optional[DType] = None,
        attention_backend: Optional[AttentionBackendName] = None,
        post_build_hook: Optional[Callable[["Transformer"], None]] = None,
        **kwargs,
    ) -> "TransformerGenerationModule":
        """
        Create a GenerationModule from a checkpoint.

        This is a convenience method that combines model initialization and checkpoint loading.

        Args:
            checkpoint_dir: Path to checkpoint directory.
            transformer_config: Configuration for the transformer model. If not provided,
                will be loaded from the checkpoint's config.json file.
            generation_config: Configuration for generation. If not provided, uses default
                GenerationConfig.
            process_group: Process group for distributed checkpoint loading.
            pre_download: Whether to pre-download remote checkpoints.
            load_thread_count: Number of threads to use for loading checkpoint.
            dtype: If provided, build the model with this dtype.
            attention_backend: If provided, override the config to use this attention backend.
            post_build_hook: Called on the freshly built model BEFORE the checkpoint is loaded.
                Use it for ``enable_*`` methods that ADD parameters (e.g.
                ``enable_nested_ffn_moe``), whose keys must exist for the load to populate them.
            **kwargs: Additional keyword arguments passed to the TransformerGenerationModule
                constructor.

        Returns:
            TransformerGenerationModule instance with loaded checkpoint.

        Raises:
            FileNotFoundError: If checkpoint directory doesn't exist.
            OLMoConfigurationError: If transformer config cannot be determined.
            RuntimeError: If checkpoint loading fails.
        """
        checkpoint_dir = normalize_path(checkpoint_dir)

        # Load transformer config from checkpoint if not provided
        tokenizer_config = None
        checkpoint_landmark_mem_id: Optional[int] = None
        if transformer_config is None and get_rank(process_group) == 0:
            config_path = join_path(checkpoint_dir, "config.json")
            with cached_path(config_path).open() as f:
                config_dict = json.load(f)
            try:
                # Avoid loading the entire experiment config b/c we don't care about validation outside
                # of the transformer config and the tokenizer config
                transformer_config = TransformerConfig.from_dict(config_dict["model"])
            except KeyError as e:
                raise OLMoConfigurationError(
                    f"Failed to load config from checkpoint at {config_path}: missing required field {e}"
                ) from e

            # The tokenizer config is only needed to synthesize a generation config when one
            # isn't provided. Tolerate checkpoints whose config stores the tokenizer elsewhere
            # or stores `dataset` in a different shape (e.g. composable data configs where
            # `dataset` is a list of sources), so loading still works when a generation config
            # is passed explicitly.
            try:
                tokenizer_config = TokenizerConfig.from_dict(config_dict["dataset"]["tokenizer"])
            except (KeyError, TypeError):
                tokenizer_config = None

            # For composable data pipelines the dataset field is a list of InstanceSourceConfig
            # dicts; a LandmarkInstanceSourceConfig carries `mem_id` at the top level.
            dataset_cfg = config_dict.get("dataset")
            if isinstance(dataset_cfg, list):
                for src in dataset_cfg:
                    if isinstance(src, dict) and "mem_id" in src:
                        checkpoint_landmark_mem_id = int(src["mem_id"])
                        break

        # Create work directory on rank 0
        work_dir = Path(
            work_dir or (tempfile.mkdtemp() if get_rank(process_group) == 0 else "/tmp")
        )

        # Broadcast config and work_dir to all ranks
        (
            transformer_config,
            work_dir,
            tokenizer_config,
            checkpoint_landmark_mem_id,
        ) = broadcast_object(
            (transformer_config, work_dir, tokenizer_config, checkpoint_landmark_mem_id)
        )

        if transformer_config is None:
            raise OLMoConfigurationError(
                "Transformer config is required. Either provide a transformer config or a "
                "checkpoint with a transformer config."
            )

        if generation_config is None:
            if tokenizer_config is None:
                raise OLMoConfigurationError(
                    "Tokenizer config is required to build generation config. Either provide a "
                    "generation config or a checkpoint with a tokenizer config."
                )
            generation_config = GenerationConfig(
                pad_token_id=tokenizer_config.pad_token_id,
                eos_token_id=tokenizer_config.eos_token_id,
                landmark_mem_id=checkpoint_landmark_mem_id,
            )
            log_or_print(
                log,
                f"No generation config provided, using defaults from checkpoint config: {generation_config}",
            )

        # Auto-fill landmark_mem_id from checkpoint config when the caller didn't set it.
        if checkpoint_landmark_mem_id is not None and generation_config.landmark_mem_id is None:
            import dataclasses

            generation_config = dataclasses.replace(
                generation_config, landmark_mem_id=checkpoint_landmark_mem_id
            )
            log_or_print(
                log,
                f"Auto-filled landmark_mem_id={checkpoint_landmark_mem_id} from checkpoint config",
            )

        # Build model and generation module
        if dtype is not None:
            dtype = DType(dtype)
            transformer_config.apply(
                lambda c: setattr(c, "dtype", dtype) if hasattr(c, "dtype") else None
            )

        if attention_backend is not None:
            attention_backend.assert_supported()

            def set_attention_backend(c):
                mixer = getattr(c, "sequence_mixer", None) or getattr(c, "attention", None)
                if mixer is None and hasattr(c, "backend"):
                    mixer = c
                if mixer is not None and hasattr(mixer, "backend"):
                    setattr(mixer, "backend", attention_backend)

            transformer_config.apply(set_attention_backend)

        log_or_print(log, f"{transformer_config}")
        log_or_print(log, f"{generation_config}")
        model = transformer_config.build()
        if post_build_hook is None and os.environ.get("OLMO_CONFIG_BUILD_HOOK", "1") != "0":
            # Callers that do not pass a hook (e.g. olmo-eval's olmo_core provider) still get the
            # features recorded in config.json (early-LoRA adapters, ffn_moe/kv_route routers)
            # plus the key guard; without it such a checkpoint loads silently as the bare backbone.
            # OLMO_CONFIG_BUILD_HOOK=0 restores the old behaviour.
            from olmo_core.nn.nested_ffn_moe import post_build_hook_from_config

            post_build_hook = post_build_hook_from_config(str(checkpoint_dir))
        if post_build_hook is not None:
            # Runs BEFORE the checkpoint load, so a hook that adds parameters (e.g.
            # ``enable_nested_ffn_moe``'s router) has its keys present when the load happens --
            # enabling such a feature afterwards would silently leave it at its init values.
            post_build_hook(model)
        generation_module = cls(model, generation_config, **kwargs)

        # Load checkpoint
        generation_module.load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            process_group=process_group,
            work_dir=work_dir,
            pre_download=pre_download,
            load_thread_count=load_thread_count,
        )

        return generation_module

    @classmethod
    @torch.no_grad()
    def from_checkpoints(
        cls,
        checkpoint_dirs: List[PathOrStr],
        dtype: Optional[DType] = None,
        **kwargs,
    ) -> "TransformerGenerationModule":
        if len(checkpoint_dirs) == 1:
            return cls.from_checkpoint(checkpoint_dirs[0], dtype=dtype, **kwargs)

        log.info(f"Merging {len(checkpoint_dirs)} checkpoints")

        # Override device to CPU for intermediate checkpoint loading to save GPU memory
        cpu_kwargs = kwargs.copy()
        cpu_kwargs["device"] = torch.device("cpu")

        log.info(
            "Loading checkpoints on CPU for merging, will transfer to target device at the end"
        )
        log.info(f"Merging {checkpoint_dirs[0]}")
        first_generation_module = cls.from_checkpoint(
            checkpoint_dirs[0], dtype=DType.float32, **cpu_kwargs
        )

        # Get the state dict from the first module to use as accumulator
        merged_state_dict = first_generation_module.state_dict()

        # Free the first module since we have its state dict
        del first_generation_module
        gc_cuda()

        # Average weights from all checkpoints
        for i, checkpoint_dir in enumerate(checkpoint_dirs[1:], start=2):
            log.info(f"Merging {checkpoint_dir}")
            next_generation_module = cls.from_checkpoint(
                checkpoint_dir, dtype=DType.float32, **cpu_kwargs
            )
            next_state_dict = next_generation_module.state_dict()
            del next_generation_module

            # Average the weights
            for key in merged_state_dict["model"].keys():
                target_tensor = merged_state_dict["model"][key]
                if torch.is_tensor(target_tensor) and torch.is_floating_point(target_tensor):
                    source_tensor = next_state_dict["model"].pop(key)
                    assert target_tensor.shape == source_tensor.shape
                    # in-place operations for better memory consumption
                    target_tensor.mul_((i - 1) / i).add_(source_tensor, alpha=1.0 / i)
                    del target_tensor
                    del source_tensor

            # Free memory from the temporary module and run garbage collection
            del next_state_dict
            gc_cuda()

        # Now load the final model on the target device with the correct dtype
        if dtype == DType.float32:
            # If target dtype is float32, we can reuse the merged state dict
            log.info("Loading merged checkpoint with dtype float32")
            final_generation_module = cls.from_checkpoint(
                checkpoint_dirs[0], dtype=DType.float32, **kwargs
            )
            final_generation_module.load_state_dict(merged_state_dict)
            return final_generation_module

        # Otherwise, load with the target dtype and convert the merged weights
        log.info(f"Loading merged checkpoint with dtype {dtype}")
        final_generation_module = cls.from_checkpoint(checkpoint_dirs[0], dtype=dtype, **kwargs)
        final_state_dict = final_generation_module.state_dict()
        assert merged_state_dict["model"].keys() == final_state_dict["model"].keys()

        # Convert merged state dict to the target dtype
        for key in merged_state_dict["model"].keys():
            merged_state_dict_tensor = merged_state_dict["model"][key]
            if not torch.is_tensor(merged_state_dict_tensor):
                continue
            final_state_dict_tensor = final_state_dict["model"][key]
            if not torch.is_tensor(final_state_dict_tensor):
                continue
            if merged_state_dict_tensor.dtype != final_state_dict_tensor.dtype:
                merged_state_dict["model"][key] = merged_state_dict_tensor.to(
                    final_state_dict_tensor.dtype
                )

        final_generation_module.load_state_dict(merged_state_dict)

        gc_cuda()

        return final_generation_module
