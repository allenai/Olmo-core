"""
Check CompletedP on the hero architecture (KDA, gated full attention, latent MoE, peri-norm).

Trains hero models for a few steps on random data with the real
:class:`~olmo_core.train.Trainer`, :class:`OLMoDDPTrainModule` and
:class:`~olmo_core.optim.OLMoDDPOptimizerConfig`. Two families of sizes:

* ``d<width>-L<depth>``: hero-like models from the hero block builders with 64 experts (top-8),
  ``d_model / 128`` heads, block 0 dense and full attention every 8th block, on one GPU. The base
  model is ``d512-L8``, and CompletedP runs use a training duration twice the base one, so the
  batch-size/duration rules are exercised too.
* ``small`` and ``medium``: the exact production hero configs (``build_model_config``, with the
  per-head QK-norm gains of the integration runs), on all GPUs. The base model is the small hero
  (``base_width`` 1024, ``base_depth`` 16) at the same batch size and duration. ``medium`` runs
  with expert parallelism over all ranks on the synchronized all-to-all path. These runs skip the
  coordinate check; ``report`` prints how every parameter was scaled from small to medium next to
  the scaling Table 1 predicts.

``run`` asserts that:

1. every block's residual multipliers are ``(n_layers / base_depth) ** -alpha``, and every hidden
   matrix and the LM head start with the CompletedP init std;
2. every trainable parameter is in exactly one optimizer group, whose ``lr``, ``eps``,
   ``weight_decay`` and ``betas`` equal the CompletedP values computed here from the parameter's
   name and layout;
3. every group's LR equals the scheduled LR times its multiplier at every step;
4. the first optimizer step did exactly what AdamW with those values predicts: from the captured
   gradient and the clip coefficient it recomputes ``exp_avg``, ``exp_avg_sq`` and the new fp32
   weights, which checks the betas, epsilon, learning rate and weight decay actually used.

It also records how much the activations on the training batch move (a coordinate check);
``report`` prints the width and depth exponents of those changes. ``run`` trains on one fixed
batch: on fresh random tokens there is nothing to learn, so the updates are noise and add up
incoherently, which is not the regime the parameterization is derived for. ``bench`` measures throughput with the model
and optimizer compiled, with CompletedP on or off.

Run on one GPU, then report::

    for spec in d512-L8 d1024-L8 d2048-L8 d512-L16 d512-L32; do
      for param in cp sp; do
        PYTHONPATH=src torchrun --nproc-per-node=1 src/examples/completedp/hero_check.py run $spec $param
      done
    done
    # Optionally, the depth sweep with another depth exponent:
    for spec in d512-L8 d512-L16 d512-L32; do
      PYTHONPATH=src torchrun --nproc-per-node=1 src/examples/completedp/hero_check.py run $spec cp --alpha 0.5
    done
    for param in sp cp sp cp; do
      PYTHONPATH=src torchrun --nproc-per-node=1 src/examples/completedp/hero_check.py bench d1024-L16 $param
    done
    PYTHONPATH=src python src/examples/completedp/hero_check.py report

The production small and medium heroes, on 8 GPUs::

    PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/completedp/hero_check.py run small cp
    PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/completedp/hero_check.py run medium cp
    PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/completedp/hero_check.py run medium cp --alpha 0.5
    PYTHONPATH=src python src/examples/completedp/hero_check.py report
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "olmo_ddp"))

import olmoe3_small_medium_models as hero  # noqa: E402

from olmo_core.config import DType  # noqa: E402
from olmo_core.data import TokenizerConfig  # noqa: E402
from olmo_core.data.composable import (  # noqa: E402
    ComposableDataLoaderConfig,
    RandomInstanceSourceConfig,
)
from olmo_core.data.utils import get_labels  # noqa: E402
from olmo_core.distributed.parallel import DataParallelType  # noqa: E402
from olmo_core.distributed.utils import get_rank  # noqa: E402
from olmo_core.nn.moe.v2.ep_config import ExpertParallelPath  # noqa: E402
from olmo_core.nn.transformer import CompletedPConfig, OLMoDDPModelConfig  # noqa: E402
from olmo_core.optim import (  # noqa: E402
    CompletedPOptimConfig,
    OLMoDDPOptimizerConfig,
    OptimGroupOverride,
)
from olmo_core.optim.scheduler import WSD  # noqa: E402
from olmo_core.train import (  # noqa: E402
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import Callback  # noqa: E402
from olmo_core.train.train_module import (  # noqa: E402
    OLMoDDPTrainModuleConfig,
    TransformerDataParallelConfig,
    TransformerExpertParallelConfig,
)
from olmo_core.utils import seed_all  # noqa: E402

TOKENIZER = TokenizerConfig.dolma2()
NUM_EXPERTS = 64
TOP_K = 8
BASE_WIDTH = 512
BASE_DEPTH = 8
HERO_SIZES = ("small", "medium")
HERO_BASE_WIDTH = 1024
HERO_BASE_DEPTH = 16
ALPHA = 1.0
LR, EPS, WEIGHT_DECAY, BETAS = 8e-4, 1e-8, 0.1, (0.9, 0.95)
INIT_STD = 0.02
STEPS = 20
SEQ_LEN = 4096
RANK_SEQUENCES = 2
BENCH_STEPS = 60
BENCH_WARMUP_STEPS = 20
BENCH_RANK_SEQUENCES = 4
LR_RTOL = 1e-6
SITES = ("embedding", "block 1", "middle block", "last block", "logits")


def parse_spec(spec: str) -> Tuple[int, int]:
    if spec in HERO_SIZES:
        g = hero.geometry(spec)
        return g.d_model, g.n_layers
    match = re.fullmatch(r"d(\d+)-L(\d+)", spec)
    if match is None:
        raise ValueError(f"Invalid spec {spec!r}; expected d<width>-L<depth>")
    return int(match.group(1)), int(match.group(2))


def spec_bases(spec: str) -> Tuple[int, int]:
    """
    The ``base_width`` and ``base_depth`` of the family that ``spec`` belongs to.
    """
    if spec in HERO_SIZES:
        return HERO_BASE_WIDTH, HERO_BASE_DEPTH
    return BASE_WIDTH, BASE_DEPTH


def spec_sde_ratio(spec: str) -> float:
    """
    The batch-size ratio over the training-duration ratio of a CompletedP run of ``spec``.
    """
    return 1.0 if spec in HERO_SIZES else 0.5


def build_hero_config(size: str, *, completedp: bool) -> OLMoDDPModelConfig:
    """
    Build the production small or medium hero, with CompletedP or with standard parameterization.
    """
    config = hero.build_model_config(size, eos_token_id=TOKENIZER.eos_token_id)
    for block in [config.block, *config.block_overrides.values()]:
        mixer = block.sequence_mixer
        if hasattr(mixer, "use_cute_kernel"):
            mixer.use_cute_kernel = False
        if hasattr(mixer, "qk_norm_per_head_gains"):
            mixer.qk_norm_per_head_gains = True
        if size == "medium" and getattr(block, "ep", None) is not None:
            block.ep.path = ExpertParallelPath.sync_1d
    if completedp:
        config.completedp = CompletedPConfig(
            base_width=HERO_BASE_WIDTH, base_depth=HERO_BASE_DEPTH, alpha=ALPHA
        )
    config.validate()
    return config


def build_model_config(spec: str, *, completedp: bool) -> OLMoDDPModelConfig:
    """
    Build a hero-like model, with CompletedP or with standard parameterization.
    """
    if spec in HERO_SIZES:
        return build_hero_config(spec, completedp=completedp)
    d_model, n_layers = parse_spec(spec)
    config = hero.build_model_config("small", eos_token_id=TOKENIZER.eos_token_id)
    n_heads = d_model // hero.HEAD_DIM
    g = hero.Geometry(
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=max(n_heads // 2, 1),
        expected_active_params=0,
        expected_active_non_embedding_params=0,
        expected_total_params=0,
    )
    norm = hero._norm()
    emo = deepcopy(config.block.routed_experts_router.emo)
    emo.max_document_expert_pool = NUM_EXPERTS
    emo.eval_document_expert_pool = NUM_EXPERTS
    emo.validate_for_router(num_experts=NUM_EXPERTS, top_k=TOP_K)
    kda_block = hero._moe_block(g, norm, hero._kda(g), emo)
    attention_block = hero._moe_block(g, norm, hero._attention(g, norm), emo)
    for block in (kda_block, attention_block):
        block.routed_experts.num_experts = NUM_EXPERTS
        block.routed_experts_router.num_experts = NUM_EXPERTS
        block.routed_experts_router.top_k = TOP_K
    config.d_model = d_model
    config.n_layers = n_layers
    config.embed_scale = math.sqrt(d_model)
    config.block = kda_block
    config.block_overrides = {
        0: hero._dense_first(g, norm),
        **{i: deepcopy(attention_block) for i in g.full_attention_layers},
    }
    for block in [config.block, *config.block_overrides.values()]:
        mixer = block.sequence_mixer
        if hasattr(mixer, "use_cute_kernel"):
            mixer.use_cute_kernel = False
    if completedp:
        config.completedp = CompletedPConfig(
            base_width=BASE_WIDTH, base_depth=BASE_DEPTH, alpha=ALPHA
        )
    config.validate()
    return config


def optim_completedp_config(global_batch_size: int, sde_ratio: float) -> CompletedPOptimConfig:
    # The base batch size, and a base duration of ``sde_ratio`` times this run's.
    return CompletedPOptimConfig(
        base_batch_size=global_batch_size,
        batch_size=global_batch_size,
        base_train_tokens=int(global_batch_size * STEPS * sde_ratio),
        train_tokens=global_batch_size * STEPS,
    )


def expected_fan_in(name: str, shape: Tuple[int, ...], d_model: int) -> Optional[int]:
    """
    The fan-in of a weight matrix, or ``None`` for embeddings and vector-like parameters, derived
    from the parameter's name and tensor layout only.
    """
    module = name.rsplit(".", 2)[-2] if name.count(".") >= 1 else ""
    if name.endswith("embeddings.weight") or "norm" in module or "conv1d" in module:
        return None
    if name.endswith("routed_experts_router.weight"):  # flattened [E * D]
        return d_model
    if len(shape) == 1:
        return None
    if name.endswith("routed_experts.w_up_gate"):  # [E, 2H, D]
        return shape[2]
    if name.endswith("experts.w_down"):  # [E, H, D]
        return shape[1]
    if name.endswith("shared_experts.w_up_gate"):  # [D, E * 2H]
        return shape[0]
    if len(shape) != 2 or not name.endswith(".weight"):
        raise ValueError(f"No fan-in rule for {name} {shape}")
    return shape[1]  # nn.Linear [out, in]


def expected_hyperparameters(
    name: str,
    shape: Tuple[int, ...],
    d_model: int,
    n_layers: int,
    *,
    completedp: bool,
    base_width: int = BASE_WIDTH,
    base_depth: int = BASE_DEPTH,
    sde_ratio: float = 0.5,
) -> Dict[str, Any]:
    """
    The CompletedP role, fan-in and ``lr`` / ``eps`` / ``weight_decay`` multipliers of a parameter
    (Table 1 of the paper), plus its base weight decay and the expected betas.
    """
    name = name.removeprefix("module.")
    base_weight_decay = 0.0 if name == "embeddings.weight" else WEIGHT_DECAY
    if not completedp:
        return dict(
            role="sp",
            fan_in=None,
            lr=1.0,
            eps=1.0,
            weight_decay=1.0,
            betas=BETAS,
            base_weight_decay=base_weight_decay,
        )
    m_n = d_model / base_width
    m_l = n_layers / base_depth
    ratio = sde_ratio
    s = ratio**0.5
    fan_in = expected_fan_in(name, shape, d_model)
    if name == "embeddings.weight" or name.startswith("embedding_norm."):
        role, mults = "input", (1.0, 1 / m_n, 1.0)
    elif name == "lm_head.w_out.weight":
        assert fan_in is not None
        r = fan_in / base_width
        role, mults = "output matrix", (1 / r, 1.0, r)
    elif name.startswith("lm_head."):
        role, mults = "output vector", (1.0, 1.0, 1.0)
    elif fan_in is not None and re.search(
        r"(experts_router|attention\.f_proj_1|attention\.g_proj_1)\.weight$", name
    ):
        # Output size fixed by the expert count or KDA's 256-wide gate bottlenecks.
        r = fan_in / base_width
        role, mults = "readout matrix", (m_l ** (ALPHA - 1) / r, m_l**-ALPHA, r)
    elif fan_in is not None:
        r = fan_in / base_width
        role, mults = "hidden matrix", (m_l ** (ALPHA - 1) / r, m_l**-ALPHA / m_n, r)
    elif (re.search(r"attention\.[qk]_norm\.weight$", name) and len(shape) == 1) or name.endswith(
        "attention.o_norm.weight"
    ):
        role, mults = "head-shared vector", (m_l ** (ALPHA - 1), m_l**-ALPHA, 1.0)
    else:
        role, mults = "hidden vector", (m_l ** (ALPHA - 1), m_l**-ALPHA / m_n, 1.0)
    lr, eps, weight_decay = mults
    return dict(
        role=role,
        fan_in=fan_in,
        lr=lr * s,
        eps=eps / s,
        weight_decay=weight_decay * s,
        betas=(1 - (1 - BETAS[0]) * ratio, 1 - (1 - BETAS[1]) * ratio),
        base_weight_decay=base_weight_decay,
    )


def param_role(name: str, attention_layers: Tuple[int, ...]) -> str:
    """
    A width- and depth-independent label: the name without its block index, prefixed by the block
    type (``dense`` for block 0, ``attn`` for full attention, ``moe`` for the other blocks).
    """
    name = name.removeprefix("module.")
    match = re.match(r"blocks\.(\d+)\.(.*)", name)
    if match is None:
        return name
    index = int(match.group(1))
    kind = "dense" if index == 0 else "attn" if index in attention_layers else "moe"
    return f"{kind}/{match.group(2)}"


def _unwrap(module: torch.nn.Module) -> torch.nn.Module:
    return getattr(module, "module", module)


def _close(a: float, b: float, rel_tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-30)


@dataclass
class CompletedPCheck(Callback):
    """
    Asserts the per-group hyperparameters and the first AdamW step, and records the coordinate
    check. See the module docstring.
    """

    spec: str
    completedp: bool
    out_path: str
    d_model: int
    n_layers: int
    attention_layers: Tuple[int, ...]
    base_width: int = BASE_WIDTH
    base_depth: int = BASE_DEPTH
    sde_ratio: float = 0.5
    probe: bool = True
    failures: List[str] = field(default_factory=list)
    records: Dict[str, Any] = field(default_factory=dict)
    _snapshot: Dict[str, torch.Tensor] = field(default_factory=dict)
    _grads: Dict[str, torch.Tensor] = field(default_factory=dict)
    _probe_initial: Dict[str, torch.Tensor] = field(default_factory=dict)
    _probe_ids: Optional[torch.Tensor] = None

    @property
    def optim(self):
        return self.trainer.train_module.optim

    def _fail(self, message: str):
        if len(self.failures) < 200:
            self.failures.append(f"[rank {get_rank()}] {message}")

    def _expected(self, name: str, shape: Tuple[int, ...], completedp: bool) -> Dict[str, Any]:
        return expected_hyperparameters(
            name,
            shape,
            self.d_model,
            self.n_layers,
            completedp=completedp,
            base_width=self.base_width,
            base_depth=self.base_depth,
            sde_ratio=self.sde_ratio,
        )

    def _check_model(self):
        model = _unwrap(self.trainer.train_module.model)
        multiplier = (self.n_layers / self.base_depth) ** -ALPHA if self.completedp else 1.0
        for key, block in model.blocks.items():
            for attr in ("attention_residual_alpha", "feed_forward_residual_alpha"):
                if not _close(getattr(block, attr), multiplier):
                    self._fail(f"block {key}: {attr} {getattr(block, attr)} != {multiplier}")
            # Peri-norm RMSNorm blocks fold the multiplier into the branch norms' gains.
            for norm, add in (
                ("attention_norm", "_attention_add_alpha"),
                ("feed_forward_norm", "_feed_forward_add_alpha"),
            ):
                scale = getattr(block, norm).output_scale * getattr(block, add)
                if not _close(scale, multiplier):
                    self._fail(f"block {key}: {norm} scale x add alpha {scale} != {multiplier}")
        block = model.blocks["1"]
        self.records["residual"] = {
            "expected": multiplier,
            "attention_residual_alpha": block.attention_residual_alpha,
            "feed_forward_residual_alpha": block.feed_forward_residual_alpha,
            "attention_norm_output_scale": block.attention_norm.output_scale,
            "feed_forward_norm_output_scale": block.feed_forward_norm.output_scale,
            "attention_add_alpha": block._attention_add_alpha,
            "feed_forward_add_alpha": block._feed_forward_add_alpha,
            "num_branches": 2 * len(model.blocks),
        }
        init_stds = {}
        for name, p in model.named_parameters():
            info = self._expected(name, tuple(p.shape), completedp=True)
            if (
                info["role"] not in ("hidden matrix", "readout matrix", "output matrix")
                and name != "embeddings.weight"
            ):
                continue
            factor = 1.0
            if self.completedp and info["role"] != "input":
                r = info["fan_in"] / self.base_width
                factor = r**-0.5 if info["role"] == "hidden matrix" else 1 / r
            expected = INIT_STD * factor
            actual = p.detach().float().std().item()
            init_stds[name] = [actual, expected]
            # Truncation at +/-3 std lowers the realized std by about 1.3%.
            if not math.isclose(actual, expected, rel_tol=0.05):
                self._fail(f"{name}: init std {actual} != {expected}")
        self.records["init_stds"] = init_stds

    def pre_train(self):
        self.records.update(
            spec=self.spec,
            completedp=self.completedp,
            alpha=ALPHA,
            base_width=self.base_width,
            base_depth=self.base_depth,
            sde_ratio=self.sde_ratio,
            d_model=self.d_model,
            n_layers=self.n_layers,
            steps=[],
            coord={site: [] for site in SITES},
        )
        self._check_model()
        groups = self.optim.param_groups
        params: Dict[str, Any] = {}
        seen: Dict[str, int] = {}
        for index, group in enumerate(groups):
            for name, p in group["named_params"].items():
                seen[name] = seen.get(name, 0) + 1
                expected = self._expected(name, tuple(p.shape), completedp=self.completedp)
                fan_in = self._expected(name, tuple(p.shape), completedp=True)["fan_in"]
                actual = {
                    "lr": float(group.get("lr_multiplier", float(group["initial_lr"]) / LR)),
                    "eps": float(group.get("eps_multiplier", float(group["eps"]) / EPS)),
                    "weight_decay": float(group.get("weight_decay_multiplier", 1.0)),
                }
                for option in ("lr", "eps", "weight_decay"):
                    if not _close(actual[option], expected[option]):
                        self._fail(
                            f"{name}: {option} multiplier {actual[option]} != {expected[option]}"
                        )
                checks = (
                    ("initial_lr", float(group["initial_lr"]), LR * expected["lr"]),
                    ("eps", float(group["eps"]), EPS * expected["eps"]),
                    (
                        "weight_decay",
                        float(group["weight_decay"]),
                        expected["base_weight_decay"] * expected["weight_decay"],
                    ),
                    ("beta1", float(group["betas"][0]), expected["betas"][0]),
                    ("beta2", float(group["betas"][1]), expected["betas"][1]),
                )
                for option, value, target in checks:
                    if not _close(value, target):
                        self._fail(f"{name}: {option} {value} != {target}")
                params[name] = {
                    "group": index,
                    "shape": list(p.shape),
                    "fan_in": fan_in,
                    "role": param_role(name, self.attention_layers),
                    "kind": self._expected(name, tuple(p.shape), completedp=True)["role"],
                    **{f"{k}_multiplier": v for k, v in actual.items()},
                    "eps": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "betas": [float(b) for b in group["betas"]],
                }
        trainable = {
            n
            for m in self.trainer.train_module.model_parts
            for n, p in m.named_parameters()
            if p.requires_grad
        }
        for name in trainable - set(seen):
            self._fail(f"{name}: in no optimizer group")
        for name, count in seen.items():
            if count != 1:
                self._fail(f"{name}: in {count} optimizer groups")
        self.records["params"] = params
        self.records["num_groups"] = len(groups)
        self.records["groups"] = [
            {
                "num_params": len(group["named_params"]),
                "pg": group.get("pg"),
                "lr": float(group["initial_lr"]),
                "eps": float(group["eps"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": [float(b) for b in group["betas"]],
                **{
                    f"{k}_multiplier": float(group.get(f"{k}_multiplier", 1.0))
                    for k in ("lr", "eps", "weight_decay")
                },
            }
            for group in groups
        ]

    def pre_step(self, batch: Dict[str, Any]):
        if self.probe and self.trainer.global_step == 1:
            self._probe_ids = batch["input_ids"].clone()
            self._probe_initial = self._probe()

    def pre_optim_step(self):
        if self.trainer.global_step != 1:
            return
        # The optimizer's fp32 states are flat and may be sharded over the data-parallel ranks;
        # take the gradient shard the optimizer will use, as in `_copy_model_grads_to_main_grads`.
        optim = self.optim
        for group in optim.param_groups:
            for name, p in group["named_params"].items():
                main = optim.states[f"{name}.main"]
                self._snapshot[name] = main.to_local().detach().reshape(-1).clone()
                grad = p._main_grad_fp32.detach().float().reshape(-1)
                grad = optim.narrow_tensor(grad, main.device_mesh, main.placements)
                if optim.moe_mesh is not None and group.get("pg") == "ep_dp":
                    grad = grad / dist.get_world_size(optim.ep_mp_mesh.get_group())
                self._grads[name] = grad.clone()

    def _check_first_step(self):
        """Recompute AdamW's first update from the captured gradients."""
        grad_norm = self.optim.latest_grad_norm.float()
        clip = torch.clamp(1.0 / (grad_norm + 1e-6), max=1.0)
        discriminating = 0
        worst: Dict[str, float] = {"exp_avg": 0.0, "exp_avg_sq": 0.0, "weights": 0.0}
        for group in self.optim.param_groups:
            lr = float(group["lr"])
            wd = float(group["weight_decay"])
            eps = float(group["eps"])
            beta1, beta2 = (float(b) for b in group["betas"])
            for name in group["named_params"]:
                g = (self._grads.pop(name) * clip).double()
                old = self._snapshot.pop(name).double()

                def state(key: str) -> torch.Tensor:
                    return self.optim.states[f"{name}.{key}"].to_local().detach().reshape(-1)

                new = state("main").double()
                exp_avg = state("exp_avg").double()
                exp_avg_sq = state("exp_avg_sq").double()
                for key, observed, expected in (
                    ("exp_avg", exp_avg, (1 - beta1) * g),
                    ("exp_avg_sq", exp_avg_sq, (1 - beta2) * g * g),
                ):
                    error = (observed - expected).abs()
                    scale = expected.abs().max().clamp_min(1e-30)
                    rel = (error.max() / scale).item()
                    worst[key] = max(worst[key], rel)
                    if rel > 1e-5:
                        self._fail(f"step 1 {name}: {key} relative error {rel:.3g}")

                def predicted(eps_value: float) -> torch.Tensor:
                    return old * (1 - lr * wd) - lr * g / (g.abs() + eps_value)

                # fp32 rounding of the bias corrections (``1 - beta``) alone can move the update
                # by about 1e-6 relative.
                tolerance = (old.abs() + lr) * 2.0**-18
                excess = ((new - predicted(eps)).abs() / tolerance).max().item()
                worst["weights"] = max(worst["weights"], excess)
                if excess > 1.0:
                    self._fail(f"step 1 {name}: weights off by {excess:.3g}x the tolerance")
                replicated = self.optim.states[f"{name}.main"].placements[0].is_replicate()
                if not _close(eps, EPS) and (get_rank() == 0 or not replicated):
                    off = (new - predicted(EPS)).abs() > tolerance
                    discriminating += int(off.sum().item())
        device = self.trainer.device
        worst_t = torch.tensor([worst[k] for k in sorted(worst)], device=device)
        count_t = torch.tensor([discriminating], device=device, dtype=torch.float64)
        dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
        self.records["first_step"] = {
            "worst_relative_error": dict(zip(sorted(worst), worst_t.tolist())),
            "elements_rejecting_base_eps": int(count_t.item()),
        }

    def post_train_batch(self):
        step = self.trainer.global_step
        scheduler = self.trainer.train_module.scheduler
        assert self.trainer.max_steps is not None
        scheduled = float(scheduler.get_lr(LR, step, self.trainer.max_steps))
        group_lrs = [float(g["lr"]) for g in self.optim.param_groups]
        for index, group in enumerate(self.optim.param_groups):
            multiplier = float(group.get("lr_multiplier", 1.0))
            expected = multiplier * scheduled
            if not math.isclose(group_lrs[index], expected, rel_tol=LR_RTOL, abs_tol=1e-15):
                self._fail(f"step {step} group {index}: lr {group_lrs[index]} != {expected}")
        if step == 1:
            if float(self.optim.step_skipped):
                self._fail("step 1 was skipped")
            self._check_first_step()
        self.records["steps"].append({"step": step, "scheduled_lr": scheduled, "lrs": group_lrs})
        if not self.probe:
            return
        probe = self._probe()
        for site in SITES:
            change = (probe[site] - self._probe_initial[site]).pow(2).mean().sqrt().item()
            self.records["coord"][site].append(change)

    def post_train(self):
        gathered: List[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, self.failures)
        self.failures = [f for rank_failures in gathered for f in rank_failures][:200]
        self.records["failures"] = self.failures
        self.records["world_size"] = dist.get_world_size()
        if get_rank() != 0:
            return
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.out_path).write_text(json.dumps(self.records))
        status = "FAILED" if self.failures else "PASSED"
        print(f"COMPLETEDP_CHECK {status} {self.spec} {run_label(self.completedp)}", flush=True)
        for failure in self.failures:
            print(f"  {failure}", flush=True)

    def _probe(self) -> Dict[str, torch.Tensor]:
        train_module = self.trainer.train_module
        model = _unwrap(train_module.model)
        n_layers = len(model.blocks)
        modules = {
            "embedding": model.embeddings,
            "block 1": model.blocks["1"],
            "middle block": model.blocks[str(n_layers // 2)],
            "last block": model.blocks[str(n_layers - 1)],
        }
        captured: Dict[str, torch.Tensor] = {}

        def capture(site: str):
            def hook(module, inputs, output):
                del module, inputs
                if isinstance(output, tuple):
                    output = output[0]
                captured[site] = output.detach().float()

            return hook

        hooks = [module.register_forward_hook(capture(site)) for site, module in modules.items()]
        try:
            assert self._probe_ids is not None
            with torch.no_grad():
                batch = {"input_ids": self._probe_ids.to(self.trainer.device)}
                output = train_module.eval_batch(dict(batch), labels=get_labels(batch))
        finally:
            for hook in hooks:
                hook.remove()
        assert output is not None and output.logits is not None
        captured["logits"] = output.logits.detach().float()
        for m in train_module.model_parts:
            m.train()
        return captured


@dataclass
class ThroughputRecorder(Callback):
    """
    Records the speed monitor's throughput and MFU, and the optimizer-step time.
    """

    out_path: str
    completedp: bool
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    _events: List[Tuple[int, Any, Any]] = field(default_factory=list)

    def pre_train(self):
        optim = self.trainer.train_module.optim
        step = optim.step

        def timed_step(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = step(*args, **kwargs)
            end.record()
            self._events.append((self.trainer.global_step, start, end))
            return result

        optim.step = timed_step
        self.metrics["num_groups"] = [float(len(optim.param_groups))]

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        if step <= BENCH_WARMUP_STEPS:
            return
        for key in ("throughput/device/TPS", "throughput/device/MFU"):
            if key in metrics:
                self.metrics.setdefault(key, []).append(float(metrics[key]))

    def post_train(self):
        torch.cuda.synchronize()
        self.metrics["optim_step_ms"] = [
            start.elapsed_time(end)
            for step, start, end in self._events
            if step > BENCH_WARMUP_STEPS
        ]
        out = Path(self.out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        runs = json.loads(out.read_text()) if out.exists() else []
        run: Dict[str, Any] = {"completedp": self.completedp}
        run.update(self.metrics)
        runs.append(run)
        out.write_text(json.dumps(runs))


def _train(
    spec: str,
    *,
    completedp: bool,
    work_dir: Path,
    steps: int,
    rank_sequences: int,
    compile_model: bool,
    fixed_batch: bool,
    callback_name: str,
    callback: Callback,
    ep_degree: Optional[int] = None,
):
    seed_all(0)
    model_config = build_model_config(spec, completedp=completedp)
    model = model_config.build(init_device="meta")
    world_size = dist.get_world_size()
    global_batch_size = world_size * rank_sequences * SEQ_LEN
    train_module = OLMoDDPTrainModuleConfig(
        rank_microbatch_size=SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        optim=OLMoDDPOptimizerConfig(
            lr=LR,
            eps=EPS,
            weight_decay=WEIGHT_DECAY,
            betas=BETAS,
            group_overrides=[
                OptimGroupOverride(params=["*embeddings.weight"], opts={"weight_decay": 0.0}),
                OptimGroupOverride(
                    params=["*routed_experts.w_up_gate", "*routed_experts.w_down"], opts={}
                ),
            ],
            compile=True,
            dtype=DType.float32,
            sigma_factor=6,
            max_grad_norm=1.0,
            use_distributed=True,
            completedp=(
                optim_completedp_config(global_batch_size, spec_sde_ratio(spec))
                if completedp
                else None
            ),
        ),
        scheduler=WSD(warmup=5, decay=5, decay_fraction=None),
        compile_model=compile_model,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.ddp,
            reduce_grads_in_fp32=True,
            accumulate_grads_in_fp32=True,
            use_reduce_scatter=False,
        ),
        ep_config=(
            None if ep_degree is None else TransformerExpertParallelConfig(degree=ep_degree)
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
    ).build(model)

    source = RandomInstanceSourceConfig(
        tokenizer=TOKENIZER,
        sequence_length=SEQ_LEN,
        avg_document_length=1024,
        seed=0,
        num_instances=world_size * rank_sequences * (1 if fixed_batch else steps + 4),
    ).build(str(work_dir))
    data_loader = ComposableDataLoaderConfig(
        tokenizer=TOKENIZER,
        global_batch_size=global_batch_size,
        seed=0,
        work_dir=str(work_dir),
        display_source_visualization=False,
    ).build(source, dp_process_group=train_module.dp_process_group)
    trainer = (
        TrainerConfig(
            save_folder=str(work_dir / "save"),
            work_dir=str(work_dir),
            save_overwrite=True,
            no_checkpoints=True,
            no_evals=True,
            metrics_collect_interval=1,
            max_duration=Duration.steps(steps),
        )
        .with_callback(callback_name, callback)
        .build(train_module, data_loader)
    )
    trainer.fit()
    return model_config


def run_label(completedp: bool) -> str:
    if not completedp:
        return "sp"
    return "cp" if ALPHA == 1.0 else f"cpa{ALPHA:g}"


def run(spec: str, completedp: bool, out_dir: Path) -> int:
    prepare_training_environment()
    real_hero = spec in HERO_SIZES
    if not real_hero:
        assert dist.get_world_size() == 1, "run hero-like sizes on a single GPU"
    try:
        label = f"{spec}-{run_label(completedp)}"
        d_model, n_layers = parse_spec(spec)
        base_width, base_depth = spec_bases(spec)
        check = CompletedPCheck(
            spec=spec,
            completedp=completedp,
            out_path=str(out_dir / f"{label}.json"),
            d_model=d_model,
            n_layers=n_layers,
            attention_layers=tuple(range(7, n_layers, 8)),
            base_width=base_width,
            base_depth=base_depth,
            sde_ratio=spec_sde_ratio(spec) if completedp else 1.0,
            probe=not real_hero,
        )
        _train(
            spec,
            completedp=completedp,
            work_dir=out_dir / "work" / label,
            steps=STEPS,
            rank_sequences=RANK_SEQUENCES,
            compile_model=False,
            fixed_batch=not real_hero,
            callback_name="completedp_check",
            callback=check,
            ep_degree=dist.get_world_size() if spec == "medium" else None,
        )
    finally:
        teardown_training_environment()
    return len(check.failures)


def bench(spec: str, completedp: bool, out_dir: Path):
    prepare_training_environment()
    assert dist.get_world_size() == 1, "run on a single GPU"
    try:
        recorder = ThroughputRecorder(
            out_path=str(out_dir / f"bench-{spec}.json"), completedp=completedp
        )
        _train(
            spec,
            completedp=completedp,
            work_dir=out_dir / "work" / f"bench-{spec}-{'cp' if completedp else 'sp'}",
            steps=BENCH_STEPS,
            rank_sequences=BENCH_RANK_SEQUENCES,
            compile_model=True,
            fixed_batch=False,
            callback_name="throughput",
            callback=recorder,
        )
    finally:
        teardown_training_environment()


def log_slope(xs: List[int], values: List[float]) -> float:
    if len(xs) < 2:
        return math.nan
    lx = [math.log(x) for x in xs]
    ly = [math.log(v) for v in values]
    x_mean, y_mean = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(lx, ly)) / sum(
        (x - x_mean) ** 2 for x in lx
    )


# Table 1 per role, as (lr, eps, weight decay) multipliers of (r, m_N, m_L, alpha), where r is the
# matrix's own fan-in over base_width; and the init-std factor of r.
TABLE1 = {
    "hidden matrix": (
        ("m_L^(a-1)/r", lambda r, mn, ml, a: ml ** (a - 1) / r),
        ("m_L^-a/m_N", lambda r, mn, ml, a: ml**-a / mn),
        ("r", lambda r, mn, ml, a: r),
    ),
    "hidden vector": (
        ("m_L^(a-1)", lambda r, mn, ml, a: ml ** (a - 1)),
        ("m_L^-a/m_N", lambda r, mn, ml, a: ml**-a / mn),
        ("1", lambda r, mn, ml, a: 1.0),
    ),
    "head-shared vector": (
        ("m_L^(a-1)", lambda r, mn, ml, a: ml ** (a - 1)),
        ("m_L^-a", lambda r, mn, ml, a: ml**-a),
        ("1", lambda r, mn, ml, a: 1.0),
    ),
    "input": (
        ("1", lambda r, mn, ml, a: 1.0),
        ("1/m_N", lambda r, mn, ml, a: 1 / mn),
        ("1", lambda r, mn, ml, a: 1.0),
    ),
    "output matrix": (
        ("1/r", lambda r, mn, ml, a: 1 / r),
        ("1", lambda r, mn, ml, a: 1.0),
        ("r", lambda r, mn, ml, a: r),
    ),
    "output vector": (
        ("1", lambda r, mn, ml, a: 1.0),
        ("1", lambda r, mn, ml, a: 1.0),
        ("1", lambda r, mn, ml, a: 1.0),
    ),
    # In-block matrices with a fixed output size: output-weight width rules, hidden depth rules.
    "readout matrix": (
        ("m_L^(a-1)/r", lambda r, mn, ml, a: ml ** (a - 1) / r),
        ("m_L^-a", lambda r, mn, ml, a: ml**-a),
        ("r", lambda r, mn, ml, a: r),
    ),
}
TABLE1_INIT = {
    "hidden matrix": ("r^-1/2", lambda r: r**-0.5),
    "output matrix": ("1/r", lambda r: 1 / r),
    "readout matrix": ("1/r", lambda r: 1 / r),
    "input": ("1", lambda r: 1.0),
}


def _table1(kind: str, fan_in: Optional[int], result: Dict[str, Any]) -> Tuple[float, ...]:
    """The Table 1 lr, eps, weight decay and init-std multipliers of a parameter in a run."""
    r = 1.0 if fan_in is None else fan_in / result["base_width"]
    m_n = result["d_model"] / result["base_width"]
    m_l = result["n_layers"] / result["base_depth"]
    s = result["sde_ratio"] ** 0.5
    lr, eps, wd = (rule(r, m_n, m_l, result["alpha"]) for _, rule in TABLE1[kind])
    init = TABLE1_INIT[kind][1](r) if kind in TABLE1_INIT else math.nan
    return lr * s, eps / s, wd * s, init


def _short(role: str) -> str:
    return role.replace(".weight", "").replace("attention.", "").replace("_experts", "")


def result_attention_layers(result: Dict[str, Any]) -> Tuple[int, ...]:
    return tuple(range(7, result["n_layers"], 8))


def report_hero(out_dir: Path) -> int:
    """
    Print how CompletedP scaled every parameter from the small to the medium hero, next to the
    scaling Table 1 predicts.
    """
    runs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for size in HERO_SIZES:
        for path in sorted(out_dir.glob(f"{size}-*.json")):
            runs[(size, path.stem.removeprefix(f"{size}-"))] = json.loads(path.read_text())
    if not runs:
        return 0
    failed = 0
    print("\nProduction heroes, hard checks:")
    for (size, label), result in sorted(runs.items()):
        first = result.get("first_step", {})
        worst = first.get("worst_relative_error", {})
        status = "FAILED" if result["failures"] else "passed"
        failed += bool(result["failures"])
        res = result["residual"]
        print(
            f"  {size:6s} {label:7s}: {status}, {result['world_size']} ranks, "
            f"{result['num_groups']} groups, {len(result['params'])} tensors; residual "
            f"multiplier {res['attention_residual_alpha']:.4g} (expected {res['expected']:.4g}) "
            f"on {res['num_branches']} branches, folded into the norms: "
            f"{res['attention_norm_output_scale']:.4g} x add {res['attention_add_alpha']:.4g}; "
            f"step 1 worst exp_avg {worst.get('exp_avg', 0):.2g}, exp_avg_sq "
            f"{worst.get('exp_avg_sq', 0):.2g}, weights {worst.get('weights', 0):.2g}x tol; "
            f"elements rejecting the base eps {first.get('elements_rejecting_base_eps', 0):,}"
        )
        for failure in result["failures"][:10]:
            print(f"      {failure}")

    for (size, label), result in sorted(runs.items()):
        print(f"\nOptimizer groups, {size} {label} (lr / eps / weight decay multipliers):")
        for index, group in enumerate(result["groups"]):
            print(
                f"  {index:2d} {group['pg'] or '':6s} {group['num_params']:4d} tensors  "
                f"lr x{group['lr_multiplier']:.4g}  eps x{group['eps_multiplier']:.4g}  "
                f"wd x{group['weight_decay_multiplier']:.4g} (= {group['weight_decay']:.4g})  "
                f"betas {tuple(round(b, 6) for b in group['betas'])}"
            )

    base = runs.get(("small", "cp"))
    targets = [k for k in sorted(runs) if k[0] == "medium" and k[1] != "sp"]
    if base is None or not targets:
        return failed

    def per_role(result):
        attention_layers = result_attention_layers(result)
        stds: Dict[str, List[float]] = {}
        for name, (actual, _) in result.get("init_stds", {}).items():
            stds.setdefault(param_role(name, attention_layers), []).append(actual)
        roles: Dict[str, Dict[str, Any]] = {}
        for name, info in result["params"].items():
            role = param_role(name.removeprefix("module."), attention_layers)
            if role in roles:
                continue
            role_std = stds.get(role)
            roles[role] = {
                "kind": info["kind"],
                "fan_in": info["fan_in"],
                "mults": (
                    info["lr_multiplier"],
                    info["eps_multiplier"],
                    info["weight_decay_multiplier"],
                ),
                "std": statistics.mean(role_std) if role_std else math.nan,
            }
        return roles

    small = per_role(base)
    mediums = {label: per_role(runs[("medium", label)]) for _, label in targets}
    categories: Dict[Tuple[Any, ...], List[str]] = {}
    for role, info in small.items():
        if not all(role in m for m in mediums.values()):
            continue
        signature: Tuple[Any, ...] = (
            info["kind"],
            info["fan_in"],
            mediums[targets[0][1]][role]["fan_in"],
            info["mults"],
            tuple(m[role]["mults"] for m in mediums.values()),
        )
        categories.setdefault(signature, []).append(role)

    def cell(m: Tuple[float, ...]) -> str:
        return "/".join(f"{v:.4g}" for v in m)

    def std_cell(v: float) -> str:
        return "-" if math.isnan(v) else f"{v:.4f}"

    def representative(roles: List[str]) -> str:
        return next((r for r in roles if not math.isnan(small[r]["std"])), roles[0])

    print("\nPer-role multipliers on the base values (lr/eps/wd) and measured init std:")
    header = f"  {'kind':19s} {'fan-in S->M':13s} {'small':>22s}"
    for _, label in targets:
        header += f" {'medium ' + label:>28s}"
    print(header + "  roles")
    for key, roles in categories.items():
        kind, fan_s, fan_m, mult_s, mult_ms = key
        rep_role = representative(roles)
        line = (
            f"  {kind:19s} {str(fan_s) + '->' + str(fan_m):13s} "
            f"{cell(mult_s) + ' ' + std_cell(small[rep_role]['std']):>22s}"
        )
        for (_, label), mult_m in zip(targets, mult_ms):
            line += f" {cell(mult_m) + ' ' + std_cell(mediums[label][rep_role]['std']):>28s}"
        shown = ", ".join(_short(r) for r in roles[:4])
        if len(roles) > 4:
            shown += f" (+{len(roles) - 4})"
        print(line + "  " + shown)

    for _, label in targets:
        target = runs[("medium", label)]
        print(
            f"\nScaling small -> medium {label} (alpha={target['alpha']:g}): measured ratio vs "
            "Table 1 (m_N = m_L = 1.5; r = fan-in / 1024):"
        )
        print(
            f"  {'kind':19s} {'fan-in S->M':13s} {'lr rule':12s} {'lr meas/exp':>15s} "
            f"{'eps rule':11s} {'eps meas/exp':>15s} {'wd meas/exp':>13s} "
            f"{'init std meas/exp':>18s}  ok"
        )
        for key, roles in categories.items():
            kind, fan_s, fan_m, _, _ = key
            role = representative(roles)
            exp_s = _table1(kind, fan_s, base)
            exp_m = _table1(kind, fan_m, target)
            meas_s = (*small[role]["mults"], small[role]["std"])
            meas_m = (*mediums[label][role]["mults"], mediums[label][role]["std"])
            exp_ratio = [m / s_ for m, s_ in zip(exp_m, exp_s)]
            meas_ratio = [m / s_ for m, s_ in zip(meas_m, meas_s)]
            ok = all(
                math.isclose(a, b, rel_tol=1e-9) for a, b in zip(meas_ratio[:3], exp_ratio[:3])
            )
            if not math.isnan(exp_ratio[3]):
                ok = ok and math.isclose(meas_ratio[3], exp_ratio[3], rel_tol=0.05)
            failed += not ok
            init = "-" if math.isnan(exp_ratio[3]) else f"{meas_ratio[3]:.3f}/{exp_ratio[3]:.3f}"
            print(
                f"  {kind:19s} {str(fan_s) + '->' + str(fan_m):13s} {TABLE1[kind][0][0]:12s} "
                f"{meas_ratio[0]:.4f}/{exp_ratio[0]:.4f} {TABLE1[kind][1][0]:11s} "
                f"{meas_ratio[1]:.4f}/{exp_ratio[1]:.4f} {meas_ratio[2]:.4f}/{exp_ratio[2]:.4f} "
                f"{init:>18s}  {'yes' if ok else 'NO'}"
            )
    return failed


def report(out_dir: Path) -> int:
    failed = report_hero(out_dir)
    results = {}
    for path in sorted(out_dir.glob("d*-L*-*.json")):
        spec, label = path.stem.rsplit("-", 1)
        results[(spec, label)] = json.loads(path.read_text())
    if not results:
        return failed
    print("Hard checks:")
    for (spec, label), result in sorted(results.items()):
        first = result.get("first_step", {})
        worst = first.get("worst_relative_error", {})
        status = "FAILED" if result["failures"] else "passed"
        failed += bool(result["failures"])
        print(
            f"  {spec:10s} {label}: {status}, {result['num_groups']} groups, "
            f"{len(result['params'])} params; step 1 worst exp_avg {worst.get('exp_avg', 0):.2g}, "
            f"exp_avg_sq {worst.get('exp_avg_sq', 0):.2g}, weights "
            f"{worst.get('weights', 0):.2g}x tol; elements rejecting the base eps: "
            f"{first.get('elements_rejecting_base_eps', 0)}"
        )
        for failure in result["failures"][:10]:
            print(f"      {failure}")

    cp_specs = sorted({s for s, label in results if label == "cp"}, key=parse_spec)
    if cp_specs:
        print("\nCompletedP multipliers by role (lr / eps / weight decay):")
        roles: Dict[str, Dict[str, str]] = {}
        for spec in cp_specs:
            for info in results[(spec, "cp")]["params"].values():
                cell = (
                    f"{info['lr_multiplier']:.3g}/{info['eps_multiplier']:.3g}/"
                    f"{info['weight_decay_multiplier']:.3g}"
                )
                roles.setdefault(info["role"], {}).setdefault(spec, cell)
        print(f"  {'role':44s}" + "".join(f"{s:>20s}" for s in cp_specs))
        for role in sorted(roles):
            print(f"  {role:44s}" + "".join(f"{roles[role].get(s, '-'):>20s}" for s in cp_specs))

    for label in sorted({label for _, label in results}, key=lambda lab: (lab != "sp", lab)):
        name = "standard parameterization"
        if label != "sp":
            name = f"CompletedP (alpha={label.removeprefix('cp').removeprefix('a') or 1})"
        specs = sorted({s for s, lab in results if lab == label}, key=parse_spec)
        print(f"\nCoordinate check, {name}: RMS change after {STEPS} steps on a fixed batch")
        print(f"  {'site':14s}" + "".join(f"{s:>11s}" for s in specs) + "  width exp  depth exp")
        for site in SITES:
            values = {s: results[(s, label)]["coord"][site][-1] for s in specs}
            widths = [s for s in specs if parse_spec(s)[1] == BASE_DEPTH]
            depths = [s for s in specs if parse_spec(s)[0] == BASE_WIDTH]
            w_exp = log_slope([parse_spec(s)[0] for s in widths], [values[s] for s in widths])
            d_exp = log_slope([parse_spec(s)[1] for s in depths], [values[s] for s in depths])
            print(
                f"  {site:14s}"
                + "".join(f"{values[s]:11.4g}" for s in specs)
                + f"  {w_exp:+9.2f}  {d_exp:+9.2f}"
            )

    for path in sorted(out_dir.glob("bench-*.json")):
        runs = json.loads(path.read_text())
        print(f"\nThroughput A/B ({path.stem}), median over steps {BENCH_WARMUP_STEPS + 1}+:")
        for run_ in runs:
            label = "CompletedP" if run_["completedp"] else "standard  "
            tps = statistics.median(run_.get("throughput/device/TPS", [float("nan")]))
            mfu = statistics.median(run_.get("throughput/device/MFU", [float("nan")]))
            optim_ms = statistics.median(run_.get("optim_step_ms", [float("nan")]))
            print(
                f"  {label} groups={int(run_['num_groups'][0]):3d}  TPS={tps:10.1f}  "
                f"MFU={mfu:6.2f}  optimizer step={optim_ms:7.2f} ms"
            )
    return failed


def main():
    global ALPHA
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "bench"):
        p = sub.add_parser(command)
        p.add_argument("spec")
        p.add_argument("param", choices=("cp", "sp"))
        p.add_argument("--alpha", type=float, default=ALPHA, help="The CompletedP depth exponent.")
        p.add_argument("--out-dir", type=Path, default=Path("/tmp/completedp_hero_check"))
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--out-dir", type=Path, default=Path("/tmp/completedp_hero_check"))
    args = parser.parse_args()
    if args.command in ("run", "bench"):
        ALPHA = args.alpha
    if args.command == "run":
        sys.exit(1 if run(args.spec, args.param == "cp", args.out_dir) else 0)
    elif args.command == "bench":
        bench(args.spec, args.param == "cp", args.out_dir)
    else:
        sys.exit(1 if report(args.out_dir) else 0)


if __name__ == "__main__":
    main()
