"""
Check that muP scales the learning rate as intended on the hero architecture.

Trains the hero stack (KDA, gated full attention, latent MoE) for a few steps on random data, using
the real :class:`~olmo_core.train.Trainer`, :class:`OLMoDDPTrainModule` and the production
:class:`~olmo_core.optim.OLMoDDPOptimizerConfig` settings. Model sizes:

* ``d512``, ``d1024``, ``d1536``, ``d2048``: the hero stack with 8 layers and 64 experts (top-8).
  The number of heads is ``d_model / 128``, so the attention head space always equals ``d_model``.
* ``small``: the exact small hero (d_model 1024, 8 heads, 16 layers, 512 experts).
* ``medium``: the exact medium hero (d_model 1536, 16 heads, 24 layers, 512 experts). Its head
  space (2048) grows faster than ``d_model``. It runs with expert parallelism over all ranks on
  the synchronized all-to-all path, since it does not fit under plain DDP on 8 GPUs.

Each run asserts that:

1. every trainable parameter is in exactly one optimizer group whose multiplier equals
   ``base_width / fan_in`` for matrices and 1 otherwise (always 1 without muP), computed here from
   the parameter's name and layout;
2. every group's LR equals the scheduled LR times its multiplier at every step;
3. the fp32 master weights actually moved by that LR. AdamW applies
   ``W <- W * (1 - lr * wd) - lr * u``, and on the first step ``u = g / (|g| + eps)``, so
   ``max |u|`` must be just under 1 for every parameter.

It also records how much a fixed probe batch's activations move from their initial values (a
coordinate check). ``report`` prints the per-role tables and the width exponents of those changes.

Run one size on 8 GPUs, then report::

    PYTHONPATH=src torchrun --nproc-per-node=8 src/examples/mup/hero_lr_check.py run d512 mup
    PYTHONPATH=src python src/examples/mup/hero_lr_check.py report
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "olmo_ddp"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import olmoe3_small_medium_models as hero  # noqa: E402
from coord_check import width_exponent  # noqa: E402

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
from olmo_core.nn.transformer import InitMethod, OLMoDDPModelConfig  # noqa: E402
from olmo_core.optim import (  # noqa: E402
    MuPConfig,
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
SPECS = ("d512", "d1024", "d1536", "d2048", "small", "medium")
HERO_SIZES = ("small", "medium")
EP_SPECS = ("medium",)
HERO_LIKE_LAYERS = 8
HERO_LIKE_EXPERTS = 64
HERO_LIKE_TOP_K = 8

LR = 8e-4
BASE_WIDTH = 1024
WEIGHT_DECAY = 0.1
STEPS = 20
SEQ_LEN = 4096
RANK_SEQUENCES = 2
FIRST_STEP_U_RANGE = (0.99, 1.005)
LR_RTOL = 1e-6
SITES = ("embedding", "block 1", "middle block", "last block", "logits")
VECTOR_ROW = "vectors (norms, convs, biases)"


def build_model_config(spec: str, *, mup: bool) -> OLMoDDPModelConfig:
    """
    Build the hero model for ``spec``, with nanochat init under muP and std 0.02 otherwise.
    """
    config = hero.build_model_config(
        spec if spec in HERO_SIZES else "small", eos_token_id=TOKENIZER.eos_token_id
    )
    if spec not in HERO_SIZES:
        d_model = int(spec.removeprefix("d"))
        n_heads = d_model // hero.HEAD_DIM
        g = hero.Geometry(
            d_model=d_model,
            n_layers=HERO_LIKE_LAYERS,
            n_heads=n_heads,
            n_kv_heads=n_heads // 2,
            expected_active_params=0,
            expected_active_non_embedding_params=0,
            expected_total_params=0,
        )
        norm = hero._norm()
        emo = deepcopy(config.block.routed_experts_router.emo)
        emo.max_document_expert_pool = HERO_LIKE_EXPERTS
        emo.eval_document_expert_pool = HERO_LIKE_EXPERTS
        emo.validate_for_router(num_experts=HERO_LIKE_EXPERTS, top_k=HERO_LIKE_TOP_K)
        kda_block = hero._moe_block(g, norm, hero._kda(g), emo)
        attention_block = hero._moe_block(g, norm, hero._attention(g, norm), emo)
        for block in (kda_block, attention_block):
            block.routed_experts.num_experts = HERO_LIKE_EXPERTS
            block.routed_experts_router.num_experts = HERO_LIKE_EXPERTS
            block.routed_experts_router.top_k = HERO_LIKE_TOP_K
        config.d_model = d_model
        config.n_layers = HERO_LIKE_LAYERS
        config.embed_scale = math.sqrt(d_model)
        config.block = kda_block
        config.block_overrides = {
            0: hero._dense_first(g, norm),
            **{i: deepcopy(attention_block) for i in g.full_attention_layers},
        }
    for block in [config.block, *config.block_overrides.values()]:
        mixer = block.sequence_mixer
        if hasattr(mixer, "use_experimental_kernels"):
            mixer.use_experimental_kernels = False
        if hasattr(mixer, "qk_norm_per_head_gains"):
            mixer.qk_norm_per_head_gains = True
        if spec in EP_SPECS and getattr(block, "ep", None) is not None:
            block.ep.path = ExpertParallelPath.sync_1d
    config.init_method = InitMethod.nanochat if mup else InitMethod.normal
    config.validate()
    return config


def expected_fan_in(name: str, shape: Tuple[int, ...], d_model: int) -> Optional[int]:
    """
    The fan-in of a hidden matrix, or ``None`` for embeddings and vector-like parameters.
    Derived from the parameter's name and tensor layout only.
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


def param_role(name: str, attention_layers: Tuple[int, ...]) -> str:
    """
    A width-independent label: the parameter name without its block index, prefixed by the block
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


@dataclass
class MuPLRCheck(Callback):
    """
    Asserts the per-group learning rates and the resulting weight updates, and records the
    coordinate check. See the module docstring.
    """

    spec: str
    mup: bool
    out_path: str
    d_model: int
    attention_layers: Tuple[int, ...]
    failures: List[str] = field(default_factory=list)
    records: Dict[str, Any] = field(default_factory=dict)
    _snapshot: Dict[str, torch.Tensor] = field(default_factory=dict)
    _probe_initial: Dict[str, torch.Tensor] = field(default_factory=dict)
    _logged_lrs: Dict[int, Dict[int, float]] = field(default_factory=dict)

    @property
    def optim(self):
        return self.trainer.train_module.optim

    def _fail(self, message: str):
        if len(self.failures) < 200:
            self.failures.append(message)

    def pre_train(self):
        groups = self.optim.param_groups
        params = {}
        seen: Dict[str, int] = defaultdict(int)
        group_records = []
        for index, group in enumerate(groups):
            if "lr_multiplier" in group:
                multiplier = float(group["lr_multiplier"])
            else:
                if self.mup:
                    self._fail(f"group {index}: no 'lr_multiplier' key under muP")
                multiplier = float(group["initial_lr"]) / LR
            for name, p in group["named_params"].items():
                seen[name] += 1
                fan_in = expected_fan_in(name.removeprefix("module."), tuple(p.shape), self.d_model)
                expected = BASE_WIDTH / fan_in if (self.mup and fan_in is not None) else 1.0
                if not math.isclose(multiplier, expected, rel_tol=1e-12):
                    self._fail(f"{name}: group multiplier {multiplier} != expected {expected}")
                if not math.isclose(float(group["initial_lr"]), LR * multiplier, rel_tol=1e-12):
                    self._fail(f"{name}: initial_lr {group['initial_lr']} != {LR * multiplier}")
                params[name] = {
                    "group": index,
                    "shape": list(p.shape),
                    "fan_in": fan_in,
                    "role": param_role(name, self.attention_layers),
                }
            group_records.append(
                {
                    "multiplier": multiplier,
                    "weight_decay": float(group["weight_decay"]),
                    "num_params": len(group["named_params"]),
                }
            )
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
        self.records.update(
            spec=self.spec,
            mup=self.mup,
            d_model=self.d_model,
            base_lr=LR,
            base_width=BASE_WIDTH,
            groups=group_records,
            params=params,
            steps=[],
            coord={site: [] for site in SITES},
        )
        self._probe_initial = self._probe()

    def pre_optim_step(self):
        self._snapshot = {
            name: self.optim.states[f"{name}.main"].to_local().detach().clone()
            for group in self.optim.param_groups
            for name in group["named_params"]
        }

    def post_train_batch(self):
        step = self.trainer.global_step
        scheduler = self.trainer.train_module.scheduler
        assert self.trainer.max_steps is not None
        scheduled = float(scheduler.get_lr(LR, step, self.trainer.max_steps))
        skipped = float(self.optim.step_skipped)
        group_lrs = [float(g["lr"]) for g in self.optim.param_groups]

        names: List[str] = []
        stats_max, stats_sum = [], []
        for index, group in enumerate(self.optim.param_groups):
            lr = group_lrs[index]
            expected = self.records["groups"][index]["multiplier"] * scheduled
            if not math.isclose(lr, expected, rel_tol=LR_RTOL, abs_tol=1e-15):
                self._fail(f"step {step} group {index}: lr {lr} != expected {expected}")
            wd = float(group["weight_decay"])
            for name in group["named_params"]:
                old = self._snapshot.pop(name)
                new = self.optim.states[f"{name}.main"].to_local().detach()
                delta = new - old
                if skipped or lr == 0.0:
                    stats_max.append(delta.abs().max() if delta.numel() else delta.new_zeros(()))
                    stats_sum.append(torch.stack([delta.new_zeros(()), delta.new_zeros(())]))
                else:
                    u = -(delta + (lr * wd) * old) / lr
                    stats_max.append(u.abs().max() if u.numel() else u.new_zeros(()))
                    stats_sum.append(torch.stack([u.pow(2).sum(), u.new_tensor(float(u.numel()))]))
                names.append(name)
        maxes = torch.stack(stats_max)
        sums = torch.stack(stats_sum)
        dist.all_reduce(maxes, op=dist.ReduceOp.MAX)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        maxes_list = maxes.tolist()
        rms_list = (sums[:, 0] / sums[:, 1].clamp_min(1)).sqrt().tolist()

        param_stats = {}
        for name, max_u, rms_u in zip(names, maxes_list, rms_list):
            if skipped or group_lrs[self.records["params"][name]["group"]] == 0.0:
                if max_u != 0.0:
                    self._fail(f"step {step} {name}: moved by {max_u} on a skipped/zero-LR step")
                continue
            param_stats[name] = [max_u, rms_u]
            if step == 1 and not (FIRST_STEP_U_RANGE[0] <= max_u <= FIRST_STEP_U_RANGE[1]):
                self._fail(f"step 1 {name}: max|u| = {max_u:.6f} outside {FIRST_STEP_U_RANGE}")

        self.records["steps"].append(
            {
                "step": step,
                "scheduled_lr": scheduled,
                "group_lrs": group_lrs,
                "skipped": skipped,
                "param_stats": param_stats,
            }
        )
        probe = self._probe()
        for site in SITES:
            change = (probe[site] - self._probe_initial[site]).pow(2).mean().sqrt().item()
            self.records["coord"][site].append(change)

    def pre_log_metrics(self, step: int, metrics: Dict[str, float]):
        applied = {s["step"]: s["group_lrs"] for s in self.records["steps"]}.get(step)
        if applied is None:
            return
        for key, value in metrics.items():
            if match := re.fullmatch(r"optim/LR \(group (\d+)\)", key):
                index = int(match.group(1))
                self._logged_lrs.setdefault(step, {})[index] = float(value)
                if not math.isclose(float(value), applied[index], rel_tol=LR_RTOL):
                    self._fail(f"step {step} group {index}: logged LR {value} != {applied[index]}")

    def post_train(self):
        # The final step's metrics are logged after post_train().
        num_steps = len(self.records["steps"])
        if get_rank() == 0 and len(self._logged_lrs) < num_steps - 1:
            self._fail(f"logged LR metrics for only {len(self._logged_lrs)}/{num_steps} steps")
        self.records["failures"] = self.failures
        if get_rank() == 0:
            Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.out_path).write_text(json.dumps(self.records))
            status = "FAILED" if self.failures else "PASSED"
            print(f"MUP_LR_CHECK {status} {self.spec} {'mup' if self.mup else 'sp'}", flush=True)
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
            with torch.no_grad():
                batch = {"input_ids": self._probe_ids()}
                output = train_module.eval_batch(dict(batch), labels=get_labels(batch))
        finally:
            for hook in hooks:
                hook.remove()
        assert output is not None and output.logits is not None
        captured["logits"] = output.logits.detach().float()
        for m in train_module.model_parts:
            m.train()
        return captured

    def _probe_ids(self) -> torch.Tensor:
        generator = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, TOKENIZER.eos_token_id, (1, SEQ_LEN), generator=generator)
        ids[:, 1023::1024] = TOKENIZER.eos_token_id
        return ids.to(self.trainer.device)


def run(spec: str, mup: bool, out_dir: Path) -> int:
    """
    Train one model with :class:`MuPLRCheck` attached and return the number of failed checks
    seen by this rank.
    """
    prepare_training_environment()
    try:
        seed_all(0)
        name = f"{spec}-{'mup' if mup else 'sp'}"
        work_dir = out_dir / "work" / name
        model_config = build_model_config(spec, mup=mup)
        model = model_config.build(init_device="meta")
        train_module = OLMoDDPTrainModuleConfig(
            rank_microbatch_size=SEQ_LEN,
            max_sequence_length=SEQ_LEN,
            optim=OLMoDDPOptimizerConfig(
                lr=LR,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.95),
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0}),
                    OptimGroupOverride(
                        params=["*routed_experts.w_up_gate", "*routed_experts.w_down"], opts={}
                    ),
                ],
                compile=True,
                dtype=DType.float32,
                sigma_factor=6,
                max_grad_norm=1.0,
                use_distributed=True,
                mup=MuPConfig(base_width=BASE_WIDTH) if mup else None,
            ),
            scheduler=WSD(warmup=5, decay=5, decay_fraction=None),
            compile_model=False,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.ddp,
                reduce_grads_in_fp32=True,
                accumulate_grads_in_fp32=True,
                use_reduce_scatter=False,
            ),
            ep_config=(
                TransformerExpertParallelConfig(degree=dist.get_world_size())
                if spec in EP_SPECS
                else None
            ),
            z_loss_multiplier=1e-5,
            max_grad_norm=1.0,
        ).build(model)

        world_size = dist.get_world_size()
        source = RandomInstanceSourceConfig(
            tokenizer=TOKENIZER,
            sequence_length=SEQ_LEN,
            avg_document_length=1024,
            seed=0,
            num_instances=world_size * RANK_SEQUENCES * (STEPS + 4),
        ).build(str(work_dir))
        data_loader = ComposableDataLoaderConfig(
            tokenizer=TOKENIZER,
            global_batch_size=world_size * RANK_SEQUENCES * SEQ_LEN,
            seed=0,
            work_dir=str(work_dir),
            display_source_visualization=False,
        ).build(source, dp_process_group=train_module.dp_process_group)

        check = MuPLRCheck(
            spec=spec,
            mup=mup,
            out_path=str(out_dir / f"{name}.json"),
            d_model=model_config.d_model,
            attention_layers=tuple(range(7, model_config.n_layers, 8)),
        )
        trainer = (
            TrainerConfig(
                save_folder=str(work_dir / "save"),
                work_dir=str(work_dir),
                save_overwrite=True,
                no_checkpoints=True,
                no_evals=True,
                metrics_collect_interval=1,
                max_duration=Duration.steps(STEPS),
            )
            .with_callback("mup_lr_check", check)
            .build(train_module, data_loader)
        )
        trainer.fit()
    finally:
        teardown_training_environment()
    return len(check.failures)


def _load(out_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    results = {}
    for spec in SPECS:
        for param in ("mup", "sp"):
            path = out_dir / f"{spec}-{param}.json"
            if path.exists():
                results[(spec, param)] = json.loads(path.read_text())
    return results


def _rows(result: Dict[str, Any], value_fn) -> Dict[str, List[float]]:
    """
    Group a per-parameter value by role. Matrices and the embedding keep their own role; every
    vector-like parameter is pooled into the :data:`VECTOR_ROW`.
    """
    values: Dict[str, List[float]] = defaultdict(list)
    for name, info in result["params"].items():
        value = value_fn(name, info)
        if value is None:
            continue
        is_vector = info["fan_in"] is None and info["role"] != "embeddings.weight"
        values[VECTOR_ROW if is_vector else info["role"]].append(value)
    return values


def _multipliers(result: Dict[str, Any]) -> Dict[str, List[float]]:
    return _rows(result, lambda name, info: result["groups"][info["group"]]["multiplier"])


def _first_step_u(result: Dict[str, Any]) -> Dict[str, List[float]]:
    stats = result["steps"][0]["param_stats"]
    return _rows(result, lambda name, info: stats[name][0] if name in stats else None)


def _median_rms_u(result: Dict[str, Any]) -> Dict[str, List[float]]:
    per_param: Dict[str, List[float]] = defaultdict(list)
    for step in result["steps"][1:]:
        for name, (_, rms) in step["param_stats"].items():
            per_param[name].append(rms)
    return _rows(
        result,
        lambda name, info: statistics.median(per_param[name]) if per_param.get(name) else None,
    )


def _span(values: List[float], fmt: str) -> str:
    if not values:
        return "-"
    lo, hi = fmt.format(min(values)), fmt.format(max(values))
    return lo if lo == hi else f"{lo}-{hi}"


def _table(title: str, columns: List[str], rows: Dict[str, List[str]], cell_width: int = 12):
    print(f"\n{title}")
    width = max([len(r) for r in rows] + [10])
    print(f"{'':{width}s}" + "".join(f"{c:>{cell_width}s}" for c in columns))
    for role, cells in rows.items():
        print(f"{role:{width}s}" + "".join(f"{c:>{cell_width}s}" for c in cells))


def report(out_dir: Path) -> int:
    """
    Print the hard-check results, the per-role tables and the coordinate check for every run in
    ``out_dir``, and return a non-zero exit code if any hard check failed.
    """
    results = _load(out_dir)
    if not results:
        print(f"No results in {out_dir}")
        return 1
    hard_failures = 0
    print("Hard checks (group membership, multipliers, LR at every step, step-1 update):")
    for (spec, param), result in results.items():
        failures = result["failures"]
        hard_failures += len(failures)
        print(f"  {spec:6s} {param:3s}: {'PASSED' if not failures else 'FAILED'}")
        for failure in failures[:10]:
            print(f"      {failure}")

    widths = [s for s in SPECS if s not in HERO_SIZES and (s, "mup") in results]
    columns = widths + [s for s in HERO_SIZES if (s, "mup") in results]
    if columns:
        tables = {
            s: (
                _multipliers(results[(s, "mup")]),
                _first_step_u(results[(s, "mup")]),
                _median_rms_u(results[(s, "mup")]),
            )
            for s in columns
        }
        row_names = sorted(
            {row for tables_s in tables.values() for row in tables_s[0]},
            key=lambda row: (row == VECTOR_ROW, row),
        )
        lo, hi = FIRST_STEP_U_RANGE
        for index, title, fmt in (
            (0, "muP LR multiplier (group LR / scheduled LR)", "{:.4g}"),
            (1, f"muP step 1: max|u| per parameter (must be in [{lo}, {hi}])", "{:.4f}"),
            (2, "muP steps 2-19: median RMS(u) = RMS(update) / group LR", "{:.3f}"),
        ):
            rows = {
                row: [_span(tables[s][index].get(row, []), fmt) for s in columns]
                for row in row_names
            }
            _table(title, columns, rows, cell_width=15)

        worst = 0.0
        for s in columns:
            result = results[(s, "mup")]
            for step in result["steps"]:
                if step["scheduled_lr"] == 0.0:
                    continue
                for info in result["params"].values():
                    if info["fan_in"] is not None:
                        lr = step["group_lrs"][info["group"]]
                        ratio = lr * info["fan_in"] / BASE_WIDTH / step["scheduled_lr"]
                        worst = max(worst, abs(ratio - 1.0))
        print(
            "\nmax over sizes, steps and matrices of "
            f"|group LR * fan_in / base_width / scheduled LR - 1| = {worst:.2e}"
        )

    for param, label in (("sp", "standard parameterization"), ("mup", "muP")):
        specs = [s for s in SPECS if s not in HERO_SIZES and (s, param) in results]
        if len(specs) < 2:
            continue
        d_models = [results[(s, param)]["d_model"] for s in specs]
        steps = len(results[(specs[0], param)]["coord"]["logits"])
        rows = {}
        for site in SITES:
            changes = [results[(s, param)]["coord"][site][-1] for s in specs]
            rows[site] = [f"{v:.4g}" for v in changes] + [
                f"{width_exponent(d_models, changes):+.2f}"
            ]
        _table(
            f"{label}: per-coordinate RMS change of the probe activations after {steps} steps",
            specs + ["exponent"],
            rows,
        )

    if ("small", "mup") in results and ("d1024", "mup") in results:
        small = {r: sorted(set(v)) for r, v in _multipliers(results[("small", "mup")]).items()}
        like = {r: sorted(set(v)) for r, v in _multipliers(results[("d1024", "mup")]).items()}
        shared = sorted(set(small) & set(like))
        mismatched = [r for r in shared if small[r] != like[r]]
        print(
            f"\nreal small hero vs d1024 hero-like: {len(shared)} shared roles, "
            f"{len(mismatched)} with different multipliers"
        )
        for role in mismatched:
            print(f"  {role}: small {small[role]} vs d1024 {like[role]}")
        hard_failures += len(mismatched)

    if ("medium", "mup") in results and ("d1536", "mup") in results:
        # Same d_model, but the medium hero has 16 heads (head space 2048) instead of 12 (1536).
        fan_ins: Dict[str, Dict[str, int]] = {}
        for spec in ("medium", "d1536"):
            for info in results[(spec, "mup")]["params"].values():
                if info["fan_in"] is not None:
                    fan_ins.setdefault(info["role"], {})[spec] = info["fan_in"]
        medium = _multipliers(results[("medium", "mup")])
        like = _multipliers(results[("d1536", "mup")])
        print("\nmedium hero vs d1536 hero-like (same d_model, head space 2048 vs 1536):")
        print(f"  {'role':34s}{'fan_in':>16s}{'multiplier':>20s}")
        for role in sorted(set(medium) & set(like) - {VECTOR_ROW}):
            if sorted(set(medium[role])) != sorted(set(like[role])):
                fan_in = f"{fan_ins[role]['medium']} vs {fan_ins[role]['d1536']}"
                mult = f"{_span(medium[role], '{:.4g}')} vs {_span(like[role], '{:.4g}')}"
                print(f"  {role:34s}{fan_in:>16s}{mult:>20s}")

    print("\nALL HARD CHECKS PASSED" if hard_failures == 0 else f"\n{hard_failures} FAILURE(S)")
    return 0 if hard_failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("spec", choices=SPECS)
    run_parser.add_argument("param", choices=("mup", "sp"))
    run_parser.add_argument("--out-dir", type=Path, default=Path("/tmp/mup_hero_check"))
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--out-dir", type=Path, default=Path("/tmp/mup_hero_check"))
    args = parser.parse_args()
    if args.command == "run":
        sys.exit(1 if run(args.spec, args.param == "mup", args.out_dir) else 0)
    else:
        sys.exit(report(args.out_dir))


if __name__ == "__main__":
    main()
