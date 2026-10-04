"""100B MT at reference-scaled K8 with fresh stage state and exact in-stage resumes."""

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass

import olmoe3_adaptive_mt_plan as p
from olmoe3_lr_sweep_watch import atomic_json

from olmo_core.distributed.utils import get_rank
from olmo_core.nn.moe.v2.router import MoERouterV2
from olmo_core.optim.scheduler import CosWithWarmup

p.install()
import olmoe3_qkgain_train as adapter  # noqa: E402


@dataclass
class MTAudit(adapter.Audit):
    """Use qualified transfer/resume audits and require matching batches across arms."""

    def pre_train(self):
        super().pre_train()
        routers = [m for m in self.trainer.train_module.model.modules() if isinstance(m, MoERouterV2)]
        assert len(routers) == 15
        for router in routers:
            assert router.top_k == 8 and router.reference_top_k == 16
            assert router.restore_weight_scale and router.normalize_expert_weights == 1.0
            assert router.original_top_k is None and not router.use_recompute_cache
        if get_rank() == 0:
            atomic_json(p.find_run(self.run_id).root / "audit/routing.json",
                        dict(passed=True, top_k=8, reference_top_k=16, multiplier=16, routed_layers=15))

    def pre_step(self, batch):
        # Match the first actual global batch in each startup segment. The later
        # arm checks the earlier arm's hash, with one lock per step/rank.
        if self.first and self.step <= 5:
            folder = p.AUTO / "data/batches"
            folder.mkdir(parents=True, exist_ok=True)
            file = folder / f"step{self.step}-rank{get_rank()}.json"
            sha = hashlib.sha256(batch["input_ids"].cpu().numpy().tobytes()).hexdigest()
            with file.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                row = json.loads(file.read_text()) if file.exists() else dict(input_sha256=sha, runs=[])
                assert row["input_sha256"] == sha, "Global data order differs across MT arms"
                row["runs"] = sorted(set(row["runs"]) | {self.run_id})
                atomic_json(file, row)
        super().pre_step(batch)


def install_adapters(r):
    from olmoe3_adaptive_mt_data import components

    adapter.scheduler = lambda _: CosWithWarmup(warmup=p.WARMUP, alpha_f=0)
    adapter.data_components = components
    original_common, original_model, original_trainer = (
        adapter.common_components, adapter.model_config, adapter.trainer_config)

    def common(ctx, **kwargs):
        c = original_common(ctx, **kwargs)
        # Separate generated indices avoid cross-job cache writes. The shared
        # frozen mixture, sample seed and batch-hash gates determine the data.
        c.work_dir = str(p.ROOT / "data-work" / r.schedule)
        return c

    def model(common):
        c = original_model(common)
        for block in [c.block, *c.block_overrides.values()]:
            router = getattr(block, "routed_experts_router", None)
            if router is not None:
                assert router.num_experts == 512 and router.top_k == 16 and router.emo is None
                router.top_k, router.reference_top_k = 8, 16
        c.validate()
        return c

    def trainer(common):
        c = original_trainer(common)
        c.callbacks["qkgain_audit"] = MTAudit(run_id=r.run_id)
        c.callbacks["checkpointer"].pre_train_checkpoint = adapter.source_for(r) == r.source
        c.callbacks["wandb"].project = "adaptive-compute"
        c.callbacks["wandb"].tags += [r.schedule, "reference-top16", "100B", "32g"]
        if int(os.environ["QKGAIN_STOP"]) <= 4:
            c.metrics_collect_interval, c.no_evals = 1, True
        return c

    adapter.common_components, adapter.model_config, adapter.trainer_config = common, model, trainer


def train():
    r = adapter.current()
    install_adapters(r)
    adapter.hero.qualified.apply_policy()
    adapter.main(config_builder=adapter.builder(r))


def validate(schedule):
    """Build fresh and resumed configs without loading weights or resolving data."""
    from olmo_core.internal.experiment import CliContext, SubCmd

    p.self_test()
    r = p.run(schedule)
    install_adapters(r)
    adapter.hero.qualified.apply_policy()
    for start, stop, source in [(0, 2, r.source), (2, 4, r.root / "step2"), (4, p.END, r.root / "step4")]:
        os.environ.update(QKGAIN_RUN=r.run_id, QKGAIN_START=str(start), QKGAIN_STOP=str(stop), QKGAIN_LOAD=str(source))
        c = adapter.builder(r)(CliContext(p.SCRIPT, SubCmd.dry_run, r.run_id, "ai2/holmes", [])).merge([])
        fresh = start == 0
        assert c.trainer.load_optim_state == c.trainer.load_trainer_state == (not fresh)
        assert c.train_module.reset_optimizer_states_on_load == fresh
        assert c.trainer.load_path == str(source) and c.trainer.max_duration.value == p.END
        assert c.trainer.hard_stop.value == stop
        assert c.train_module.scheduler.get_lr(r.lr, p.WARMUP, p.END) == r.lr
        assert c.train_module.scheduler.get_lr(r.lr, p.END, p.END) == 0
        assert c.data_loader.seed == p.SEED and c.data_loader.global_batch_size == p.BATCH
        assert c.dataset.source_mixture_config.requested_tokens == p.REQUESTED_TOKENS
        assert c.dataset.source_mixture_config.seed == p.SEED
        assert c.train_module.rank_microbatch_size == 32768 and c.dataset.sequence_length == 8192
        assert c.init_seed == 12536 and c.train_module.ep_config is None and c.train_module.pp_config is None
        assert c.trainer.callbacks["checkpointer"].fixed_steps == r.saves
        atomic_json(p.AUTO / "configs" / f"{schedule}-{start}.json", c.as_dict(json_safe=True))
    print("ADAPTIVE_MT_CONFIG_PASSED", schedule, flush=True)
