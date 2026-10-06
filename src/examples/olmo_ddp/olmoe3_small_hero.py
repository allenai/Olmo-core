"""Local-Dolma, qualified optimized small hero pair with EMO as the only model switch."""

import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.update(
    OLMOE3_INTEGRATION_ARM="optimized",
    OLMOE3_INTEGRATION_POLICY="core-docpool-top16-wgrad-rs",
    OLMOE3_INTEGRATION_BASELINE="optimized100b",
    OLMOE3_INTEGRATION_COMMUNICATION="none",
    OLMOE3_INTEGRATION_SMOKE="0",
)

import olmoe3_small_integration as qualified
import torch
import torch.distributed as dist
from olmoe3_lr_sweep_watch import atomic_json
from olmoe3_small_hero_plan import (
    BATCH,
    CAMPAIGN,
    CONTROL,
    DATA_ROOT,
    DOLMA_MOUNT,
    FINAL_STEPS,
    GPUS,
    INITIAL_STOP,
    LR,
    MOUNT,
    WARMUP,
    WORKSPACE,
    disk_action,
    find_run,
    fixed_checkpoint_steps,
    runs,
    validate_plan,
)

from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.internal.experiment import CliContext, SubCmd, build_config, main
from olmo_core.launch.beaker import BeakerWekaBucket
from olmo_core.optim.scheduler import ConstantWithWarmup
from olmo_core.train import Duration
from olmo_core.train.callbacks import Callback
from olmo_core.train.callbacks.checkpoint_ready_notifier import (
    CheckpointReadyNotifierCallback,
)
from olmo_core.train.common import LoadStrategy

qualified.SYSTEM = replace(qualified.SYSTEM, num_nodes=GPUS // 8)


def state_sample(trainer):
    """Sample live BF16 weights, FP32 optimizer shards, buffers and skip-step history."""
    tm = trainer.train_module
    tensors = dict(tm.optim.states)
    tensors.update(tm._persistent_model_buffer_state_dict())
    tensors.update((f"model_param/{n}", p) for n, p in tm.model.named_parameters())
    hashes = {}
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        tensor = tensor.detach().reshape(-1)
        assert tensor.numel(), name
        count = min(128, tensor.numel())
        offsets = torch.tensor(
            [i * (tensor.numel() - 1) // max(1, count - 1) for i in range(count)],
            device=tensor.device,
        )
        data = tensor[offsets].contiguous().view(torch.uint8).cpu().numpy().tobytes()
        hashes[name] = [tensor.numel(), str(tensor.dtype), hashlib.sha256(data).hexdigest()]
    return {
        "step": trainer.global_step,
        "tokens": trainer.global_train_tokens_seen,
        "rank": get_rank(),
        "gpus": get_world_size(),
        "tensors": hashes,
        "loss_history": [float(v.item()) for v in tm.optim._losses],
        "norm_history": [float(v.item()) for v in tm.optim._grad_norms],
    }


@dataclass
class HeroAudit(qualified.IntegrationAudit):
    """Validate identity/resume, monitor finite metrics, and guard synchronous saves."""

    priority: ClassVar[int] = 10
    run_id: str = ""

    def post_checkpoint_loaded(self, path):
        source = json.loads((Path(path) / "resume_audit" / "rank0.json").read_text())
        if source["gpus"] != get_world_size():
            from olmoe3_hero_reshard_audit import verify_doubled_dp

            proof = verify_doubled_dp(self.trainer, path, state_sample(self.trainer))
            atomic_json(
                Path(self.output_dir) / f"restore-step{self.step}-rank{get_rank()}.json",
                {"source": str(path), "step": self.step, "sampled_state_exact": True, **proof},
            )
            return
        expected = Path(path) / "resume_audit" / f"rank{get_rank()}.json"
        # The audit is part of the immutable uploaded checkpoint, so HF rehydration
        # does not depend on a side file that was only present on the original Weka mount.
        saved = json.loads(expected.read_text())
        actual = state_sample(self.trainer)
        if saved != actual:
            changed = [key for key in actual if saved.get(key) != actual[key]]
            raise RuntimeError(f"Resume state samples changed: {expected}, fields={changed}")
        atomic_json(
            Path(self.output_dir) / f"restore-step{self.step}-rank{get_rank()}.json",
            {"source": str(path), "step": self.step, "sampled_state_exact": True},
        )

    def pre_train(self):
        r = find_run(self.run_id)
        reg = json.loads((CONTROL / "registrations" / f"{r.run_id}.json").read_text())
        assert reg["run_id"] == reg["lineage_id"] == r.run_id
        assert reg["checkpoint_root"] == str(r.root) == str(self.trainer.save_folder)
        assert reg["bucket_id"] == r.bucket and reg["remote_prefix"] == r.prefix
        assert reg["enabled"] and reg["deletion_mode"] == "apply"
        assert reg["min_local_checkpoints"] >= 2 and reg["delete_grace_seconds"] >= 3600
        assert get_world_size() == GPUS
        assert MOUNT.is_mount() and DOLMA_MOUNT.is_mount()
        assert self.trainer.global_train_tokens_seen == self.step * BATCH
        # The loader counts only the current epoch; 14T children resume in epoch 2.
        loader = self.trainer.data_loader
        prior_batches = (self.trainer.epoch - 1) * loader.total_batches
        assert loader.tokens_processed == loader.batches_processed * BATCH
        assert prior_batches + loader.batches_processed == self.step
        if r.smoke:
            assert self.step == int(os.environ["OLMO35_HERO_EXPECTED_START"])
        else:
            assert self.step >= int(os.environ.get("OLMO35_HERO_EXPECTED_START", "0"))
        assert self.step <= int(os.environ.get("OLMO35_HERO_STOP", INITIAL_STOP))
        if (r.root / "STORAGE_PAUSED.json").exists():
            raise RuntimeError("Storage pause is latched; operator must explicitly approve resume")
        output = Path(self.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self._first_batch = True
        if self.step == 0:
            digest = hashlib.sha256()
            for i, (name, p) in enumerate(self.trainer.train_module.model.named_parameters()):
                if i % get_world_size() == get_rank():
                    digest.update(name.encode())
                    digest.update(
                        p.detach()
                        .contiguous()
                        .reshape(-1)
                        .view(torch.uint8)
                        .cpu()
                        .numpy()
                        .tobytes()
                    )
            hashes = [None] * get_world_size()
            dist.all_gather_object(hashes, digest.hexdigest(), group=self.trainer.bookkeeping_pg)
            if get_rank() == 0:
                qualified.write_or_verify(
                    output / "initial-weights-sha256.json", json.dumps(hashes)
                )
        tm = self.trainer.train_module
        original_save = tm.save_state_dict_direct

        def guarded_save(directory, **kwargs):
            before = state_sample(self.trainer)
            original_save(directory, **kwargs)
            after = state_sample(self.trainer)
            if before != after:
                raise RuntimeError("Synchronous save mutated live model/optimizer state")
            # Still inside Checkpointer's stepN-tmp transaction, BEFORE .metadata.json
            # marks completion. The uploader inventories these files with the payload.
            atomic_json(Path(directory).parent / "resume_audit" / f"rank{get_rank()}.json", after)

        tm.save_state_dict_direct = guarded_save
        if get_rank() == 0:
            record = {
                **r.as_dict(),
                "start_step": self.step,
                "tokens": self.trainer.global_train_tokens_seen,
                "source_commit": os.environ.get("GIT_REF"),
                "settings": qualified.SETTINGS,
                "registration": reg,
            }
            atomic_json(
                output / f"session-{os.environ.get('BEAKER_JOB_ID')}-{self.step}.json", record
            )
            print("HERO_START", json.dumps(record), flush=True)


@dataclass
class StorageGuard(Callback):
    """Check shared free space every 25 steps and latch a graceful checkpointed stop."""

    run_id: str = ""

    def _check(self):
        if get_rank() != 0:
            return
        r = find_run(self.run_id)
        fs = os.statvfs(MOUNT)
        free = fs.f_bavail * fs.f_frsize
        result = {"free_bytes": free, "action": disk_action(free), "step": self.step}
        if result["action"] != "ok" or self.step % 100 == 0:
            print("HERO_STORAGE", json.dumps(result), flush=True)
        if result["action"] == "stop":
            atomic_json(r.root / "STORAGE_PAUSED.json", result)
            # Let the trainer's existing cancellation protocol synchronize this.
            # A new foreground collective on the async bookkeeping group could race
            # its metric reductions. The reserve covers the <=25-step propagation delay.
            self.trainer.cancel_run("Dedicated checkpoint mount below 5 TB free")

    def pre_train(self):
        self._check()

    def post_step(self):
        if self.step % 25 == 0:
            self._check()


@dataclass
class CompletionAudit(Callback):
    """Record expected completion or a deliberate storage pause after final checkpointing."""

    priority: ClassVar[int] = -20
    run_id: str = ""

    def post_train(self):
        r = find_run(self.run_id)
        if get_rank() == 0:
            atomic_json(
                r.root / "audit" / f"complete-step{self.step}.json",
                {
                    "step": self.step,
                    "tokens": self.trainer.global_train_tokens_seen,
                    "storage_paused": (r.root / "STORAGE_PAUSED.json").exists(),
                    "requested_stop": int(os.environ.get("OLMO35_HERO_STOP", INITIAL_STOP)),
                },
            )


def common_components(cli_context, **kwargs):
    """Mount only the dedicated dataset and checkpoint volumes."""
    r = find_run(cli_context.run_name)
    common = qualified.common_components(cli_context, **kwargs)
    common.save_folder = str(r.root)
    if common.launch:
        common.launch.workspace = WORKSPACE
        common.launch.weka_buckets.append(BeakerWekaBucket("dolma-3p5", str(DOLMA_MOUNT)))
        common.launch.cmd = [
            "python",
            "src/examples/olmo_ddp/olmoe3_small_hero_node.py",
            r.run_id,
            cli_context.cluster,
        ]
    return common


def data_components(common):
    """Change only the data transport; preserve manifest order, filters and seeds."""
    data = qualified.base.build_data_components(common)
    data.dataset.mix_base_dir = str(DATA_ROOT)
    return data


def model_config(common):
    """Both arms retain all qualified optimizations and per-head QK norm gains."""
    r = find_run(common.run_name)
    model = qualified.model_config(common)
    for block in [model.block, *model.block_overrides.values()]:
        router = getattr(block, "routed_experts_router", None)
        if router is not None and not r.emo:
            router.emo = None
    return model


def train_module_config(common):
    """The stable WSD trunk does not decay at the initial ~3T stopping point."""
    config = qualified.train_module_config(common)
    config.optim.lr = LR
    config.scheduler = ConstantWithWarmup(warmup=WARMUP)
    return config


def trainer_config(common):
    """Automatic 100->250->500 cadence; immutable full-state synchronous checkpoints."""
    r = find_run(common.run_name)
    config = qualified.trainer_config(common)
    config.callbacks.pop("integration_audit")
    stop = int(os.environ.get("OLMO35_HERO_STOP", "2" if r.smoke else str(INITIAL_STOP)))
    assert 0 < stop <= (4 if r.smoke else FINAL_STEPS)
    if not r.smoke and stop != INITIAL_STOP:
        assert os.environ.get("OLMO35_HERO_ALLOW_CONTINUATION") == "1"
    config.max_duration = Duration.steps(FINAL_STEPS)
    config.hard_stop = Duration.steps(stop)
    config.load_strategy = (
        LoadStrategy.always
        if os.environ.get("OLMO35_HERO_EXPECTED_START", "0") != "0"
        else LoadStrategy.if_available
    )
    cp = config.callbacks["checkpointer"]
    cp.save_interval = 2 if r.smoke else 500
    cp.fixed_steps = [] if r.smoke else fixed_checkpoint_steps()
    config.add_callback("hero_audit", HeroAudit(output_dir=str(r.root / "audit"), run_id=r.run_id))
    config.add_callback("hero_storage", StorageGuard(run_id=r.run_id))
    config.add_callback("hero_complete", CompletionAudit(run_id=r.run_id))
    config.add_callback(
        "checkpoint_ready",
        CheckpointReadyNotifierCallback(
            inbox_dir=str(CONTROL / "inbox"), run_id=r.run_id, lineage_id=r.run_id
        ),
    )
    if r.smoke:
        config.callbacks["lm_evaluator"].eval_interval = 2
        config.callbacks["lm_evaluator"].eval_duration = Duration.steps(2)
    wb = config.callbacks["wandb"]
    wb.project = "olmo3p5-hero"
    wb.group = CAMPAIGN + ("-smoke" if r.smoke else "")
    wb.tags = [
        r.arm,
        f"small-{GPUS}g",
        "16mi",
        "mb4",
        f"ga{BATCH // (GPUS * 4 * 8192)}",
        "qknorm-pr855",
        "qualified-optimized100b",
        "local-dolma3p5",
        "bf16",
        "wsd-trunk",
        "sync-checkpoints",
        "3t-stop-14t-continuable",
    ]
    wb.notes = json.dumps(r.as_dict())
    return config


def config_builder():
    """Build the exact qualified topology for both production and resume smokes."""
    return partial(
        build_config,
        global_batch_size=BATCH,
        max_sequence_length=8192,
        num_nodes=GPUS // 8,
        common_config_builder=common_components,
        data_config_builder=data_components,
        model_config_builder=model_config,
        train_module_config_builder=train_module_config,
        trainer_config_builder=trainer_config,
        beaker_image=qualified.base.BEAKER_IMAGE,
        beaker_workspace=WORKSPACE,
        include_default_evals=False,
        num_execution_units=1,
    )


def validate():
    """Build real configs in the production image and compare the two arms recursively."""
    validate_plan()
    models = []
    for r in runs() + runs(True):
        c = config_builder()(CliContext(__file__, SubCmd.dry_run, r.run_id, "ai2/holmes", []))
        c.as_dict(json_safe=True)
        tm, tr = c.train_module, c.trainer
        assert c.model.num_active_params == 794_233_472 and c.model.num_params == 12_496_341_632
        assert c.data_loader.global_batch_size == BATCH and c.init_seed == 12536
        assert c.dataset.mix_base_dir == str(DATA_ROOT)
        assert tm.rank_microbatch_size == 4 * 8192 and tm.dp_config.use_reduce_scatter
        assert tm.ac_config is None and tm.float8_config is None
        assert tm.ep_config is None and tm.pp_config is None
        assert tr.load_optim_state and tr.load_trainer_state and not tr.save_overwrite
        cp = tr.callbacks["checkpointer"]
        assert not cp.save_async and cp.max_checkpoints is None and cp.pre_train_checkpoint is None
        assert cp.fixed_steps == ([] if r.smoke else fixed_checkpoint_steps())
        assert cp.save_interval == (2 if r.smoke else 500)
        assert tm.scheduler.get_lr(LR, WARMUP, FINAL_STEPS) == LR
        assert tm.scheduler.get_lr(LR, INITIAL_STOP, FINAL_STEPS) == LR
        assert tm.scheduler.get_lr(LR, FINAL_STEPS, FINAL_STEPS) == LR
        model = c.model.as_dict(json_safe=True)
        routers = []

        def strip_emo(obj, routers=routers):
            if isinstance(obj, dict):
                if "emo" in obj:
                    routers.append(obj.pop("emo"))
                for value in obj.values():
                    strip_emo(value)
            elif isinstance(obj, list):
                for value in obj:
                    strip_emo(value)

        strip_emo(model)
        assert routers and all((v is not None) == r.emo for v in routers)
        models.append(model)
        print("HERO_CONFIG_VALIDATED", json.dumps(r.as_dict()), flush=True)
    assert all(m == models[0] for m in models)
    print("HERO_PAIR_DIFF_ONLY_EMO", flush=True)


if __name__ == "__main__":
    qualified.apply_policy()
    if sys.argv[1:] == ["--validate-only"]:
        validate()
    else:
        main(config_builder=config_builder())
