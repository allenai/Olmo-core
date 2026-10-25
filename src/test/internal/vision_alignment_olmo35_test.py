"""
OLMo 3.5 support in the alignment recipe: with ``recipe.text_config`` every text-side setting is
inherited from the text team's resolved mid-training config and the alignment phase differs from
it exactly by :data:`~olmo_core.internal.vision_alignment.MULTIMODAL_OVERRIDES`; without it the
LM config comes from the checkpoint through the legacy normalizer with EMO cleared.
"""

import dataclasses
import fnmatch
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import vision_alignment
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_alignment import (
    MULTIMODAL_OVERRIDES,
    VisionAlignmentExperimentConfig,
)
from olmo_core.nn.attention.kda import KimiDeltaAttentionConfig
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.train import TrainerConfig
from olmo_core.train.train_module.transformer.config import OLMoDDPTrainModuleConfig

FIXTURE = Path(__file__).parent.parent / "fixtures" / "olmo35_text_midtraining_config.json"


@pytest.fixture
def text_config() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def hero_checkpoint(alignment_recipe, text_config):
    """The alignment fixture's pretraining checkpoint rewritten as the OLMo 3.5 8T checkpoint's
    config.json: the text LM with the legacy ``use_cute_kernel`` key and EMO routing."""
    saved = json.loads((alignment_recipe.base / "config.json").read_text())
    model = json.loads(json.dumps(text_config["model"]))
    for block in [model["block"], *model["block_overrides"].values()]:
        mixer = block["sequence_mixer"]
        if "use_experimental_kernels" in mixer:
            mixer["use_cute_kernel"] = mixer.pop("use_experimental_kernels")
        router = block.get("routed_experts_router")
        if router is not None:
            router["emo"] = {"pool_size": 8}
    saved["model"] = model
    (alignment_recipe.base / "config.json").write_text(json.dumps(saved))
    return alignment_recipe.base


def _flatten(value, prefix=""):
    out = {}
    if isinstance(value, dict) and value:
        for key, item in value.items():
            out.update(_flatten(item, f"{prefix}{key}."))
    else:
        out[prefix.rstrip(".")] = json.dumps(value, sort_keys=True)
    return out


def _differing_keys(text: dict, multimodal: dict) -> set[str]:
    """Keys whose values differ between the text config and the multimodal config, with the
    text LM compared against ``model.lm``."""
    # Top-level settings the experiment config of this tree does not define cannot be
    # inherited yet (they arrive with newer text-side code); they are inherited once present.
    known = {f.name for f in dataclasses.fields(VisionAlignmentExperimentConfig)}
    text = {key: value for key, value in text.items() if key in known}
    text_model = text.pop("model")
    flat_text = _flatten(text)
    flat_text.update(_flatten({"model": {"lm": text_model}}))
    flat_mm = _flatten(multimodal)
    return {key for key in set(flat_text) | set(flat_mm) if flat_text.get(key) != flat_mm.get(key)}


def _round_trip(text_config: dict) -> dict:
    """The text config as this tree's classes serialize it, so comparisons see only real
    differences (not key order or defaults filled in by newer classes)."""
    text = json.loads(json.dumps(text_config))
    text["model"] = OLMoDDPModelConfig.from_dict(text["model"]).as_config_dict()
    text["train_module"] = OLMoDDPTrainModuleConfig.from_dict(text["train_module"]).as_config_dict()
    text["trainer"] = TrainerConfig.from_dict(text["trainer"]).as_config_dict()
    return text


def _hero_phases(alignment_recipe) -> dict[str, dict]:
    """Resolved config of the bridge phase built from the text config (perception/joint handoffs
    are covered by the phases layer)."""
    override = f"--recipe.text_config={FIXTURE}"
    configs, parent = {}, None
    for phase in ("bridge",):
        config = alignment_recipe.build(phase, parent, overrides=[override])
        configs[phase] = config.as_config_dict()
        parent = alignment_recipe.save(config)
    return configs


def _covered(key: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(key, pattern) or key.startswith(pattern + ".")


def test_hero_phases_differ_from_the_text_config_only_by_the_override_table(
    alignment_recipe, hero_checkpoint, text_config
):
    text = _round_trip(text_config)
    differing_by_phase = {
        phase: _differing_keys(text, config)
        for phase, config in _hero_phases(alignment_recipe).items()
    }
    for phase, differing in differing_by_phase.items():
        uncovered = sorted(
            key
            for key in differing
            if not any(_covered(key, pattern) for pattern in MULTIMODAL_OVERRIDES)
        )
        assert not uncovered, f"{phase}: differences not in MULTIMODAL_OVERRIDES: {uncovered}"


def test_hero_bridge_inherits_text_side_settings(alignment_recipe, hero_checkpoint, text_config):
    config = alignment_recipe.build(overrides=[f"--recipe.text_config={FIXTURE}"])
    lm = config.model.lm
    text_lm = OLMoDDPModelConfig.from_dict(text_config["model"])
    assert lm.n_layers == 16 and lm.block_overrides[7].sequence_mixer.backend == "flash_4"
    assert lm.recompute_each_block == text_lm.recompute_each_block is False
    assert isinstance(lm.block, OLMoDDPTransformerBlockConfig)
    assert isinstance(text_lm.block, OLMoDDPTransformerBlockConfig)
    assert lm.block.ep == text_lm.block.ep  # expert dispatch settings are the text run's own
    assert lm.block.routed_experts_router is not None
    assert lm.block.routed_experts_router.emo is None
    if hasattr(config, "process_group_timeout_seconds"):
        assert config.process_group_timeout_seconds == text_config["process_group_timeout_seconds"]
    optim = config.train_module.optim
    assert optim.eps == 1e-8 and optim.sigma_factor == 6 and optim.compile
    assert optim.weight_decay == 0.1 and optim.betas == (0.9, 0.95)
    assert optim.lr == 2e-4 and optim.clip_grad_norm_by_scheduler_group
    assert optim.foreach_chunk_size == 50_000_000
    module = config.train_module
    assert module.z_loss_multiplier == 1e-5 and module.compile_model
    assert module.ep_config is None and module.reset_optimizer_states_on_load
    assert module.dp_config.reduce_grads_in_fp32 and module.dp_config.accumulate_grads_in_fp32
    assert module.rank_microbatch_size == 4 * 8192
    trainer = config.trainer
    assert trainer.checkpointer.save_thread_count == 3 and trainer.checkpointer.throttle_uploads
    assert trainer.cancel_check_interval == 1000 and trainer.metrics_collect_interval == 10
    assert trainer.callbacks["garbage_collector"].gc_interval == 1000
    assert "profiler" in trainer.callbacks and "slack_notifier" not in trainer.callbacks
    assert trainer.callbacks["checkpointer"].save_interval == 500
    assert trainer.callbacks["checkpointer"].ephemeral_save_interval == 50
    assert config.data_loader.prefetch_workers == 8


def test_document_mode_turns_the_experimental_kda_kernels_off(alignment_recipe, hero_checkpoint):
    config = alignment_recipe.build(overrides=[f"--recipe.text_config={FIXTURE}"])
    mixers = [
        block.sequence_mixer
        for block in [config.model.lm.block, *config.model.lm.block_overrides.values()]
        if isinstance(block.sequence_mixer, KimiDeltaAttentionConfig)
    ]
    assert mixers and all(not mixer.use_experimental_kernels for mixer in mixers)
    assert "model.lm.block*.sequence_mixer.use_experimental_kernels" in MULTIMODAL_OVERRIDES


def test_without_a_text_config_the_checkpoint_config_is_normalized(
    alignment_recipe, hero_checkpoint, text_config
):
    config = alignment_recipe.build()
    lm = config.model.lm
    assert isinstance(lm.block, OLMoDDPTransformerBlockConfig)
    assert lm.block.routed_experts_router is not None
    assert lm.block.routed_experts_router.emo is None
    assert isinstance(lm.block.sequence_mixer, KimiDeltaAttentionConfig)
    assert lm.n_layers == 16 and not lm.block.sequence_mixer.use_experimental_kernels
    # Legacy defaults still apply to everything the text config would otherwise provide.
    assert config.train_module.optim.eps == 1e-6 and config.train_module.ep_config.degree == 8
    assert config.train_module.z_loss_multiplier == 1e-4


def test_text_config_must_match_the_pretraining_checkpoint(
    alignment_recipe, hero_checkpoint, tmp_path, text_config
):
    mismatched = json.loads(json.dumps(text_config))
    mismatched["model"]["n_layers"] = 12
    path = tmp_path / "other-text-config.json"
    path.write_text(json.dumps(mismatched))
    with pytest.raises(OLMoConfigurationError, match="n_layers"):
        alignment_recipe.build(overrides=[f"--recipe.text_config={path}"])


def test_text_config_accepts_a_dump_wrapping_the_config(
    alignment_recipe, hero_checkpoint, text_config, tmp_path
):
    path = tmp_path / "dump.json"
    path.write_text(json.dumps({"olmo_core_version": "x", "config": text_config}))
    config = alignment_recipe.build(overrides=[f"--recipe.text_config={path}"])
    assert config.recipe.text_config == str(path) and config.model.lm.n_layers == 16


def test_launch_inherits_the_text_image_resources_and_environment(monkeypatch, text_config):
    from gantry.api import GitRepoState

    from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig

    build_launch = Mock(
        side_effect=lambda **kwargs: BeakerLaunchConfig(
            name=kwargs["name"],
            cmd=kwargs["cmd"],
            clusters=[kwargs["cluster"]],
            workspace=kwargs["workspace"],
            budget=kwargs["budget"],
            beaker_image=kwargs["beaker_image"],
            num_nodes=kwargs["num_nodes"],
            num_gpus=8,
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret="JASONR_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret="JASONR_WANDB_API_KEY"),
                BeakerEnvSecret(name="COMET_API_KEY", secret="JASONR_COMET_API_KEY"),
            ],
            git=GitRepoState(
                repo="allenai/OLMo-core",
                repo_url="https://github.com/allenai/OLMo-core",
                ref="a" * 40,
                branch="vision",
            ),
        )
    )
    monkeypatch.setattr(vision_alignment, "build_launch_config", build_launch)
    cli = CliContext(
        script="src/scripts/train/Vision-Align.py",
        cmd=SubCmd.launch,
        run_name="alignment-launch",
        cluster="ai2/holmes",
        overrides=["--recipe.phase=bridge"],
    )
    launch = vision_alignment._build_launch(cli, work_dir="/tmp/cache", text=text_config)
    assert launch is not None
    text_launch = text_config["launch"]
    assert launch.beaker_image == text_launch["beaker_image"]
    assert launch.post_setup == text_launch["post_setup"]
    assert launch.num_nodes == 1 and launch.num_gpus == 8
    assert launch.shared_memory == "128GiB" and launch.priority == "urgent"
    assert launch.min_runtime == "8h" and launch.follow is False
    assert launch.workspace == "ai2/oe-olmo3p5-mt"
    assert build_launch.call_args.kwargs["budget"] == "ai2/oe-other"
    assert launch.google_credentials_secret == "GOOGLE_CREDENTIALS"
    assert {bucket.bucket for bucket in launch.weka_buckets} >= {
        entry["bucket"] for entry in text_launch["weka_buckets"]
    }
    env = {entry.name: entry.value for entry in launch.env_vars}
    assert len(env) == len(launch.env_vars)
    for entry in text_launch["env_vars"]:
        if entry["name"] != "PYTHONPATH":
            assert env[entry["name"]] == entry["value"]
    assert env["PYTHONPATH"] == "/gantry-runtime/src"
    assert env["OLMO_CORE_DATA_VERIFICATION_CACHE_DIR"] == "/tmp/cache/data-verification"
    assert "OLMO_USE_OWN_SYMM_MEM" not in env
    secrets = {entry.name: entry.secret for entry in launch.env_secrets}
    assert secrets["BEAKER_TOKEN"] == "jasonr_BEAKER_TOKEN"
    assert secrets["WANDB_API_KEY"] == "jasonr_WANDB_API_KEY"
    assert secrets["COMET_API_KEY"] == "JASONR_COMET_API_KEY"
    assert launch.aws_config_secret is launch.aws_credentials_secret is None
