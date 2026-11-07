import json
import os
from unittest.mock import Mock

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal import vision_alignment
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_alignment import VisionAlignmentExperimentConfig
from olmo_core.internal.vision_alignment_data import ALIGNMENT_MEAN_LOSS_WEIGHTS
from olmo_core.nn.transformer import OLMoDDPModelConfig

OUR_HUB_CACHE = "/weka/oe-training-default/jasonr/hf-home/hub"
OUR_DATASETS_CACHE = "/weka/oe-training-default/jasonr/hf-home/datasets"
RUSTINS_DATASETS_CACHE = "/weka/oe-training-default/rustin/hf-cache/datasets"


@pytest.mark.parametrize("weight", [0.0, 0.025])
@pytest.mark.parametrize("layout", ["default", "override", "mixed"])
@pytest.mark.parametrize(
    "d_model,num_experts,top_k,granularity",
    [(64, 4, 1, "local_batch"), (96, 12, 3, "instance")],
)
def test_router_load_balancing_override_only_changes_requested_coefficient(
    alignment_recipe, weight, layout, d_model, num_experts, top_k, granularity
):
    from olmo_core.nn.moe.v2.router import MoERouterConfigV2

    config_path = alignment_recipe.base / "config.json"
    base = json.loads(config_path.read_text())
    lm = OLMoDDPModelConfig.from_dict(base["model"])
    lm.d_model = d_model
    lm.n_layers = 5
    router = MoERouterConfigV2(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        lb_loss_weight=0.02,
        lb_loss_granularity=granularity,
        z_loss_weight=0.003,
        normalize_expert_weights=1.0,
        restore_weight_scale=True,
    )
    lm.block_overrides = {3: lm.block.copy()}
    if layout != "override":
        lm.block.routed_experts_router = router.copy()
    if layout != "default":
        lm.block_overrides[3].routed_experts_router = router.copy()
        lm.block_overrides[3].routed_experts_router.lb_loss_weight = 0.04
    base["model"] = lm.as_config_dict()
    config_path.write_text(json.dumps(base))

    inherited = alignment_recipe.build()
    overridden = alignment_recipe.build(overrides=[f"--recipe.router_lb_loss_weight={weight}"])
    expected = inherited.model.copy()
    for block in [expected.lm.block, *expected.lm.block_overrides.values()]:
        if block.routed_experts_router is not None:
            block.routed_experts_router.lb_loss_weight = weight
    assert overridden.model == expected
    assert overridden.train_module == inherited.train_module
    assert overridden.dataset == inherited.dataset
    assert overridden.recipe.router_lb_loss_weight == weight
    assert VisionAlignmentExperimentConfig.from_dict(overridden.as_config_dict()) == overridden


@pytest.mark.parametrize("phase", [vision_alignment.AlignmentPhase.bridge])
def test_launch_uses_standard_experiment_command_and_preset(monkeypatch, phase):
    from gantry.api import GitRepoState

    from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig
    from olmo_core.launch.beaker_presets import get_preset

    build_launch = Mock(
        side_effect=lambda **kwargs: BeakerLaunchConfig(
            name=kwargs["name"],
            cmd=kwargs["cmd"],
            clusters=[kwargs["cluster"]],
            workspace=kwargs["workspace"],
            num_nodes=kwargs["num_nodes"],
            num_gpus=8,
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret="RUSTINS_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret="OTHER_WANDB_API_KEY"),
            ],
            git=GitRepoState(
                repo="allenai/OLMo-core",
                repo_url="https://github.com/allenai/OLMo-core",
                ref="a" * 40,
                branch="vision-moe",
            ),
        )
    )
    monkeypatch.setattr(vision_alignment, "build_launch_config", build_launch)
    cli = CliContext(
        script="src/scripts/train/Vision-Align.py",
        cmd=SubCmd.launch,
        run_name="alignment-launch",
        cluster="ai2/holmes",
        overrides=[f"--recipe.phase={phase}"],
    )
    launch = vision_alignment._build_launch(
        cli, work_dir="/tmp/alignment-data-cache", hf_datasets_cache_dir=OUR_DATASETS_CACHE
    )
    assert launch is not None
    assert launch.cmd == [cli.script, "train", cli.run_name, cli.cluster, *cli.overrides]
    assert launch.num_nodes == 2 and launch.num_gpus == 8
    assert launch.workspace == "ai2/oe-olmo3p5-mt"
    assert build_launch.call_args.kwargs["budget"] == "ai2/oe-other"
    assert not launch.allow_dirty
    preset = get_preset("olmo-ddp")
    assert launch.beaker_image == preset.beaker_image
    assert launch.post_setup == preset.post_setup
    env = {entry.name: entry.value for entry in launch.env_vars}
    assert all(env[key] == value for key, value in preset.env_vars)
    assert (
        env["OLMO_CORE_DATA_VERIFICATION_CACHE_DIR"]
        == "/tmp/alignment-data-cache/data-verification"
    )
    assert env["HF_DATASETS_CACHE"] == OUR_DATASETS_CACHE
    assert "OLMO_CORE_FS_CACHE_DIR" not in env
    assert launch.priority == "urgent"
    assert launch.min_runtime == "8h"
    assert launch.shared_memory == "32GiB"
    assert launch.follow is False
    secrets = {entry.name: entry.secret for entry in launch.env_secrets}
    assert len(secrets) == len(launch.env_secrets)
    assert secrets["BEAKER_TOKEN"] == "jasonr_BEAKER_TOKEN"
    assert secrets["WANDB_API_KEY"] == "jasonr_WANDB_API_KEY"
    assert launch.aws_config_secret is launch.aws_credentials_secret is None
    assert build_launch.call_args.kwargs["step_timeout"] is None
    assert build_launch.call_args.kwargs["step_soft_timeout"] is None
    assert launch.step_timeout is launch.step_soft_timeout is None


@pytest.mark.parametrize(
    "hub,expected",
    [
        (OUR_HUB_CACHE, OUR_DATASETS_CACHE),
        ("/weka/oe-training-default/rustin/hf-cache/hub", RUSTINS_DATASETS_CACHE),
        ("/scratch/hf-cache", "/scratch/hf-cache/datasets"),
        ("/scratch/hf-home/hub/", "/scratch/hf-home/datasets"),
        (None, None),
    ],
)
def test_hf_datasets_cache_is_the_hub_caches_sibling(hub, expected):
    assert vision_alignment.default_hf_datasets_cache_dir(hub) == expected


def test_hf_datasets_cache_dir_derives_from_the_recipe_hub_cache(alignment_recipe):
    """The recipe's ``datasets`` cache follows its Hub cache, an explicit value wins, an explicit
    ``null`` leaves the library default, and the field changes nothing else in the config."""
    default = alignment_recipe.build()
    assert default.recipe.hf_datasets_cache_dir == RUSTINS_DATASETS_CACHE
    ours = alignment_recipe.build(overrides=[f"--recipe.hf_cache_dir={OUR_HUB_CACHE}"])
    assert ours.recipe.hf_datasets_cache_dir == OUR_DATASETS_CACHE
    explicit = alignment_recipe.build(
        overrides=[
            f"--recipe.hf_cache_dir={OUR_HUB_CACHE}",
            "--recipe.hf_datasets_cache_dir=/scratch/arrow",
        ]
    )
    assert explicit.recipe.hf_datasets_cache_dir == "/scratch/arrow"
    disabled = alignment_recipe.build(overrides=["--recipe.hf_datasets_cache_dir=null"])
    assert disabled.recipe.hf_datasets_cache_dir is None
    no_hub = alignment_recipe.build(overrides=["--recipe.hf_cache_dir=null"])
    assert no_hub.recipe.hf_datasets_cache_dir is None

    # Model, train module, data loader, dataset and trainer are untouched: the resolved dumps
    # differ only by the recipe field (``as_config_dict`` omits the ``None`` of the disabled one).
    default_dump, disabled_dump = default.as_config_dict(), disabled.as_config_dict()
    assert default_dump["recipe"].pop("hf_datasets_cache_dir") == RUSTINS_DATASETS_CACHE
    assert "hf_datasets_cache_dir" not in disabled_dump["recipe"]
    assert default_dump == disabled_dump
    for config in (default, ours, explicit, disabled):
        assert VisionAlignmentExperimentConfig.from_dict(config.as_config_dict()) == config


def test_launched_alignment_carries_the_recipe_hf_datasets_cache(alignment_recipe, monkeypatch):
    from gantry.api import GitRepoState

    from olmo_core.launch.beaker import BeakerLaunchConfig

    monkeypatch.setattr(
        vision_alignment,
        "build_launch_config",
        lambda **kwargs: BeakerLaunchConfig(
            name=kwargs["name"],
            cmd=kwargs["cmd"],
            clusters=[kwargs["cluster"]],
            workspace=kwargs["workspace"],
            num_nodes=kwargs["num_nodes"],
            git=GitRepoState(
                repo="allenai/OLMo-core",
                repo_url="https://github.com/allenai/OLMo-core",
                ref="a" * 40,
                branch="vision",
            ),
        ),
    )

    def build(*overrides):
        return vision_alignment.build_config(
            CliContext(
                script="src/scripts/train/Vision-Align.py",
                cmd=SubCmd.launch,
                run_name="alignment-bridge",
                cluster="ai2/holmes",
                overrides=[
                    "--recipe.phase=bridge",
                    f"--recipe.pretraining_checkpoint={alignment_recipe.base}",
                    f"--recipe.artifact_root={alignment_recipe.base.parent}/artifacts",
                    f"--recipe.output_root={alignment_recipe.base.parent}/runs",
                    f"--recipe.work_dir={alignment_recipe.base.parent}/cache",
                    *(
                        f"--dataset.mean_loss_weight.{name}={mean}"
                        for name, mean in ALIGNMENT_MEAN_LOSS_WEIGHTS["bridge"].items()
                    ),
                    *overrides,
                ],
            )
        )

    def env_of(config):
        env = {entry.name: entry.value for entry in config.launch.env_vars}
        assert len(env) == len(config.launch.env_vars)
        return env

    ours = env_of(build(f"--recipe.hf_cache_dir={OUR_HUB_CACHE}"))
    assert ours.pop("HF_DATASETS_CACHE") == OUR_DATASETS_CACHE
    explicit = env_of(
        build(f"--recipe.hf_cache_dir={OUR_HUB_CACHE}", "--recipe.hf_datasets_cache_dir=/arrow")
    )
    assert explicit.pop("HF_DATASETS_CACHE") == "/arrow"
    disabled = env_of(build("--recipe.hf_datasets_cache_dir=null"))
    assert "HF_DATASETS_CACHE" not in disabled
    # Every other entry is the same in all three.
    assert ours == explicit == disabled
    assert ours["PYTHONPATH"] == "/gantry-runtime/src"


def test_set_hf_datasets_cache_replaces_only_its_own_entry():
    from gantry.api import GitRepoState

    from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig

    original = [
        BeakerEnvVar(name="OTHER", value="kept"),
        BeakerEnvVar(name="HF_DATASETS_CACHE", value="/local/disk"),
        BeakerEnvVar(name="LAST", value="kept too"),
    ]
    launch = BeakerLaunchConfig(
        name="ladders-mixed",
        cmd=["train"],
        env_vars=original,
        git=GitRepoState(
            repo="allenai/scaling-ladders",
            repo_url="https://github.com/allenai/scaling-ladders",
            ref="a" * 40,
            branch="main",
        ),
    )
    vision_alignment.set_hf_datasets_cache(launch, None)
    assert launch.env_vars is original
    vision_alignment.set_hf_datasets_cache(launch, OUR_DATASETS_CACHE)
    assert [(entry.name, entry.value) for entry in launch.env_vars] == [
        ("OTHER", "kept"),
        ("LAST", "kept too"),
        ("HF_DATASETS_CACHE", OUR_DATASETS_CACHE),
    ]
    # The builder's own list (shared through a shallow ``replace``) is not mutated.
    assert [entry.value for entry in original] == ["kept", "/local/disk", "kept too"]


def test_training_exports_the_recipe_hf_datasets_cache(alignment_recipe, monkeypatch):
    """A job whose launch predates the launch variable still honours the recipe; the launch
    environment, when present, wins; ``None`` leaves the environment alone."""
    config = alignment_recipe.build(overrides=[f"--recipe.hf_cache_dir={OUR_HUB_CACHE}"])
    seen = []

    def stop(seed):
        seen.append(os.environ.get("HF_DATASETS_CACHE"))
        raise RuntimeError("stop before building the model")

    monkeypatch.setattr(vision_alignment, "seed_all", stop)
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    with pytest.raises(RuntimeError, match="stop before"):
        vision_alignment.train(config)
    monkeypatch.setenv("HF_DATASETS_CACHE", "/from/the/launch")
    with pytest.raises(RuntimeError, match="stop before"):
        vision_alignment.train(config)
    monkeypatch.delenv("HF_DATASETS_CACHE")
    config.recipe.hf_datasets_cache_dir = None
    with pytest.raises(RuntimeError, match="stop before"):
        vision_alignment.train(config)
    assert seen == [OUR_DATASETS_CACHE, "/from/the/launch", None]
    assert "HF_DATASETS_CACHE" not in os.environ


def test_local_config_does_not_construct_beaker_launch(monkeypatch):
    build_launch = Mock()
    monkeypatch.setattr(vision_alignment, "build_launch_config", build_launch)
    cli = CliContext(
        script="src/scripts/train/Vision-Align.py",
        cmd=SubCmd.dry_run,
        run_name="alignment-local",
        cluster="local",
        overrides=[],
    )
    assert vision_alignment._build_launch(cli) is None
    build_launch.assert_not_called()


def test_parse_cli_args_reads_argv(monkeypatch):
    argv = ["Vision-Align.py", "dry_run", "run01", "local", "--recipe.phase=bridge"]
    monkeypatch.setattr("sys.argv", argv)
    assert vision_alignment.parse_cli_args() == CliContext(
        "Vision-Align.py", SubCmd.dry_run, "run01", "local", ["--recipe.phase=bridge"]
    )


@pytest.mark.parametrize("cmd", ["prep", "eval_checkpoints", "bogus"])
def test_parse_cli_args_rejects_unsupported_subcommands(monkeypatch, cmd):
    monkeypatch.setattr("sys.argv", ["Vision-Align.py", cmd, "run01", "local"])
    with pytest.raises(SystemExit):
        vision_alignment.parse_cli_args()


@pytest.mark.parametrize("cmd", [SubCmd.prep, SubCmd.eval_checkpoints])
def test_run_rejects_unsupported_subcommands(cmd):
    config = Mock(launch=None)
    with pytest.raises(OLMoConfigurationError, match="does not support"):
        vision_alignment.run(cmd, config)
