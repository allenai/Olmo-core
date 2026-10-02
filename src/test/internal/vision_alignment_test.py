import json
from unittest.mock import Mock

import pytest

from olmo_core.internal import vision_alignment
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_alignment import VisionAlignmentExperimentConfig
from olmo_core.nn.transformer import OLMoDDPModelConfig


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
    launch = vision_alignment._build_launch(cli, work_dir="/tmp/alignment-data-cache")
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
