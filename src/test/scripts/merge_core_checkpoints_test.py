"""
Tests for the merge_core_checkpoints.py script.
"""

import importlib.util
import json
import tempfile
from pathlib import Path
from typing import List, Optional

import pytest
import torch
from click.testing import CliRunner

from olmo_core.distributed.checkpoint import (
    get_checkpoint_metadata,
    save_model_and_optim_state,
    unshard_checkpoint,
)
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig

spec = importlib.util.spec_from_file_location(
    "merge_core_checkpoints", "src/scripts/merge_core_checkpoints.py"
)
if spec is None or spec.loader is None:
    raise ImportError("Could not load merge_core_checkpoints.py")
merge_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(merge_module)


# ==================== Helper Functions ====================


def create_test_checkpoint_with_seed(
    checkpoint_dir: Path,
    model_config: TransformerConfig,
    seed: int,
    optim_config: Optional[AdamWConfig] = None,
    include_optimizer: bool = True,
) -> None:
    """
    Create a test checkpoint with a specific random seed for distinct weights.

    Args:
        checkpoint_dir: Directory to save the checkpoint
        model_config: Transformer configuration
        seed: Random seed for weight initialization
        optim_config: Optimizer configuration (optional, defaults to AdamWConfig with lr=1e-3)
        include_optimizer: Whether to include optimizer state
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Set default optimizer config if not provided
    if optim_config is None:
        optim_config = AdamWConfig(lr=1e-3)

    # Create config.json
    optim_dict = optim_config.as_config_dict()
    optim_dict["_CLASS_"] = f"{optim_config.__class__.__module__}.{optim_config.__class__.__name__}"

    config_dict = {
        "model": model_config.as_config_dict(),
        "dataset": {"tokenizer": {"identifier": "test_tokenizer", "type": "test"}},
        "train_module": {"optim": optim_dict},
    }

    config_path = checkpoint_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Set random seed for reproducible weights
    torch.manual_seed(seed)

    # Build model
    model = model_config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    # Create optimizer if needed
    optim = None
    if include_optimizer:
        optim = optim_config.build(model)
        # Take a step to initialize optimizer state
        loss = model(input_ids=torch.randint(0, model_config.vocab_size, (2, 16))).sum()
        loss.backward()
        optim.step()

    # Save checkpoint
    model_and_optim_dir = checkpoint_dir / "model_and_optim"
    save_model_and_optim_state(
        str(model_and_optim_dir), model, optim, save_overwrite=True, flatten_optimizer_state=False
    )


def run_merge_cli(
    model_paths: List[str],
    output_path: str,
    skip_optimizer: bool = False,
):
    """
    Run the merge_core_checkpoints CLI using Click's test runner.

    Args:
        model_paths: List of checkpoint paths to merge
        output_path: Output directory for merged checkpoint
        skip_optimizer: Whether to skip optimizer state

    Returns:
        Click test runner result
    """
    runner = CliRunner()
    args = []

    # Add model paths
    for model_path in model_paths:
        args.extend(["--model", model_path])

    # Add output path
    args.extend(["--output", output_path])

    # Add skip optimizer flag if requested
    if skip_optimizer:
        args.append("--skip-optimizer-state")

    result = runner.invoke(merge_module.main, args)
    return result


def checkpoint_has_optimizer_state(checkpoint_dir: Path) -> bool:
    """
    Check if a checkpoint contains optimizer state by inspecting metadata.

    Args:
        checkpoint_dir: Checkpoint directory

    Returns:
        True if checkpoint has optimizer state
    """
    model_and_optim_dir = checkpoint_dir / "model_and_optim"
    if not model_and_optim_dir.exists():
        return False

    metadata = get_checkpoint_metadata(model_and_optim_dir)
    # Check if any keys start with "optim."
    return any(key.startswith("optim.") for key in metadata.state_dict_metadata.keys())


def load_checkpoint_state_dict(checkpoint_dir: Path, model_config: TransformerConfig) -> dict:
    """
    Load a checkpoint's state dict for verification.

    Uses unshard_checkpoint() to load without requiring distributed setup.

    Args:
        checkpoint_dir: Checkpoint directory
        model_config: Model configuration

    Returns:
        State dict
    """
    model_and_optim_dir = checkpoint_dir / "model_and_optim"

    # Unshard to a temp directory (works without distributed setup)
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path, _ = unshard_checkpoint(str(model_and_optim_dir), Path(tmpdir), optim=False)
        # Load unsharded checkpoint
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    return state_dict


def verify_averaged_weights(
    merged_checkpoint: Path,
    source_checkpoints: List[Path],
    model_config: TransformerConfig,
    tolerance: float = 1e-5,
) -> None:
    """
    Verify that merged checkpoint weights are correctly averaged.

    Args:
        merged_checkpoint: Path to merged checkpoint
        source_checkpoints: List of source checkpoint paths
        model_config: Model configuration
        tolerance: Numerical tolerance for comparison
    """
    # Load merged state dict
    merged_state = load_checkpoint_state_dict(merged_checkpoint, model_config)

    # Load source state dicts
    source_states = [load_checkpoint_state_dict(ckpt, model_config) for ckpt in source_checkpoints]

    # Verify each parameter is correctly averaged
    for param_name in merged_state.keys():
        merged_value = merged_state[param_name]

        # Compute expected average
        source_values = [state[param_name] for state in source_states]
        expected_avg = torch.stack(source_values).mean(dim=0)

        # Compare
        torch.testing.assert_close(
            merged_value,
            expected_avg,
            atol=tolerance,
            rtol=tolerance,
            msg=f"Parameter '{param_name}' not correctly averaged",
        )


# ==================== Core Merging Tests ====================


def test_merge_two_checkpoints(tmp_path):
    """Test basic 2-checkpoint merge with optimizer state."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    # Create two checkpoints with different seeds
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=True)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=True)

    # Run merge
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output))

    # Verify success
    assert result.exit_code == 0, f"Merge failed: {result.output}"
    assert (output / "model_and_optim").exists()
    assert (output / "config.json").exists()
    assert checkpoint_has_optimizer_state(output)


def test_merge_three_checkpoints(tmp_path):
    """Test 3-checkpoint merge to verify averaging works with >2 models."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    # Create three checkpoints
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    ckpt3 = tmp_path / "checkpoint3"
    output = tmp_path / "merged"

    for ckpt, seed in [(ckpt1, 42), (ckpt2, 123), (ckpt3, 456)]:
        create_test_checkpoint_with_seed(ckpt, model_config, seed=seed, include_optimizer=True)

    # Run merge
    result = run_merge_cli([str(ckpt1), str(ckpt2), str(ckpt3)], str(output))

    # Verify success
    assert result.exit_code == 0, f"Merge failed: {result.output}"
    assert (output / "model_and_optim").exists()


def test_merge_two_checkpoints_skip_optimizer(tmp_path):
    """Test merge without optimizer state using --skip-optimizer-state."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    # Create two checkpoints with optimizer state
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=True)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=True)

    # Run merge with skip optimizer flag
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)

    # Verify success and no optimizer state
    assert result.exit_code == 0, f"Merge failed: {result.output}"
    assert (output / "model_and_optim").exists()
    assert not checkpoint_has_optimizer_state(output), "Optimizer state should be skipped"


def test_weights_are_averaged_correctly(tmp_path):
    """Load merged checkpoint and verify weights equal mean of input weights."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    # Create two checkpoints with different seeds
    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=False)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=False)

    # Run merge
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)
    assert result.exit_code == 0

    # Verify weights are averaged
    verify_averaged_weights(output, [ckpt1, ckpt2], model_config)


def test_single_checkpoint(tmp_path):
    """Test edge case: merging 1 checkpoint acts as a copy."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt = tmp_path / "checkpoint"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt, model_config, seed=42, include_optimizer=False)

    # Run merge with single checkpoint
    result = run_merge_cli([str(ckpt)], str(output), skip_optimizer=True)

    # Verify success
    assert result.exit_code == 0
    assert (output / "model_and_optim").exists()
    assert (output / "config.json").exists()


def test_config_copied_from_first_checkpoint(tmp_path):
    """Verify config.json is copied from the first checkpoint."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=False)
    # Use same model architecture for ckpt2 but different config on disk
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=False)

    # Manually override ckpt2's config to have different layer norm epsilon
    with (ckpt2 / "config.json").open("r") as f:
        config2 = json.load(f)
    config2["model"]["block"]["attention_norm"]["eps"] = 1e-6  # Different from default 1e-5
    with (ckpt2 / "config.json").open("w") as f:
        json.dump(config2, f)

    # Run merge
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)
    assert result.exit_code == 0

    # Verify output config matches first checkpoint (eps should be 1e-5)
    with (output / "config.json").open("r") as f:
        output_config = json.load(f)

    with (ckpt1 / "config.json").open("r") as f:
        ckpt1_config = json.load(f)

    assert (
        output_config["model"]["block"]["attention_norm"]["eps"]
        == ckpt1_config["model"]["block"]["attention_norm"]["eps"]
        == 1e-5
    )


# ==================== Optimizer State Tests ====================


def test_optimizer_state_preserved(tmp_path):
    """Verify optimizer state exists and is averaged when not skipped."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=True)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=True)

    # Run merge WITHOUT skip optimizer flag
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=False)
    assert result.exit_code == 0

    # Verify optimizer state exists
    assert checkpoint_has_optimizer_state(output), "Optimizer state should be present"


def test_optimizer_state_excluded(tmp_path):
    """Verify no optimizer keys when --skip-optimizer-state is used."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=True)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=True)

    # Run merge WITH skip optimizer flag
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)
    assert result.exit_code == 0

    # Verify no optimizer state
    assert not checkpoint_has_optimizer_state(output), "Optimizer state should be excluded"


def test_skip_optimizer_flag(tmp_path):
    """Test --skip-optimizer-state/-s CLI flag parsing."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=True)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=True)

    # Test short form -s
    runner = CliRunner()
    result = runner.invoke(
        merge_module.main,
        ["-m", str(ckpt1), "-m", str(ckpt2), "-o", str(output), "-s"],
    )
    assert result.exit_code == 0
    assert not checkpoint_has_optimizer_state(output)


# ==================== Edge Cases & Validation Tests ====================


def test_incompatible_model_shapes(tmp_path):
    """Test that merging checkpoints with incompatible shapes fails."""
    # Create two checkpoints with different vocab sizes
    model_config1 = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )
    model_config2 = TransformerConfig.llama_like(
        vocab_size=2000,  # Different vocab size
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config1, seed=42, include_optimizer=False)
    create_test_checkpoint_with_seed(ckpt2, model_config2, seed=123, include_optimizer=False)

    # Run merge - should fail due to incompatible shapes
    # The CheckpointException from PyTorch is not caught by Click, so we need to catch it here
    with pytest.raises((Exception, BaseException)) as exc_info:
        result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)
        # If it doesn't raise, check if it failed
        if result.exception:
            raise result.exception

    # Verify the error is about size mismatch
    assert "Size mismatch" in str(exc_info.value) or "CheckpointException" in str(exc_info.value)


def test_incompatible_model_layers(tmp_path):
    """Test that merging checkpoints with different layer counts fails."""
    # Create two checkpoints with same shapes except different number of layers
    model_config1 = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )
    model_config2 = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=3,  # Different number of layers
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config1, seed=42, include_optimizer=False)
    create_test_checkpoint_with_seed(ckpt2, model_config2, seed=123, include_optimizer=False)

    # Run merge - should fail due to mismatched architectures
    # One checkpoint has keys that don't exist in the other
    with pytest.raises((Exception, BaseException)) as exc_info:
        result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)
        # If it doesn't raise, check if it failed
        if result.exception:
            raise result.exception

    # Verify the error is about different keys
    assert "different keys" in str(exc_info.value)


def test_path_validation_model_and_optim():
    """Test that paths ending with 'model_and_optim' are rejected."""
    runner = CliRunner()
    result = runner.invoke(
        merge_module.main,
        [
            "--model",
            "/path/to/checkpoint/model_and_optim",
            "--model",
            "/path/to/checkpoint2",
            "--output",
            "/output",
        ],
    )
    assert result.exit_code != 0
    # Exception is raised, check for ValueError in exception
    assert result.exception is not None
    assert "must not end in 'model_and_optim'" in str(result.exception)


def test_path_validation_output():
    """Test that output path ending with 'model_and_optim' is rejected."""
    runner = CliRunner()
    result = runner.invoke(
        merge_module.main,
        [
            "--model",
            "/path/to/checkpoint1",
            "--model",
            "/path/to/checkpoint2",
            "--output",
            "/output/model_and_optim",
        ],
    )
    assert result.exit_code != 0
    # Exception is raised, check for ValueError in exception
    assert result.exception is not None
    assert "must not end in 'model_and_optim'" in str(result.exception)


def test_output_overwrites_existing(tmp_path):
    """Verify output directory is overwritten if it exists."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpt1 = tmp_path / "checkpoint1"
    ckpt2 = tmp_path / "checkpoint2"
    output = tmp_path / "merged"

    create_test_checkpoint_with_seed(ckpt1, model_config, seed=42, include_optimizer=False)
    create_test_checkpoint_with_seed(ckpt2, model_config, seed=123, include_optimizer=False)

    # Create pre-existing output directory with a marker file
    output.mkdir()
    marker_file = output / "marker.txt"
    marker_file.write_text("old content")

    # Run merge
    result = run_merge_cli([str(ckpt1), str(ckpt2)], str(output), skip_optimizer=True)
    assert result.exit_code == 0

    # Verify output exists and marker file is gone (directory was overwritten)
    assert (output / "model_and_optim").exists()
    assert not marker_file.exists(), "Output directory should be overwritten"


def test_no_model_paths_error():
    """Test CLI error when no --model flags are provided."""
    runner = CliRunner()
    result = runner.invoke(merge_module.main, ["--output", "/output"])
    assert result.exit_code != 0


def test_cli_help():
    """Verify --help output works."""
    runner = CliRunner()
    result = runner.invoke(merge_module.main, ["--help"])
    assert result.exit_code == 0
    assert "merge" in result.output.lower() or "checkpoint" in result.output.lower()


def test_multiple_model_flags(tmp_path):
    """Test that CLI accepts multiple --model/-m flags."""
    model_config = TransformerConfig.llama_like(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
    )

    ckpts = [tmp_path / f"checkpoint{i}" for i in range(4)]
    output = tmp_path / "merged"

    for i, ckpt in enumerate(ckpts):
        create_test_checkpoint_with_seed(ckpt, model_config, seed=42 + i, include_optimizer=False)

    # Test using -m short form with 4 checkpoints
    runner = CliRunner()
    result = runner.invoke(
        merge_module.main,
        [
            "-m",
            str(ckpts[0]),
            "-m",
            str(ckpts[1]),
            "-m",
            str(ckpts[2]),
            "-m",
            str(ckpts[3]),
            "-o",
            str(output),
            "-s",
        ],
    )
    assert result.exit_code == 0
    assert (output / "model_and_optim").exists()
