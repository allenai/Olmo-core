"""Protect the six-arm comparison and the requested single-GPU launch resources."""
import importlib
from pathlib import Path

import pytest


@pytest.fixture
def family(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts/train/memexpress/leaky_sft"
    monkeypatch.syspath_prepend(str(path))
    return importlib.import_module("_qwen35_contradiction_common")


def context(arm, cmd="dry_run", overrides=None):
    from olmo_core.internal.experiment import CliContext, SubCmd

    return CliContext(
        script=__file__,
        cmd=SubCmd(cmd),
        run_name=f"leaky-test-{arm}",
        cluster="ai2/jupiter-cirrascale-2",
        overrides=overrides or [],
    )


def test_six_configs_differ_only_in_alpha_and_names(family):
    configs = []
    assert list(family.ARMS.values()) == [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    for arm, alpha in family.ARMS.items():
        cfg = family.build_leaky_experiment(context(arm), arm=arm)
        assert cfg.train_module.cpt_loss_scaler == alpha
        assert cfg.trainer.max_duration.value == 1 and cfg.trainer.max_duration.unit == "epochs"
        assert cfg.train_module.cp_config is None
        assert cfg.train_module.dp_config.shard_degree == 1
        assert cfg.train_module.dp_config.num_replicas == 1
        assert cfg.data_loader.global_batch_size == cfg.train_module.rank_microbatch_size == 32768
        assert cfg.trainer.cancel_check_interval == 1
        assert cfg.trainer.load_optim_state is False and cfg.trainer.load_trainer_state is False
        assert cfg.train_module.state_dict_load_opts is None  # strict default
        assert cfg.trainer.save_overwrite is False
        assert len(cfg.dataset) == 1 and len(cfg.dataset[0].sources) == 1
        assert cfg.data_loader.generate_doc_lengths
        assert cfg.data_loader.tokenizer.bos_token_id is None
        # Construct actual attention on meta; counting parameters alone misses bad options.
        cfg.model.block["attn"].sequence_mixer.build(
            cfg.model.d_model, layer_idx=3, n_layers=cfg.model.n_layers, init_device="meta"
        )
        raw = cfg.as_config_dict()
        raw["run_name"] = "NAME"
        raw["trainer"]["save_folder"] = "SAVE"
        raw["trainer"]["callbacks"]["wandb"]["name"] = "NAME"
        raw["train_module"]["cpt_loss_scaler"] = "ALPHA"
        configs.append(raw)
    assert all(config == configs[0] for config in configs)


def test_requested_resources_on_every_command(family, tmp_path):
    launcher = importlib.import_module("launch_sweep")
    plan = launcher.make_plan("fresh", "a" * 40, tmp_path)
    assert len(plan) == 6 and len({row["run_name"] for row in plan}) == 6
    for row in plan:
        cmd = row["command"]
        for flag, value in [
            ("--gpus", "1"),
            ("--replicas", "1"),
            ("--priority", "urgent"),
            ("--min-runtime", "1h"),
            ("--cluster", "ai2/jupiter-cirrascale-2"),
        ]:
            assert cmd.count(flag) == 1
            assert cmd[cmd.index(flag) + 1] == value
        assert "--torchrun" in cmd and "--no-logs" in cmd
        assert row["min_runtime"] == "1h" and row["epochs"] == 1
        assert row["cpt_loss_scaler"] == family.ARMS[row["arm"]]


def test_uses_256k_corpus_and_excludes_long_examples(family):
    from olmo_core.data.composable import LongDocStrategy

    cfg = family.build_leaky_experiment(context("a00", "train"), arm="a00")
    dataset = cfg.dataset[0]
    assert family.DATA_ROOT.endswith("xlong5_2k256k_qwen35/shards_full/contradiction_train")
    assert dataset.sources[0].source_paths == [family.DATA_ROOT + "/token_ids_part_*.npy"]
    assert dataset.sources[0].label_mask_paths == [family.DATA_ROOT + "/labels_mask_*.npy"]
    assert dataset.sources[0].max_document_length is None  # do not truncate before filtering
    assert dataset.long_doc_strategy == LongDocStrategy.exclude
    assert dataset.sequence_length == 32768


def test_builtin_launch_cannot_bypass_resource_policy(family):
    with pytest.raises(ValueError, match="launch_sweep"):
        family.build_leaky_experiment(context("a01", "launch"), arm="a01")


def test_filter_boundary_preserves_complete_examples(tmp_path, family):
    from olmo_core.data import TokenizerConfig
    from olmo_core.data.composable import PackingInstanceSource, LongDocStrategy
    from olmo_core.data.composable.token_source import InMemoryDocumentSource

    tok = TokenizerConfig(vocab_size=100, eos_token_id=99, pad_token_id=98)
    # Unique IDs identify examples below, exactly at, and above the inclusive cutoff.
    tokens = [1] * 32766 + [99] + [2] * 32767 + [99] + [3] * 32768 + [99]
    source = InMemoryDocumentSource(tokens=tokens, tokenizer=tok, work_dir=tmp_path)
    packed = PackingInstanceSource(
        source,
        tokenizer=tok,
        work_dir=tmp_path,
        sequence_length=family.SEQUENCE_LENGTH,
        long_doc_strategy=LongDocStrategy.exclude,
    )
    assert len(packed) == 2
    retained = []
    for instance in packed:
        retained.extend(
            t for t, valid in zip(instance["input_ids"], instance["loss_valid_mask"]) if valid
        )
    assert retained.count(1) == 32766 and retained.count(2) == 32767
    assert retained.count(99) == 2 and 3 not in retained
