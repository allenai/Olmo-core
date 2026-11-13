"""CPU config-build checks for the tech-report entry points in ``src/examples/olmo_ddp``.

These tests build model, train-module, and trainer configs only; they do not
instantiate models or touch GPUs, data, or the network.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("beaker")  # olmo_core.internal, imported by every script, requires it.

SCRIPT_DIR = Path(__file__).parents[2] / "examples" / "olmo_ddp"
VOCAB_SIZE = 100_352  # dolma2 tokenizer, padded to a multiple of 128.


def _load(script: str) -> ModuleType:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))  # scripts import moe_8l_common as a sibling module
    name = "olmo_ddp_example_" + script.removesuffix(".py").replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _common(module: ModuleType) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer=SimpleNamespace(padded_vocab_size=lambda: VOCAB_SIZE),
        max_sequence_length=getattr(module, "SEQUENCE_LENGTH", 8192),
        global_batch_size=getattr(module, "GLOBAL_BATCH_SIZE", None),
        run_name="test",
        work_dir="/tmp/olmo-ddp-example-test",
        save_folder="/tmp/olmo-ddp-example-test",
    )


ENTRY_POINTS = sorted(p.name for p in SCRIPT_DIR.glob("*.py") if p.name != "moe_8l_common.py")


@pytest.mark.parametrize("script", ENTRY_POINTS)
def test_entry_point_configs_build(script):
    module = _load(script)
    common = _common(module)
    model = module.build_model_config(common)
    train_module = module.build_train_module_config(common)
    assert model.num_params >= model.num_active_params > 0
    assert train_module.rank_microbatch_size > 0


# (script, EP degree, PP degree, global batch in Mi tokens, rank microbatch tokens,
#  random routing, active params in B, total params in B), as reported.
PRODUCTION = [
    ("OLMoE3-dev-t001-random.py", None, None, 8, 65_536, True, 1.59, 12.91),
    ("OLMoE3-dev-t001.py", None, None, 8, 65_536, False, 1.59, 12.91),
    ("OLMoE3-dev-t002.py", 8, None, 8, 65_536, False, 1.59, 12.91),
    ("OLMoE3-dev-s001-random.py", 8, None, 24, 24_576, True, 4.29, 40.86),
    ("OLMoE3-dev-s001.py", 8, None, 24, 24_576, False, 4.29, 40.86),
    ("OLMoE3-dev-s001-random-16mi.py", 8, None, 16, 16_384, True, 4.29, 40.86),
    ("OLMoE3-dev-m001-64e-16mi.py", 8, 2, 16, 16_384, True, 7.37, 70.22),
    ("OLMoE3-dev-m001.py", 8, 2, 24, 16_384, True, 7.38, 103.75),
    ("OLMoE3-dev-m002.py", 8, 4, 24, 16_384, True, 7.38, 137.27),
    ("OLMoE3-dev-l001.py", 8, 4, 32, 8_192, True, 15.14, 295.99),
    ("OLMoE3-dev-l001-pp8-24mi.py", 8, 8, 24, 8_192, True, 15.14, 295.99),
    ("OLMoE3-dev-u001.py", 8, 8, 64, 8_192, True, 58.36, 1200.0),
    ("OLMoE3-dev-u002.py", 32, 8, 64, 16_384, True, 58.41, 2380.0),
]


@pytest.mark.parametrize(
    ("script", "ep", "pp", "gbs_mi", "rank_mbsz", "random", "active_b", "total_b"), PRODUCTION
)
def test_production_operating_points_match_report(
    script, ep, pp, gbs_mi, rank_mbsz, random, active_b, total_b
):
    module = _load(script)
    common = _common(module)
    model = module.build_model_config(common)
    train_module = module.build_train_module_config(common)

    assert round(model.num_active_params / 1e9, 2) == active_b
    # Trillion-scale totals are reported to four significant figures (1.200T, 2.380T).
    total_digits = 0 if total_b >= 1000 else 2
    assert round(model.num_params / 1e9, total_digits) == round(total_b, total_digits)
    assert module.GLOBAL_BATCH_SIZE == gbs_mi * 1024 * 1024
    assert train_module.rank_microbatch_size == rank_mbsz
    assert (train_module.ep_config.degree if train_module.ep_config else None) == ep
    assert (train_module.pp_config.degree if train_module.pp_config else None) == pp
    assert model.block.routed_experts_router.random_expert_assignment is random


@pytest.fixture
def benchmark(monkeypatch):
    module = _load("moe_8l_ddp.py")
    monkeypatch.setattr(module, "EP_PATH_NAME", "auto")
    monkeypatch.setattr(module, "PARALLEL_DEGREE", 8)
    monkeypatch.setattr(module, "RANK_MICROBATCH_SEQUENCES", 2)
    monkeypatch.setattr(module, "FORCE_FUSED_ATTENTION", False)
    monkeypatch.setattr(module, "RECOMPUTE_EACH_BLOCK", False)
    monkeypatch.setattr(module, "TWO_BATCH_OVERLAP", False)
    monkeypatch.setattr(module, "UNIFORM_ROUTING", False)
    return module


def test_benchmark_default_schedule_and_random_routing(benchmark):
    config = benchmark.build_model_config(_common(benchmark))

    assert config.two_batch_overlap is False
    assert config.block.ep.schedule == benchmark.ExpertParallelSchedule.normal
    assert config.block.ep.shared_slots == 1
    assert config.block.routed_experts_router.uniform_expert_assignment is False
    assert config.block.routed_experts_router.random_expert_assignment is True


def test_benchmark_fused_attention_can_be_forced_without_mxfp8(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark, "FORCE_FUSED_ATTENTION", True)
    monkeypatch.setattr(benchmark, "MXFP8_ATTN_QKV", False)
    monkeypatch.setattr(benchmark, "MXFP8_ATTN_OUT", False)
    monkeypatch.setattr(benchmark, "MXFP8_ATTN_SAVE_QKV", False)

    config = benchmark.build_model_config(_common(benchmark))

    attention = config.block.sequence_mixer
    assert attention.name == benchmark.AttentionType.fused_v2
    assert attention.mxfp8_qkv_projection is None
    assert attention.mxfp8_out_projection is None
    assert not attention.mxfp8_save_qkv_for_backward


def test_benchmark_rejects_two_batch_overlap(benchmark, monkeypatch):
    monkeypatch.setattr(benchmark, "TWO_BATCH_OVERLAP", True)

    with pytest.raises(ValueError, match="not supported yet"):
        benchmark.build_model_config(_common(benchmark))
