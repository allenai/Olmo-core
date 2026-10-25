"""CPU tests for eval_devce.py / compare_devce.py (tiny OLMo-3-shaped model, torch attention).

    PYTHONPATH=src:research/token_routing/olmo3_cptdrop python -m pytest -q research/token_routing/olmo3_cptdrop/test_eval_devce_cpu.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

import compare_devce
import eval_devce as ev
from olmo_core.nn.attention import AttentionBackendName, SlidingWindowAttentionConfig
from olmo_core.nn.transformer import TransformerConfig

L = 4096 + 512  # two bins: 0-4k, 4k-4k (clipped)


def _tiny(vocab_size):
    cfg = TransformerConfig.llama_like(
        d_model=64, vocab_size=vocab_size, n_layers=4, n_heads=4, block_name="reordered_norm",
        qk_norm=True, rope_theta=500_000, attn_backend=AttentionBackendName.torch,
        sliding_window=SlidingWindowAttentionConfig(
            force_full_attention_on_first_layer=False, force_full_attention_on_last_layer=True,
            pattern=[64, 64, 64, -1]),
    )
    return cfg


def test_unknown_family_raises():
    with pytest.raises(SystemExit, match="unknown --family"):
        ev.build_cfg("qwen3_5", None)


def test_real_config_yarn_layers_match_stage3():
    cfg = ev.build_cfg("olmo3_7b", 8.0)
    assert cfg.vocab_size == 100352 and cfg.n_layers == 32
    assert sorted(cfg.block_overrides) == list(range(3, 32, 4))
    sc = cfg.block_overrides[3].sequence_mixer.rope.scaling
    assert (sc.factor, sc.beta_fast, sc.beta_slow, sc.old_context_len) == (8.0, 32, 1, 8192)
    assert cfg.block.sequence_mixer.sliding_window.pattern == [4096, 4096, 4096, -1]
    assert cfg.block.sequence_mixer.rope.scaling is None  # SWA layers keep plain RoPE
    assert ev.build_cfg("olmo3_7b", None).block_overrides is None
    # same layer set + params as the converted stage-3 checkpoint, when it is visible from here
    p = "/net/mooney/data/prasann/olmo3_cptdrop/base/stage3-step11921/config.json"
    if os.path.isfile(p):
        saved = ev._yarn_by_layer(json.load(open(p))["model"])
        assert saved == {str(k): 8.0 for k in range(3, 32, 4)}


def test_parse_yarn():
    assert ev.parse_yarn("none") is None and ev.parse_yarn("4") == 4.0
    with pytest.raises(Exception):
        ev.parse_yarn("0.5")


def test_bins():
    assert [b[0] for b in ev.bins_for(32768)] == ["0k-4k", "4k-8k", "8k-16k", "16k-32k"]
    assert ev.bins_for(4096) == [("0k-4k", 0, 4096)]


def _write_shard(d, rows, masks, tok="allenai/dolma2-tokenizer"):
    os.makedirs(d, exist_ok=True)
    rows.astype(np.uint32).tofile(f"{d}/token_ids_part_0000.npy")  # raw, like the builder
    masks.astype(bool).tofile(f"{d}/labels_mask_0000.npy")
    json.dump({"dtype": "uint32", "mask_dtype": "bool", "num_instances": len(rows), "tokenizer": tok,
               "max_example_len": rows.shape[1], "min_example_len": rows.shape[1], "source": "test"},
              open(f"{d}/metadata.json", "w"))


def test_row_ce_matches_model_loss_and_end_to_end(tmp_path):
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(0)
    ev.FAMILIES["tiny"] = (lambda v, b: _tiny(v), 1024, "allenai/dolma2-tokenizer")
    cfg = ev.build_cfg("tiny", 4.0, "torch")
    assert sorted(cfg.block_overrides) == [3]
    model = cfg.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))
    model.eval()
    g = np.random.default_rng(0)
    rows = g.integers(0, 1000, size=(3, L))
    masks = np.ones_like(rows, dtype=bool)
    masks[1, 100:300] = False
    masks[:, 0] = False
    # reference: the model's own loss with olmo-core pre-shifted labels
    x = torch.from_numpy(rows[1])
    m = torch.from_numpy(masks[1])
    ce = ev.row_token_ce(model, x, m)
    lab = torch.full((1, L), -100)
    lab[0, :-1] = torch.where(m[1:], x[1:], torch.full_like(x[1:], -100))
    with torch.no_grad():
        ref = model(x[None], labels=lab, loss_reduction="none").loss.reshape(-1)
    ok = lab[0] != -100
    torch.testing.assert_close(ce[1:][ok[:-1]], ref[ok].float(), rtol=1e-4, atol=1e-4)
    assert torch.isnan(ce[0]) and torch.isnan(ce[100:300]).all()

    # end to end through the distcp load and the JSON
    ck = tmp_path / "ck"
    save_model_and_optim_state(str(ck / "model_and_optim"), model)
    shard = tmp_path / "dev"
    _write_shard(str(shard), rows, masks)
    try:
        out = ev.main(["--ckpt", str(ck), "--dev", str(shard), "--yarn-factor", "4", "--family", "tiny",
                       "--attn-backend", "torch", "--out", str(tmp_path / "a.json"), "--label", "a"])
        out2 = ev.main(["--ckpt", str(ck), "--dev", str(shard), "--yarn-factor", "none", "--family", "tiny",
                        "--attn-backend", "torch", "--out", str(tmp_path / "b.json"), "--label", "b"])
    finally:
        del ev.FAMILIES["tiny"]
    assert out["eval_size"] == 3
    assert list(out["summary"]["bins"]) == ["0k-4k", "4k-4k"]
    tot = out["summary"]["overall"]["tokens"]
    assert tot == int(masks[:, 1:].sum())
    assert tot == sum(v["tokens"] for v in out["summary"]["bins"].values())
    # row 1 scored by row_token_ce above with the same weights -> same overall row CE
    r1 = np.nansum(ce.numpy().astype(np.float64)) / np.sum(~np.isnan(ce.numpy()))
    assert abs(sum(out["per_row"]["loss_sum"][1]) / sum(out["per_row"]["tokens"][1]) - r1) < 1e-4
    # yarn only touches the full-attention layer, positions < old_context_len barely move; the
    # comparison must still be exactly paired
    tab = compare_devce.paired(compare_devce.load(str(tmp_path / "a.json")), compare_devce.load(str(tmp_path / "b.json")))
    assert [r["bin"] for r in tab] == ["overall", "0k-4k", "4k-4k"]
    assert all(np.isfinite(r["paired_se"]) for r in tab)
    same = compare_devce.paired(compare_devce.load(str(tmp_path / "a.json")), compare_devce.load(str(tmp_path / "a.json")))
    assert all(r["delta"] == 0 and r["paired_se"] == 0 for r in same)


def test_shard_tokenizer_mismatch_raises(tmp_path):
    rows = np.zeros((2, 16), dtype=np.int64)
    _write_shard(str(tmp_path), rows, np.ones_like(rows, dtype=bool), tok="Qwen/Qwen3.5-0.8B")
    with pytest.raises(SystemExit, match="tokenizer"):
        ev.load_shard(str(tmp_path), None, "olmo3_7b")


def test_ctx_gain_matches_manual_and_merges(tmp_path):
    import merge_devce
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(1)
    ev.FAMILIES["tiny"] = (lambda v, b: _tiny(v), 1024, "allenai/dolma2-tokenizer")
    try:
        cfg = ev.build_cfg("tiny", 4.0, "torch")
        model = cfg.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))
        model.eval()
        g = np.random.default_rng(1)
        rows = g.integers(0, 1000, size=(4, L))
        masks = np.ones_like(rows, dtype=bool)
        masks[:, 0] = False
        masks[2, L - 100:] = False  # unscored tail targets drop out of both sums
        W = 512
        sc = ev.score_rows(model, rows, masks, "cpu", ctx_window=W)
        for i in (0, 2):
            x, m = torch.from_numpy(rows[i]), torch.from_numpy(masks[i])
            full = ev.row_token_ce(model, x, m).numpy().astype(np.float64)[L - W // 2:]
            win = ev.row_token_ce(model, x[L - W:], m[L - W:]).numpy().astype(np.float64)[W // 2:]
            ok = ~np.isnan(full)
            assert sc["ctx"][i, 2] == ok.sum() == (W // 2 - (100 if i == 2 else 0))
            assert abs(sc["ctx"][i, 0] - full[ok].sum()) < 1e-6 and abs(sc["ctx"][i, 1] - win[ok].sum()) < 1e-6
        s = ev.summarize_ctx(sc["ctx"], W)
        assert s["tokens"] == int(sc["ctx"][:, 2].sum()) and np.isfinite(s["gain_row_se"])
        with pytest.raises(SystemExit):
            ev.score_rows(model, rows, masks, "cpu", ctx_window=L)
        # chunked eval + merge reproduces the single-run ctx_gain
        ck = tmp_path / "ck"
        save_model_and_optim_state(str(ck / "model_and_optim"), model)
        shard = tmp_path / "dev"
        _write_shard(str(shard), rows, masks)
        base = ["--ckpt", str(ck), "--dev", str(shard), "--yarn-factor", "4", "--family", "tiny",
                "--attn-backend", "torch", "--ctx-window", str(W)]
        whole = ev.main(base + ["--out", str(tmp_path / "w.json")])
        ev.main(base + ["--rows", "2", "--out", str(tmp_path / "c0.json")])
        ev.main(base + ["--row-start", "2", "--rows", "2", "--out", str(tmp_path / "c1.json")])
    finally:
        del ev.FAMILIES["tiny"]
    import sys
    old = sys.argv
    sys.argv = ["merge_devce.py", "--out", str(tmp_path / "m.json"), str(tmp_path / "c1.json"), str(tmp_path / "c0.json")]
    try:
        merge_devce.main()
    finally:
        sys.argv = old
    m = json.load(open(tmp_path / "m.json"))
    assert abs(m["summary"]["ctx_gain"]["gain"] - whole["summary"]["ctx_gain"]["gain"]) < 1e-5


def test_early_lora_reinstalled_before_load(tmp_path):
    """A checkpoint with an early_lora block in config.json is scored WITH its adapters: the eval re-installs them
    before the load (a nonzero lora_B changes the logits vs the base)."""
    early_lora = pytest.importorskip("olmo_core.nn.early_lora")
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(2)
    ev.FAMILIES["tiny"] = (lambda v, b: _tiny(v), 1024, "allenai/dolma2-tokenizer")
    try:
        cfg = ev.build_cfg("tiny", 4.0, "torch")
        model = cfg.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))
        early_lora.install_early_lora(model, 2, 4, 32.0, check_calls=False)
        with torch.no_grad():
            for m in model._early_lora["modules"]:
                m.lora_B.normal_(0, 0.05)
        model.eval()
        ck = tmp_path / "ck"
        save_model_and_optim_state(str(ck / "model_and_optim"), model)
        json.dump({"model": {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "vocab_size": cfg.vocab_size},
                   "early_lora": {"n_blocks": 2, "rank": 4, "scale": 32.0}}, open(ck / "config.json", "w"))
        x = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            ref = model(x)
        m2, _ = ev.load_model(str(ck), cfg, "cpu", 64)
        with torch.no_grad():
            out = m2(x)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        base = cfg.build(init_device="cpu")
        assert ev.reinstall_early_lora(str(tmp_path), base) is None  # no config.json -> no-op
    finally:
        del ev.FAMILIES["tiny"]


def test_repeat_category_and_bins():
    ids = [5, 6, 7, 8, 9, 5, 6, 7, 8] + [100 + i for i in range(5000)] + [5, 6, 7, 8]
    cat = ev.repeat_category(ids)
    assert cat[:8].tolist() == [0] * 8
    assert cat[8] == 1  # 4-gram 5,6,7,8 last ended at index 3, 5 back
    assert cat[-1] == 2  # same 4-gram, last ended at index 8, > 4096 back
    assert int((cat == 2).sum()) == 1 and int((cat == 1).sum()) == 1
    # summarize_repeat: per-type CE = sum / count
    rs = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 0.0]])
    rc = np.array([[2, 4, 3], [2, 4, 0]])
    s = ev.summarize_repeat(rs, rc)
    assert abs(s["novel"]["ce"] - 0.5) < 1e-12 and abs(s["near"]["ce"] - 0.5) < 1e-12 and abs(s["far"]["ce"] - 1.0) < 1e-12


def test_all_local_train_window_config():
    """--olmo3-train-window 4096: every layer windowed (the 8 full-attention ones too), YaRN still on those 8."""
    import sys as _s
    _s.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "recipes", "train", "research_trainer"))
    from olmo3_configs import olmo3_7B_ctc_swa

    native = olmo3_7B_ctc_swa(vocab_size=100352, yarn_factor=4.0)
    local = olmo3_7B_ctc_swa(vocab_size=100352, yarn_factor=4.0, train_window=4096)
    assert sorted(native.block_overrides) == sorted(local.block_overrides) == [3, 7, 11, 15, 19, 23, 27, 31]

    def win(cfg, i):
        blk = (cfg.block_overrides or {}).get(i, cfg.block)
        sw = blk.sequence_mixer.sliding_window
        return -1 if sw is None or not sw.should_use_swa(i, cfg.n_layers) else sw._get_window_size(i, cfg.n_layers)

    assert [win(native, i) for i in range(32)] == [(-1 if i % 4 == 3 else 4096) for i in range(32)]
    assert [win(local, i) for i in range(32)] == [4096] * 32
    for i in local.block_overrides:
        assert local.block_overrides[i].sequence_mixer.rope.scaling.factor == 4.0


def test_early_lora_rebuilt_from_keys_and_refuses_partial(tmp_path):
    """config.json without an early_lora block (router branch <= 1e5a14c85) but lora keys in the checkpoint: the
    block is rebuilt from the keys + provenance.json; without provenance the eval refuses instead of scoring the
    base silently."""
    early_lora = pytest.importorskip("olmo_core.nn.early_lora")
    from olmo_core.distributed.checkpoint import save_model_and_optim_state

    torch.manual_seed(3)
    ev.FAMILIES["tiny"] = (lambda v, b: _tiny(v), 1024, "allenai/dolma2-tokenizer")
    try:
        cfg = ev.build_cfg("tiny", 4.0, "torch")
        model = cfg.build(init_device="cpu")
        model.init_weights(device=torch.device("cpu"))
        early_lora.install_early_lora(model, 2, 4, 32.0, check_calls=False)
        with torch.no_grad():
            for m in model._early_lora["modules"]:
                m.lora_B.normal_(0, 0.05)
        model.eval()
        ck = tmp_path / "ck"
        save_model_and_optim_state(str(ck / "model_and_optim"), model)
        json.dump({"model": {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "vocab_size": cfg.vocab_size},
                   "early_lora": None}, open(ck / "config.json", "w"))
        with pytest.raises(SystemExit):
            ev.load_model(str(ck), cfg, "cpu", 64)
        json.dump({"args": {"st_otr_early_lora": 4, "st_otr_early_lora_scale": 32.0}}, open(ck / "provenance.json", "w"))
        x = torch.randint(0, 1000, (1, 64))
        with torch.no_grad():
            ref = model(x)
        m2, _ = ev.load_model(str(ck), cfg, "cpu", 64)
        with torch.no_grad():
            torch.testing.assert_close(m2(x), ref, rtol=1e-5, atol=1e-5)
    finally:
        del ev.FAMILIES["tiny"]
