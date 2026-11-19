"""Dense long-context training step benchmark (Qwen3.5 hybrid, olmo_core), for the 1M-token feasibility study.

One full training step = forward over the whole row (every token attends over its full causal context), backward,
AdamW step. Same model build / FSDP(HSDP) / Ulysses CP / activation-checkpointing path as the research trainer
(``recipes/train/research_trainer/train_ctc_suite.py``: ``parallelize_model`` with bf16 params, fp32 reduce,
fused-linear CE), without its data pipeline, so lengths / CP degree / memory options can be swept quickly.

    torchrun --nproc_per_node 8 dense_step_bench.py --scale 0.8b --seq-len 1048576 --cp 8 --ac full \\
        --gdn-safe --steps 3 --out res.json [--init /path/to/base/model_and_optim] [--lora-rank 16]

Rows: random tokens (memory / speed are data-independent) or ``--rows-npz heldout_lc.npz --rows-set lc1m`` (real
contradiction rows; labels = the row's answer mask), which also makes the loss meaningful.
``--check-grads`` (short T): runs the step twice, plain vs with the memory options, and reports loss / grad diffs.

Prints one JSON line per step (rank-max peak GiB allocated / reserved, seconds) and writes a summary to ``--out``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist


def log0(*a):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*a, flush=True)


SCALES = {"0.8b": "qwen3_5_0_8B", "2b": "qwen3_5_2B", "4b": "qwen3_5_4B", "9b": "qwen3_5_9B"}


def analytic_pf(cfg_model, T: int, n_label: int, lora: bool) -> dict:
    """Training FLOPs of one row (PF), counted as 2 x MACs: linears 6*N*T (LoRA: 4*N*T, no weight grads), causal
    full attention 2*T^2*H*D fwd per layer x3 for fwd+bwd, LM head only at labelled positions unless under CP
    (the fused CE then runs over every position), GDN chunk kernel ~ approx 2*T*Hv*(4*K*V) fwd per layer x3."""
    d = cfg_model
    nl = d["n_layers"]
    n_attn = len([i for i in range(nl) if (i + 1) % 4 == 0])
    n_gdn = nl - n_attn
    lin = d["block_linear_params"]
    mult = 4 if lora else 6
    linear = mult * lin * T
    attn = n_attn * 3 * 2 * T * T * d["n_heads"] * d["head_dim"] / 2 * 2  # causal half of 4*T^2*H*D fwd, x3
    gdn = n_gdn * 3 * 2 * T * d["n_v_heads"] * 4 * 128 * 128
    head_tokens = n_label if d.get("sparse_head") else T
    head = mult * d["vocab"] * d["d_model"] * head_tokens
    tot = linear + attn + gdn + head
    return dict(total_pf=tot / 1e15, linear_pf=linear / 1e15, attn_pf=attn / 1e15, gdn_pf=gdn / 1e15,
                head_pf=head / 1e15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="0.8b", choices=list(SCALES))
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--cp", type=int, default=1, help="Ulysses CP degree (1 = off)")
    ap.add_argument("--ac", default="full", choices=["none", "full", "selected"],
                    help="activation checkpointing: full = every block; selected = every --ac-interval-th block")
    ap.add_argument("--ac-interval", type=int, default=2)
    ap.add_argument("--lora-rank", type=int, default=0, help="0 = full fine-tune")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--init", default="random", help="'random' or a distcp model_and_optim dir")
    ap.add_argument("--gdn-safe", action="store_true", help="enable_int32_safe_long_seq (exact; needed past 2^31 offsets)")
    ap.add_argument("--gdn-segment", type=int, default=0, help="--gdn-segment-tokens (training GDN segments)")
    ap.add_argument("--ffn-chunk", type=int, default=0, help="--ffn-chunk-tokens (sequence-chunked FFN + checkpoint)")
    ap.add_argument("--sparse-head", action="store_true",
                    help="LM head only at labelled positions (no-CP: the trainer's --sparse-loss-positions)")
    ap.add_argument("--label-tokens", type=int, default=40, help="random rows: supervised tail length")
    ap.add_argument("--rows-npz", default="")
    ap.add_argument("--rows-set", default="lc1m")
    ap.add_argument("--row-index", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--attn-backend", default="flash_2")
    ap.add_argument("--offload-ckpt", action="store_true",
                    help="keep the activation-checkpoint block inputs in pinned CPU memory (torch save_on_cpu)")
    ap.add_argument("--check-grads", action="store_true",
                    help="run plain (no gdn-safe/ffn-chunk/gdn-segment/offload) then with options; compare loss + grads")
    ap.add_argument("--mem-snapshot", default="", help="write a torch.cuda memory snapshot pickle of step 1 here")
    ap.add_argument("--forward-only", action="store_true",
                    help="no backward / optimizer: answer CE of the row under no_grad (base-model validity checks)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    from olmo_core.distributed.parallel import (
        DataParallelType,
        build_world_mesh,
    )
    from olmo_core.nn.attention import AttentionBackendName
    from olmo_core.nn.lm_head import LMLossImplementation
    from olmo_core.nn.transformer import (
        TransformerActivationCheckpointingMode,
        TransformerConfig,
        TransformerDataParallelWrappingStrategy,
    )
    from olmo_core.config import DType
    from olmo_core.train.train_module.transformer.common import parallelize_model
    from olmo_core.train.train_module.transformer.config import (
        TransformerActivationCheckpointingConfig,
        TransformerContextParallelConfig,
        TransformerDataParallelConfig,
    )

    if world % a.cp:
        raise SystemExit("world size not divisible by --cp")
    # ---- data ----
    T = a.seq_len
    if a.rows_npz:
        import numpy as np
        z = np.load(a.rows_npz)
        idx = [i for i in range(len(z["sets"])) if str(z["sets"][i]) == a.rows_set]
        i = idx[a.row_index % len(idx)]
        ids = torch.from_numpy(z["ids"][z["offsets"][i]:z["offsets"][i + 1]].astype("int64"))
        mask = torch.from_numpy(z["mask"][z["offsets"][i]:z["offsets"][i + 1]].astype(bool))
        if T and ids.numel() > T:  # keep the tail (the question + answer are at the end)
            ids, mask = ids[-T:], mask[-T:]
        T = ids.numel()
    else:
        g = torch.Generator().manual_seed(1234)
        ids = torch.randint(0, 150000, (T,), generator=g)
        mask = torch.zeros(T, dtype=torch.bool)
        mask[-a.label_tokens:] = True
    pad_to = 64 * a.cp
    if T % pad_to:  # pad on the left with masked tokens so CP shards are even
        p = pad_to - T % pad_to
        ids = torch.cat([torch.zeros(p, dtype=ids.dtype), ids])
        mask = torch.cat([torch.zeros(p, dtype=torch.bool), mask])
        T = ids.numel()
    # next-token labels: label[t] = ids[t+1] where mask[t+1]
    labels = torch.full_like(ids, -100)
    labels[:-1] = torch.where(mask[1:], ids[1:], torch.full_like(ids[1:], -100))
    n_label = int((labels != -100).sum())
    input_ids = ids.unsqueeze(0).to(dev)
    labels = labels.unsqueeze(0).to(dev)

    # ---- model ----
    vocab = 248320
    cfg = getattr(TransformerConfig, SCALES[a.scale])(vocab_size=vocab, attn_backend=AttentionBackendName(a.attn_backend))
    cfg.lm_head.loss_implementation = LMLossImplementation.fused_linear
    if a.sparse_head and a.cp == 1:
        cfg.lm_head.sparse_loss_positions = True

    def build(options: bool):
        model = cfg.build(init_device="meta")
        if a.lora_rank:
            from olmo_core.nn.early_lora import install_early_lora
            install_early_lora(model, len(model.blocks), a.lora_rank, 32.0)
        if options:
            if a.gdn_safe:
                from olmo_core.nn.ffn_chunk import enable_int32_safe_long_seq
                enable_int32_safe_long_seq(model)
            if a.gdn_segment:
                from olmo_core.nn.ffn_chunk import install_gdn_segments
                install_gdn_segments(model, a.gdn_segment)
            if a.ffn_chunk:
                from olmo_core.nn.ffn_chunk import install_ffn_chunk
                install_ffn_chunk(model, a.ffn_chunk)
        dp_cfg = TransformerDataParallelConfig(
            name=DataParallelType.hsdp if a.cp > 1 else DataParallelType.fsdp,
            param_dtype=DType.bfloat16, reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        )
        cp_cfg = TransformerContextParallelConfig.ulysses(degree=a.cp) if a.cp > 1 else None
        ac_cfg = None
        if a.ac == "full":
            ac_cfg = TransformerActivationCheckpointingConfig(mode=TransformerActivationCheckpointingMode.full)
        elif a.ac == "selected":
            ac_cfg = TransformerActivationCheckpointingConfig(
                mode=TransformerActivationCheckpointingMode.selected_blocks, block_interval=a.ac_interval)
        mesh = build_world_mesh(dp=dp_cfg, cp=cp_cfg, device_type="cuda")
        model = parallelize_model(model, world_mesh=mesh, device=dev, max_sequence_length=T,
                                  rank_microbatch_size=T, compile_model=False, dp_config=dp_cfg, cp_config=cp_cfg,
                                  ac_config=ac_cfg)
        if a.init != "random":
            if a.lora_rank:  # the trainer's tolerant load: base keys only, refuses missing backbone keys
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
                from recipes.train.variants import tolerant_base_load
                tolerant_base_load(a.init, model, "/tmp/dense_bench_save")
            else:
                from olmo_core.distributed.checkpoint import load_model_and_optim_state
                load_model_and_optim_state(a.init, model)
        if a.lora_rank:
            from olmo_core.nn.early_lora import _reset_lora
            torch.manual_seed(0)
            for m in model._early_lora["modules"]:
                _reset_lora(m)
        model.train()
        return model

    def info_of(model):
        b = model.blocks["0"] if "0" in model.blocks else list(model.blocks.values())[0]
        lin = 0
        for blk in model.blocks.values():
            for mod in blk.modules():
                if isinstance(mod, torch.nn.Linear):
                    lin += mod.in_features * mod.out_features
        att = [blk for blk in model.blocks.values() if hasattr(blk.attention, "n_kv_heads")][0].attention
        gdn = b.attention
        return dict(n_layers=len(model.blocks), block_linear_params=lin, n_heads=att.n_heads,
                    head_dim=getattr(att, "head_dim", 256), n_v_heads=gdn.n_v_heads, vocab=vocab, d_model=cfg.d_model,
                    sparse_head=bool(a.sparse_head and a.cp == 1))

    def run(model, steps, tag):
        params = [p for p in model.parameters() if p.requires_grad]
        n_train = sum(p.numel() for p in params)
        opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
        rows = []
        loss_v = None
        for step in range(1, steps + 1):
            torch.cuda.synchronize(); dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            if a.mem_snapshot and step == 1 and rank == 0:
                torch.cuda.memory._record_memory_history(max_entries=200000)
            t0 = time.time()
            ctx = torch.autograd.graph.save_on_cpu(pin_memory=True) if a.offload_ckpt else _null()
            if a.forward_only:
                with torch.no_grad():
                    out = model(input_ids, labels=labels, loss_reduction="sum")
                loss = out.loss if hasattr(out, "loss") else out[0]
            else:
                with ctx:
                    out = model(input_ids, labels=labels, loss_reduction="sum")
                loss = out.loss if hasattr(out, "loss") else out[0]
                (loss / max(1, n_label)).backward()
            t_fb = time.time() - t0
            # loss for reporting: sum over CP ranks / labelled tokens
            lv = loss.detach().float().clone()
            if a.cp > 1:
                dist.all_reduce(lv)
                lv /= (world // a.cp)
            else:
                dist.all_reduce(lv)
                lv /= world
            loss_v = (lv / max(1, n_label)).item()
            if step == steps and a.check_grads:
                grads = {n: (p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad).detach().float().cpu()
                         for n, p in model.named_parameters() if p.grad is not None}
            else:
                grads = None
            if not a.forward_only:
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = time.time() - t0
            if a.mem_snapshot and step == 1 and rank == 0:
                torch.cuda.memory._dump_snapshot(a.mem_snapshot)
                torch.cuda.memory._record_memory_history(enabled=None)
            pk = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
                              device=dev)
            dist.all_reduce(pk, op=dist.ReduceOp.MAX)
            r = dict(tag=tag, step=step, loss=round(loss_v, 5), s=round(dt, 2), s_fwd_bwd=round(t_fb, 2),
                     peak_alloc_gib=round(pk[0].item(), 2), peak_reserved_gib=round(pk[1].item(), 2))
            log0(json.dumps(r))
            rows.append(r)
        return rows, grads, n_train

    t_build = time.time()
    res = dict(args=vars(a), T=T, n_label=n_label, world=world, gpu=torch.cuda.get_device_name(dev))
    try:
        if a.check_grads:
            torch.manual_seed(0)
            m0 = build(False)
            rows0, g0, _ = run(m0, 1, "plain")
            del m0; torch.cuda.empty_cache()
            m1 = build(True)
            rows1, g1, ntr = run(m1, 1, "options")
            if rank == 0:
                rel = {n: ((g0[n] - g1[n]).norm() / (g0[n].norm() + 1e-20)).item() for n in g0}
                worst = sorted(rel.items(), key=lambda kv: -kv[1])[:5]
                res["check"] = dict(loss_plain=rows0[0]["loss"], loss_options=rows1[0]["loss"],
                                    max_grad_rel=max(rel.values()), worst=worst,
                                    plain=rows0[0], options=rows1[0])
                log0(json.dumps(res["check"]))
        else:
            model = build(True)
            res["build_s"] = round(time.time() - t_build, 1)
            res["model"] = info_of(model)
            res["pf"] = analytic_pf(res["model"], T, n_label, bool(a.lora_rank))
            rows, _, ntr = run(model, a.steps, "bench")
            res["steps"] = rows
            res["trainable_params"] = ntr
            st = rows[1:] if len(rows) > 1 else rows
            res["s_per_step"] = round(sum(r["s"] for r in st) / len(st), 2)
            res["peak_alloc_gib"] = max(r["peak_alloc_gib"] for r in rows)
            res["peak_reserved_gib"] = max(r["peak_reserved_gib"] for r in rows)
            res["ok"] = all(math.isfinite(r["loss"]) for r in rows)
            log0("SUMMARY " + json.dumps({k: res[k] for k in ("T", "world", "s_per_step", "peak_alloc_gib",
                                                              "peak_reserved_gib", "ok", "pf")}))
    except torch.cuda.OutOfMemoryError as e:
        res["oom"] = str(e)[:400]
        log0("OOM " + str(e)[:400])
    if rank == 0 and a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    dist.barrier()
    dist.destroy_process_group()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
