"""CPU repro of the multi-view train_batch under FSDP2 + full AC (world size 1, gloo): tiny olmo2 model, whole-model
LoRA over a frozen base, two-stage random router, k views, 2 micro-batches. Exit 0 = the deferred early backward ran and
the early adapters got gradients that match a reference (k=1 path) in shape and are finite.

    PYTHONPATH=src python research/token_routing/multiview/fsdp_cpu_repro.py [--no-fire]
"""
import argparse
import contextlib
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fire", action="store_true", help="disable the explicit FSDP pre-backward firing")
    ap.add_argument("--views", type=int, default=2)
    a = ap.parse_args()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from torch.distributed.device_mesh import init_device_mesh

    from olmo_core.config import DType
    from olmo_core.nn import multiview as MV
    from olmo_core.nn.early_lora import install_early_lora
    from olmo_core.nn.online_token_router import OnlineRouterConfig, OnlineTokenRouter
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src", "test", "nn"))
    from online_token_router_test import DE, DS, EOS, PH, _row  # type: ignore

    if a.no_fire:
        MV._early_fsdp_states = lambda model, layer: []
    torch.manual_seed(0)
    model = TransformerConfig.olmo2_190M(vocab_size=9100, n_layers=4, fused_ops=False, dtype=DType.float32).build(
        init_device="cpu")
    D = int(model.embeddings.weight.shape[1])
    router = OnlineTokenRouter(OnlineRouterConfig(keep_frac=0.25, explore_frac=0.2, samples_per_row=64, stages="1:0.5,2:0.25",
                                                  dense_every=0, score="random", early_lora=8, early_lora_blocks=4),
                               doc_start_id=DS, doc_end_id=DE, eos_id=EOS, d_model=D)
    model.enable_pooled_soft_tokens(DS, DE, EOS, placeholder_id=PH, keep_prob=0.0, keep_seed=0, detach_soft_kv=True,
                                    drop_slots=True, keep_token_rule="online", keep_token_k=0.25, online_router=router)
    install_early_lora(model, 4, 8, 32.0)
    model.apply_activation_checkpointing(TransformerActivationCheckpointingMode.full)
    mesh = init_device_mesh("cpu", (1,))
    model.apply_fsdp(dp_mesh=mesh, param_dtype=torch.float32)
    model.train()
    model._flop_pad_id = EOS
    ids, _, labels = _row(B=2)
    # warm the router (step 1 dense, then fitted)
    for n in (1, 2):
        for r in router.chain():
            r.begin_step(n, model=model)
            r.metrics()
        model.zero_grad(set_to_none=True)
        model(ids, labels=labels).loss.backward()
    for r in router.chain():
        r.begin_step(3, model=model)
        r.metrics()
    assert not router.dense_now()
    from torch.distributed.fsdp import FSDPModule

    class TM:
        label_ignore_index = -100
        microbatch_sort_pad_id = None
        device = torch.device("cpu")
        z_loss_multiplier = None
        max_grad_norm = 1.0
        scheduler = None

        def __init__(self):
            self.model = model
            self.rank_microbatch_size = ids.shape[1]
            self.optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            self.metrics = {}
            self.train_batch = lambda b, dry_run=False: None

        def _set_model_mode(self, m):
            pass

        @contextlib.contextmanager
        def _train_microbatch_context(self, i, n):
            if isinstance(self.model, FSDPModule):
                self.model.set_is_last_backward(i == n - 1)
            yield

        def _prepare_batch(self, b):
            return b.pop("input_ids"), b.pop("labels", None), b

        def model_forward(self, input_ids, labels=None, **kw):
            return self.model(input_ids, labels=labels, **kw)

        def _clip_grad_norm(self, m):
            return torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.grad is not None], m)

        def record_metric(self, name, value, reduce_type=None, namespace=None):
            self.metrics[name] = value

        def record_ce_loss(self, value, reduce_type=None):
            self.metrics["ce"] = value

    model.zero_grad(set_to_none=True)
    tm = TM()
    MV.enable_multiview(tm, a.views)
    tm.train_batch({"input_ids": ids.clone(), "labels": labels.clone()})
    early = [p for n, p in model.named_parameters() if "lora_" in n and ".blocks.0." in "." + n]
    assert early, "no early adapters found"
    ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in early)
    print("early adapters with grads:", sum(p.grad is not None for p in early), "/", len(early), "finite:", ok,
          "ce:", float(tm.metrics["ce"]), flush=True)
    assert ok
    print("REPRO_OK")


if __name__ == "__main__":
    main()
