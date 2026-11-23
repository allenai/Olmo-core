from typing import Dict, List, Optional

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig
from olmo_core.testing import run_distributed_test
from olmo_core.train.train_module.transformer import (
    TransformerDataParallelConfig,
    TransformerTrainModuleConfig,
)

VOCAB_SIZE = 32
SEQ_LEN = 8
NUM_INSTANCES = 4
IGNORE_INDEX = -100


class _StubTrainer:
    """Minimal stand-in: ``train_batch`` only reaches the trainer to record metrics."""

    def __init__(self) -> None:
        self.ce_losses: List[torch.Tensor] = []

    def record_metric(self, *args, **kwargs) -> None:
        del args, kwargs

    def record_ce_loss(self, value, *args, **kwargs) -> None:
        del args, kwargs
        self.ce_losses.append(value.detach().clone())


def _model_config() -> TransformerConfig:
    return TransformerConfig.llama_like(
        d_model=16,
        vocab_size=VOCAB_SIZE,
        n_layers=1,
        n_heads=2,
        feed_forward=FeedForwardConfig(hidden_size=32, bias=False),
        attn_backend=AttentionBackendName.torch,
    )


def _global_batch(zero_loss_tokens_on_last_instance: bool = False) -> Dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (NUM_INSTANCES, SEQ_LEN), generator=generator)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, :-1] = input_ids[:, 1:]
    # Very unequal numbers of loss tokens per instance (as with SFT label masking):
    # instances 0, 1, 2, 3 have 7, 6, 1 and 2 loss tokens.
    labels[1, :1] = IGNORE_INDEX
    labels[2, :6] = IGNORE_INDEX
    labels[3, :5] = IGNORE_INDEX
    if zero_loss_tokens_on_last_instance:
        # With 4 DP ranks, the last rank then has no loss tokens at all.
        labels[3, :] = IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


def _full_grads(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    grads = {}
    for name, param in model.named_parameters():
        assert param.grad is not None, name
        grad = param.grad
        grads[name] = grad.full_tensor() if isinstance(grad, DTensor) else grad
    return grads


def _run_global_token_loss_normalization(
    dp_type: DataParallelType,
    instances_per_microbatch: int,
    zero_loss_tokens_on_last_instance: bool = False,
    shard_degree: Optional[int] = None,
):
    dp_ws = dist.get_world_size()
    rank = dist.get_rank()
    device = torch.device("cpu")

    module = TransformerTrainModuleConfig(
        rank_microbatch_size=instances_per_microbatch * SEQ_LEN,
        max_sequence_length=SEQ_LEN,
        optim=AdamWConfig(),
        dp_config=TransformerDataParallelConfig(name=dp_type, shard_degree=shard_degree),
        label_ignore_index=IGNORE_INDEX,
    ).build(_model_config().build(init_device="cpu"), device=device)
    module._trainer = _StubTrainer()  # type: ignore[assignment]

    # Reference: a non-distributed copy of the same model trained on the whole global batch,
    # with the loss being the mean over all loss tokens in the global batch.
    state_dict = {
        k: v.full_tensor() if isinstance(v, DTensor) else v
        for k, v in module.model.state_dict().items()
    }
    ref_model = _model_config().build(init_device="cpu")
    ref_model.load_state_dict(state_dict)
    global_batch = _global_batch(zero_loss_tokens_on_last_instance)
    ref_logits = ref_model(global_batch["input_ids"])
    assert isinstance(ref_logits, torch.Tensor)
    ref_loss = F.cross_entropy(
        ref_logits.view(-1, VOCAB_SIZE),
        global_batch["labels"].view(-1),
        ignore_index=IGNORE_INDEX,
    )
    ref_loss.backward()
    ref_grads = _full_grads(ref_model)

    # Each DP rank gets a contiguous slice of the global batch.
    n = NUM_INSTANCES // dp_ws
    local_batch = {k: v[rank * n : (rank + 1) * n].clone() for k, v in global_batch.items()}
    module.zero_grads()
    module.train_batch(local_batch)

    grads = _full_grads(module.model)
    assert grads.keys() == ref_grads.keys()
    for name, grad in grads.items():
        torch.testing.assert_close(grad, ref_grads[name], msg=lambda m: f"{name}: {m}")

    # The logged CE loss (averaged over ranks by the trainer) should be the global token mean.
    ce_loss = module.trainer.ce_losses[-1]  # type: ignore[attr-defined]
    dist.all_reduce(ce_loss)
    torch.testing.assert_close(ce_loss / dp_ws, ref_loss.detach())


@pytest.mark.parametrize("dp_type", [DataParallelType.fsdp, DataParallelType.ddp])
@pytest.mark.parametrize(
    "instances_per_microbatch", [pytest.param(1, id="grad-accum"), pytest.param(2, id="no-accum")]
)
def test_train_batch_normalizes_loss_by_global_token_count(
    dp_type: DataParallelType, instances_per_microbatch: int
):
    run_distributed_test(
        _run_global_token_loss_normalization,
        backend="gloo",
        start_method="spawn",
        func_args=(dp_type, instances_per_microbatch),
    )


@pytest.mark.parametrize(
    "dp_type, shard_degree",
    [
        pytest.param(DataParallelType.fsdp, None, id="fsdp"),
        pytest.param(DataParallelType.ddp, None, id="ddp"),
        pytest.param(DataParallelType.hsdp, 2, id="hsdp-2x2"),
    ],
)
@pytest.mark.parametrize(
    "zero_loss_tokens_on_last_instance",
    [pytest.param(False, id="uneven"), pytest.param(True, id="rank-with-no-loss-tokens")],
)
def test_train_batch_normalizes_loss_by_global_token_count_4_ranks(
    dp_type: DataParallelType, shard_degree: Optional[int], zero_loss_tokens_on_last_instance: bool
):
    run_distributed_test(
        _run_global_token_loss_normalization,
        world_size=4,
        backend="gloo",
        start_method="spawn",
        func_args=(dp_type, 1, zero_loss_tokens_on_last_instance, shard_degree),
    )
