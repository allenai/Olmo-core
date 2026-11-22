from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from olmo_core.data.utils import get_labels
from olmo_core.eval.lm_evaluator import LMEvaluator
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train.callbacks import EvaluatorCallback


class _LogitsTrainModule:
    """Mimics ``TransformerTrainModule.eval_batch()`` with fixed logits."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def eval_batch(self, batch: Dict[str, Any], labels: Optional[torch.Tensor] = None):
        batch.pop("input_ids")
        assert labels is not None
        logits = self.logits.to(labels.device)
        ce_loss = F.cross_entropy(
            logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
        )
        return LMOutputWithLoss(logits, ce_loss.sum(), ce_loss, None)


def test_lm_eval_ce_loss_matches_next_token_loss_on_padded_instances():
    torch.manual_seed(0)
    vocab_size, seq_len = 16, 8
    input_ids = torch.randint(2, vocab_size, (2, seq_len))
    # Two padded instances with 3 and 6 real tokens, like NumpyPaddedFSLDataset produces.
    label_mask = torch.tensor([[True] * 3 + [False] * 5, [True] * 6 + [False] * 2])
    input_ids[~label_mask] = 1
    logits = torch.randn(2, seq_len, vocab_size)
    batch = {"input_ids": input_ids, "label_mask": label_mask, "metadata": [{"label": "val"}] * 2}

    # The expected loss is the mean next-token CE over real target tokens only.
    labels = get_labels(batch)
    expected = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100)

    evaluator = LMEvaluator(name="lm", batches=[batch], labels=["val"], device=torch.device("cpu"))
    recorded: Dict[str, torch.Tensor] = {}
    trainer = SimpleNamespace(
        dp_process_group=None,
        train_module=_LogitsTrainModule(logits),
        record_metric=lambda name, value, **kwargs: recorded.__setitem__(name, value),
    )
    callback = EvaluatorCallback(evaluators=[evaluator])
    callback.trainer = trainer  # type: ignore[assignment]

    callback.perform_eval()

    torch.testing.assert_close(recorded["eval/lm/val/CE loss"].cpu(), expected)
