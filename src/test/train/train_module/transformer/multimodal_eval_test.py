import contextlib

import torch

from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
)


def test_multimodal_eval_uses_explicit_labels_and_summed_weighted_loss():
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module._cp_config = None
    train_module._tp_config = None
    train_module._pp_config = None
    train_module.response_logits_only = True
    train_module.label_ignore_index = -100

    class ModelPart:
        def __init__(self):
            self.eval_calls = 0
            self.reset_calls = 0
            self.lm = self

        @staticmethod
        def routed_blocks():
            yield from ()

        def eval(self):
            self.eval_calls += 1

        def reset_auxiliary_metrics(self):
            self.reset_calls += 1

    model_part = ModelPart()
    object.__setattr__(train_module, "model_parts", [model_part])
    captured = {}

    def model_forward(input_ids, labels=None, **kwargs):
        captured.update(input_ids=input_ids, labels=labels, kwargs=kwargs)
        loss = torch.tensor(3.0, requires_grad=True)
        return LMOutputWithLoss(None, loss, loss, None)

    train_module.model_forward_no_pipeline = model_forward
    explicit_labels = torch.tensor([[7, 8, -100]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": explicit_labels,
        "loss_masks": torch.tensor([[0.0, 1.0, 0.0]]),
        "router_token_mask": torch.tensor([[True, True, False]]),
    }

    output = train_module.eval_batch(batch, labels=torch.tensor([[2, 3, -100]]))

    assert output.ce_loss.item() == 3.0
    assert not output.loss.requires_grad
    assert not output.ce_loss.requires_grad
    assert captured["labels"] is explicit_labels
    assert captured["kwargs"]["loss_reduction"] == "sum"
    assert captured["kwargs"]["return_logits"] is False
    assert captured["kwargs"]["response_logits_only"] is True
    assert batch["input_ids"].shape == (1, 3)
    assert batch["labels"] is explicit_labels
    assert model_part.eval_calls == 1
    assert model_part.reset_calls == 1


def _tiny_text_eval_module(model):
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    train_module._cp_config = None
    train_module._tp_config = None
    train_module._pp_config = None
    train_module.response_logits_only = True
    train_module.label_ignore_index = -100
    train_module.trim_microbatch_image_padding = False
    train_module.device = torch.device("cpu")
    object.__setattr__(train_module, "model_parts", [model])
    return train_module


def test_multimodal_text_eval_runs_the_plain_lm_loss_on_the_real_model():
    """A downstream (text-only) batch has no ``loss_masks``: the wrapper must defer to the
    language model's per-token loss with ``loss_reduction="none"`` and return full logits."""
    from test.nn.vision.multimodal_olmo_ddp_test import _model, _text_batch

    model = _model()
    train_module = _tiny_text_eval_module(model)
    input_ids, labels, _ = _text_batch()
    batch = {"input_ids": input_ids.clone()}

    output = train_module.eval_batch(batch, labels=labels)

    assert isinstance(output, LMOutputWithLoss)
    assert output.logits is not None and output.logits.shape == (2, 8, 64)
    assert output.loss.shape == (2, 8) and output.ce_loss.shape == (2, 8)
    assert not output.logits.requires_grad and not output.loss.requires_grad
    with torch.no_grad():
        logits = model(input_ids)
    expected = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, 64), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape(2, 8)
    # The OLMoDDP LM head runs its loss path in BF16 activations.
    torch.testing.assert_close(output.ce_loss.float(), expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(output.logits.float(), logits.float(), rtol=2e-2, atol=2e-2)
    # Ignored positions contribute no loss, like the LM-only path.
    assert torch.equal(
        output.ce_loss[labels == -100], torch.zeros_like(output.ce_loss[labels == -100])
    )


def test_multimodal_olmo_ddp_text_eval_forces_eager_with_grad_enabled(monkeypatch):
    train_module = object.__new__(MultimodalOLMoDDPTrainModule)
    events = []

    @contextlib.contextmanager
    def set_stance(stance):
        events.append(("enter", stance))
        try:
            yield
        finally:
            events.append(("exit", stance))

    monkeypatch.setattr(torch.compiler, "set_stance", set_stance)

    with torch.no_grad():
        with train_module._eval_batch_context():
            assert torch.is_grad_enabled()

    assert events == [("enter", "force_eager"), ("exit", "force_eager")]
