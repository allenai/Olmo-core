import json
from types import SimpleNamespace

import pytest
import torch

from olmo_core.eval import vision_fast_text as fast
from olmo_core.nn.lm_head import LMOutputWithLoss


def _result():
    return {
        "metrics": {"accuracy": 0.5},
        "batches_per_ep_dp_rank": 1,
        "total_batches_per_ep_dp_rank": 1,
        "instances_per_ep_dp_rank": 2,
    }


def test_fast_defaults_and_complete_task_panel():
    args = fast._parse_args(
        ["--checkpoint", "checkpoint", "--output", "out.json", "--tokenizer", "tokenizer.json"]
    )
    assert (args.ep_degree, args.max_sequence_length, args.rank_batch_size) == (8, 2048, 8192)
    assert (args.pad_token_id, args.eos_token_id) == (100277, 100257)
    assert len(fast.FAST_TASKS) == len(set(fast.FAST_TASKS)) == 26


def test_task_cache_rejects_other_identity_and_incomplete_tasks(tmp_path):
    path = tmp_path / "task.json"
    identity = {"format": "olmo_core_vision_fast_text_v2", "checkpoint": "checkpoint"}
    assert fast._read_task(path, "task", identity) is None
    fast._atomic_json(path, {"identity": identity, "task": "task", "result": _result()})
    assert fast._read_task(path, "task", identity) == _result()
    with pytest.raises(ValueError, match="different checkpoint or protocol"):
        fast._read_task(path, "task", identity | {"checkpoint": "other"})
    bad_result = _result() | {"total_batches_per_ep_dp_rank": 2}
    fast._atomic_json(path, {"identity": identity, "task": "task", "result": bad_result})
    with pytest.raises(ValueError, match="Incomplete task"):
        fast._read_task(path, "task", identity)


def test_complete_result_rejects_legacy_receipts_and_missing_load(tmp_path):
    path = tmp_path / "results.json"
    identity = {"format": "olmo_core_vision_fast_text_v2"}
    payload = {
        "schema_version": 2,
        "identity": identity,
        "native_checkpoint_load": {"complete": True, "load_completed": True},
        "results": {task: _result() for task in fast.FAST_TASKS},
    }
    fast._atomic_json(path, payload)
    fast._validate_complete(path, identity, fast.FAST_TASKS)
    for change in (
        {"schema_version": 1},
        {"results": {}},
        {"native_checkpoint_load": {"complete": True}},
    ):
        fast._atomic_json(path, payload | change)
        with pytest.raises(ValueError):
            fast._validate_complete(path, identity, fast.FAST_TASKS)


def test_complete_batches_and_harness_scores_are_preserved(monkeypatch):
    seen = []
    batch = {"input_ids": torch.tensor([[1, 2, 3], [2, 3, 4]])}
    output = LMOutputWithLoss(torch.zeros(2, 3, 5), None, torch.ones(2, 3), None)

    class Evaluator:
        total_batches = 2

        def __init__(self, **kwargs):
            seen.append(kwargs)

        def reset_metrics(self):
            pass

        def __iter__(self):
            return iter([batch, batch])

        def update_metrics(self, actual_batch, ce_loss, logits):
            assert actual_batch["input_ids"].equal(batch["input_ids"])
            assert ce_loss is output.ce_loss
            assert logits is output.logits

        def compute_metrics(self):
            return {"accuracy": torch.tensor(0.75)}

    monkeypatch.setattr(fast, "DownstreamEvaluator", Evaluator)
    monkeypatch.setattr(fast, "gc_cuda", lambda: None)
    module = SimpleNamespace(
        device=torch.device("cpu"),
        eval_batch_spec="batch_spec",
        dp_process_group="group",
        eval_batch=lambda batch, labels: output,
    )
    result = fast.evaluate_tasks(module, ["task"], "tokenizer", max_batches=None)["task"]
    assert result["metrics"] == {"accuracy": 0.75}
    assert result["batches_per_ep_dp_rank"] == result["total_batches_per_ep_dp_rank"] == 2
    assert result["instances_per_ep_dp_rank"] == 4
    assert seen == [
        {
            "name": "downstream",
            "task": "task",
            "batch_spec": "batch_spec",
            "tokenizer": "tokenizer",
            "device": torch.device("cpu"),
            "dp_process_group": "group",
        }
    ]


def test_rank_zero_cache_errors_are_collective(monkeypatch):
    monkeypatch.setattr(fast, "get_rank", lambda: 0)
    monkeypatch.setattr(fast.dist, "broadcast_object_list", lambda packet, src: None)
    with pytest.raises(RuntimeError, match="JSONDecodeError"):
        fast._rank_zero_call(json.loads, "invalid")
