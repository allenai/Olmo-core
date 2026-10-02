import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("scipy")  # decoded pointing metrics use scipy.optimize

from olmo_core.config import Config
from olmo_core.data.multimodal.alignment import MultimodalSourceConfig
from olmo_core.eval import vision_decoded as runner
from olmo_core.eval.multimodal_decoding import score_prediction


@pytest.fixture
def saved(tmp_path, monkeypatch):
    checkpoint, panel = tmp_path / "checkpoint", tmp_path / "panel"
    data = {
        "model": {},
        "train_module": {},
        "dataset": {
            "tokenizer": {"identifier": "tokenizer"},
            "tokenizer_revision": "revision",
        },
        "pretraining_checkpoint": "parent",
    }
    for path in (checkpoint, panel):
        path.mkdir()
        (path / ".metadata.json").write_text("{}")
        (path / "config.json").write_text(json.dumps(data))
    blocks = [
        SimpleNamespace(
            routed_experts_router=SimpleNamespace(lb_loss_weight=0.015),
            ep=SimpleNamespace(capacity_factor=value),
        )
        for value in (1.1875, 2)
    ]
    model = SimpleNamespace(
        lm=SimpleNamespace(vocab_size=100352, block=blocks[0], block_overrides={1: blocks[1]})
    )
    model.as_config_dict = lambda: {"capacity": [b.ep.capacity_factor for b in blocks]}
    module = SimpleNamespace(
        max_sequence_length=8192,
        rank_microbatch_size=8192,
        response_logits_only=True,
        ep_config=SimpleNamespace(degree=8),
        freeze_params=["lm.lm_head.w_out.weight"],
        train_embedding_rows=list(range(100278, 100284)),
    )
    module.as_config_dict = lambda: {
        "freeze": module.freeze_params,
        "mb": module.rank_microbatch_size,
    }
    evaluator = SimpleNamespace(
        eval_dataset=SimpleNamespace(
            tokenizer=SimpleNamespace(identifier="tokenizer"),
            tokenizer_revision="revision",
            model_vocab_size=100352,
        ),
        sequence_length=2560,
        rank_batch_size=4,
        examples_per_source=64,
    )
    monkeypatch.setattr(runner.MultimodalLMConfig, "from_dict", lambda value: model)
    monkeypatch.setattr(
        runner.MultimodalOLMoDDPTrainModuleConfig, "from_dict", lambda value: module
    )
    return SimpleNamespace(
        checkpoint=checkpoint,
        panel=panel,
        data=data,
        model=model,
        module=module,
        evaluator=evaluator,
        blocks=blocks,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_adapter_preserves_weights_architecture_and_freeze_surface(saved, legacy):
    metadata = deepcopy(saved.data)
    if legacy:
        metadata.pop("dataset")
        metadata.pop("pretraining_checkpoint")
        metadata["artifacts"] = {
            "tokenizer_id": "tokenizer",
            "tokenizer_revision": "revision",
            "base_checkpoint": "parent",
        }
        (saved.checkpoint / "config.json").write_text(json.dumps(metadata))
    freeze = deepcopy(saved.module.freeze_params)
    rows = deepcopy(saved.module.train_embedding_rows)
    model, module, manifest = runner.checkpoint_configs(
        saved.checkpoint, saved.evaluator, saved.panel, 16
    )
    assert model is saved.model
    assert module.freeze_params == freeze and module.train_embedding_rows == rows
    assert module.rank_microbatch_size == 10240 and module.max_sequence_length == 2560
    assert all(block.ep.capacity_factor == 8 for block in saved.blocks)
    assert all(block.routed_experts_router.lb_loss_weight == 0.015 for block in saved.blocks)
    assert json.loads((saved.checkpoint / "config.json").read_text()) == metadata
    assert manifest["execution_overrides"]["destination_capacity_factor"] == 8


@pytest.mark.parametrize("field,value", [("max_sequence_length", 128), ("cp_config", object())])
def test_unsupported_checkpoint_execution_is_rejected(saved, field, value):
    setattr(saved.module, field, value)
    with pytest.raises(ValueError):
        runner.checkpoint_configs(saved.checkpoint, saved.evaluator, saved.panel, 16)


@pytest.mark.parametrize("world_size", [0, 3, 24, 32])
def test_incomplete_distributed_panel_or_ep_group_is_rejected(saved, world_size):
    with pytest.raises(ValueError):
        runner.checkpoint_configs(saved.checkpoint, saved.evaluator, saved.panel, world_size)


@pytest.mark.parametrize("change", ["tokenizer", "revision", "ancestry"])
def test_checkpoint_identity_mismatch_is_rejected(saved, change):
    value = deepcopy(saved.data)
    if change == "tokenizer":
        value["dataset"]["tokenizer"]["identifier"] = "different"
    elif change == "revision":
        value["dataset"]["tokenizer_revision"] = "different"
    else:
        value["pretraining_checkpoint"] = "different"
    (saved.checkpoint / "config.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.checkpoint_configs(saved.checkpoint, saved.evaluator, saved.panel, 16)


def make_row(source, index, reference, prediction, stop="eos"):
    row = {
        "source": source,
        "panel_index": index,
        "base_source_index": index + 100,
        "reference": reference,
        "prediction": prediction,
        "reference_complete": True,
        "stop_reason": stop,
        "reference_token_ids": [1, 2],
        "generated_token_ids": [1, 2],
        "output_tokens": 2,
    }
    row["metrics"] = score_prediction(runner.SCORE_SOURCES[source], row)
    row["output_diagnostics"] = runner.output_diagnostics(row, 768)
    return row


def test_point_summary_separates_absence_from_positive_localization():
    rows = [
        make_row("pixmo_points_basic", 0, "There are none.", "There are none."),
        make_row(
            "pixmo_points_basic",
            1,
            '<points coords="1 1 100 100">x</points>',
            '<points coords="1 1 900 900">x</points>',
        ),
    ]
    result = runner.source_summary(rows, "pixmo_points_basic", 2)
    assert result["point_f1_at_005"] == 0.5
    assert result["positive_examples"] == 1 and result["positive_point_f1_at_005"] == 0
    assert result["correct_empty_predictions"] == 1 and result["matched_points"] == 0
    assert result["parsed_outputs"] == 2


def test_repetition_and_reference_limits_are_reported_without_changing_primary_score():
    row = make_row("pixmo_caption", 0, "red car", "red car " * 30, stop="max_tokens")
    row["reference_token_ids"] = [1] * 769
    row["output_diagnostics"] = runner.output_diagnostics(row, 768)
    result = runner.source_summary([row], "pixmo_caption", 1)
    assert result["lexical_token_f1"] == 0
    assert result["completed_outputs"] == 0
    assert result["reference_exceeds_generation_limit"] == 1
    assert result["mean_word_fourgram_repeat_fraction"] > 0.9


def test_rescoring_serialized_outputs_preserves_all_source_aggregates():
    answers = {
        "scalar_count": [("7", "7"), ("7", "8")],
        "pixmo_points_basic": [
            (
                '<points coords="1 1 100 100">x</points>',
                '<points coords="1 1 100 100">x</points>',
            ),
            ("There are none.", '<points coords="1 1 900 900">x</points>'),
        ],
        "ocr_document": [("cat", "cat"), ("cat", "bat")],
        "pixmo_caption": [("red car", "red car"), ("red car", "red bus")],
    }
    expected_scores = {
        "scalar_count": {"strict_count_accuracy": 0.5},
        "pixmo_points_basic": {"point_f1_at_005": 0.5, "positive_point_f1_at_005": 1.0},
        "ocr_document": {
            "normalized_exact_match": 0.5,
            "normalized_character_error_rate": 1 / 6,
        },
        "pixmo_caption": {"lexical_token_f1": 0.75},
    }
    for source, scorer in runner.SCORE_SOURCES.items():
        saved = json.loads(
            json.dumps(
                [make_row(source, index, *answer) for index, answer in enumerate(answers[scorer])]
            )
        )
        original = deepcopy(saved)
        for row in saved:
            row["metrics"] = score_prediction(scorer, row)
        assert saved == original
        summary = runner.source_summary(saved, source, 2)
        assert summary == runner.source_summary(original, source, 2)
        for metric, value in expected_scores[scorer].items():
            assert summary[metric] == pytest.approx(value)


class Evaluator:
    def __init__(self, name, indices=range(100, 104)):
        self.name = f"{name}-validation"
        self.batches = SimpleNamespace(
            total_batches=1,
            reset=lambda: None,
            dp_world_size=1,
            dp_rank=0,
            dataset=SimpleNamespace(
                indices=range(4), dataset=SimpleNamespace(indices=list(indices))
            ),
        )

    def __iter__(self):
        yield {}


@pytest.fixture
def decoded_run(tmp_path, monkeypatch):
    sources = list(runner.SCORE_SOURCES)
    selected = tmp_path / "saved.indices"
    selected.write_text("100\n101\n102\n103\n")
    panel_manifest = {
        "examples_per_source": 4,
        "panels": {
            source: {
                "rows": [
                    {"panel_index": index, "base_source_index": index + 100} for index in range(4)
                ]
            }
            for source in sources
        },
    }
    eval_config = SimpleNamespace(
        examples_per_source=4,
        rank_batch_size=4,
        eval_dataset=SimpleNamespace(
            sources={
                source: MultimodalSourceConfig(dataset=Config(), selection_path=str(selected))
                for source in sources
            },
            build_tokenizer=lambda: (None, None),
        ),
        as_config_dict=lambda: {"sources": sources},
        build=lambda context: SimpleNamespace(evaluators=[Evaluator(s) for s in sources]),
    )
    module = SimpleNamespace(device="cpu", dp_process_group=None, load_state_dict_direct=Mock())
    model_config = SimpleNamespace(build=lambda **kwargs: None)
    module_config = SimpleNamespace(build=Mock(return_value=module))
    monkeypatch.setattr(runner, "load_frozen_evaluator", lambda path: (eval_config, panel_manifest))
    monkeypatch.setattr(
        runner,
        "checkpoint_configs",
        lambda *args: (model_config, module_config, {"init_seed": 1}),
    )
    monkeypatch.setattr(runner, "get_world_size", lambda: 1)
    monkeypatch.setattr(runner, "get_rank", lambda: 0)
    monkeypatch.setattr(runner, "is_distributed", lambda: False)
    calls = []
    fail = [False]

    def decode(module, batch, tokenizer, token_ids, identities, source, **kwargs):
        calls.append(source)
        if source == sources[1] and fail[0]:
            raise RuntimeError("injected late source failure")
        answer = "There are none." if source == "pixmo_points_basic" else "7"
        return [make_row(source, i, answer, answer) for i in range(4)]

    monkeypatch.setattr(runner, "decode_batch", decode)
    args = (
        tmp_path / "checkpoint",
        tmp_path / "panel",
        tmp_path / "panel.json",
        tmp_path / "outputs",
    )
    return SimpleNamespace(
        args=args,
        calls=calls,
        fail=fail,
        sources=sources,
        selected=selected,
        evaluator=eval_config,
        module_config=module_config,
    )


def test_late_source_failure_keeps_completed_sources_and_retry_reuses_them(decoded_run):
    args, sources, calls, fail = (
        decoded_run.args,
        decoded_run.sources,
        decoded_run.calls,
        decoded_run.fail,
    )
    fail[0] = True
    with pytest.raises(RuntimeError, match="injected"):
        runner.run(*args)
    assert (args[-1] / f"{sources[0]}-rank0.json").is_file()
    assert not (args[-1] / "results.json").exists()
    fail[0] = False
    runner.run(*args)
    result = json.loads((args[-1] / "results.json").read_text())
    assert result["completed"] and len(result["rows"]) == 4 * len(sources)
    assert calls.count(sources[0]) == 1
    before = list(calls)
    runner.run(*args)
    assert calls == before


@pytest.mark.parametrize("source", ["scalar_count", "pixmo_points_basic", "pixmo_caption"])
@pytest.mark.parametrize("cache", ["rank", "complete", "check_complete"])
@pytest.mark.parametrize("change", ["wrong_row", "reorder"])
def test_cached_panel_row_identity_is_validated(decoded_run, source, cache, change):
    args = decoded_run.args
    runner.run(*args)
    result_path = args[-1] / "results.json"
    if cache == "rank":
        result_path.unlink()
        result_path = args[-1] / f"{source}-rank0.json"
    result = json.loads(result_path.read_text())
    rows = [row for row in result["rows"] if row["source"] == source]
    rows[0]["base_source_index"] = 101
    if change == "reorder":
        rows[1]["base_source_index"] = 100
    result_path.write_text(json.dumps(result))
    calls = list(decoded_run.calls)
    with pytest.raises((ValueError, RuntimeError), match="row identities differ"):
        runner.run(*args, check_complete=cache == "check_complete", world_size=1)
    assert decoded_run.calls == calls


@pytest.mark.parametrize("cache", ["rank", "complete", "check_complete"])
def test_changed_selection_is_rejected_before_any_cache_reuse(decoded_run, cache):
    args = decoded_run.args
    runner.run(*args)
    if cache == "rank":
        (args[-1] / "results.json").unlink()
    decoded_run.selected.write_text("101\n100\n102\n103\n")
    decoded_run.module_config.build.reset_mock()
    calls = list(decoded_run.calls)
    with pytest.raises((ValueError, RuntimeError), match="row identities differ"):
        runner.run(*args, check_complete=cache == "check_complete", world_size=1)
    decoded_run.module_config.build.assert_not_called()
    assert decoded_run.calls == calls


def test_built_source_identity_is_validated_before_decoding_or_rank_cache(decoded_run):
    # Simulate a selection change after the lightweight check but before source construction.
    args, sources = decoded_run.args, decoded_run.sources
    runner.run(*args)
    (args[-1] / "results.json").unlink()
    decoded_run.evaluator.build = lambda context: SimpleNamespace(
        evaluators=[Evaluator(source, [101, 100, 102, 103]) for source in sources]
    )
    calls = list(decoded_run.calls)
    with pytest.raises(RuntimeError, match="Panel preparation.*row identities differ"):
        runner.run(*args)
    assert decoded_run.calls == calls


def test_shared_decoder_legacy_defaults_are_not_changed():
    from olmo_core.eval.multimodal_decoding import TOKEN_LIMITS

    assert runner.TOKEN_LIMITS["pixmo_points_basic"] == 256
    assert TOKEN_LIMITS["pixmo_caption"] == 512
    assert TOKEN_LIMITS["ocr_document"] == 64
    assert runner.TOKEN_LIMITS["pixmo_caption"] == 768
    assert runner.TOKEN_LIMITS["ocr_document"] == 256


@pytest.mark.parametrize("world_size,expected", [(None, 8), (8, 8), (16, 16)])
def test_explicit_world_size_is_recorded_by_metadata_checks(
    saved, tmp_path, monkeypatch, capsys, world_size, expected
):
    selected = {"examples_per_source": 64, "rank_batch_size": 4, "fixed_panel": True}
    panel_manifest = {
        "panels": {
            source: {
                "rows": [
                    {"panel_index": index, "base_source_index": index + 100} for index in range(64)
                ]
            }
            for source in runner.SCORE_SOURCES
        }
    }
    saved.evaluator.as_config_dict = lambda: selected
    monkeypatch.setattr(
        runner,
        "load_frozen_evaluator",
        lambda path: (saved.evaluator, panel_manifest),
    )
    index_check = Mock()
    monkeypatch.setattr(runner, "validate_frozen_panel_indices", index_check)
    output = tmp_path / "outputs"
    args = (saved.checkpoint, saved.panel, tmp_path / "panel.json", output)
    runner.run(*args, dry_run=True, world_size=world_size)
    index_check.assert_not_called()
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["receipt_version"] == 1
    assert manifest["execution"]["world_size"] == expected
    assert "world_size" not in manifest["benchmark_definition"]
    assert manifest["benchmark_definition"]["selected_evaluator"] == selected
    assert saved.module.rank_microbatch_size == 4 * saved.evaluator.sequence_length
    output.mkdir()
    (output / "results.json").write_text(
        json.dumps(
            {
                "manifest": manifest,
                "completed": True,
                "rows": [
                    make_row(source, index, "7", "7")
                    for source in runner.SCORE_SOURCES
                    for index in range(64)
                ],
            }
        )
    )
    summary = Mock()
    monkeypatch.setattr(runner, "source_summary", summary)
    runner.run(*args, check_complete=True, world_size=world_size)
    index_check.assert_called_once_with(saved.evaluator, panel_manifest)
    assert summary.call_count == len(runner.SCORE_SOURCES)
    if expected == 16:
        with pytest.raises(ValueError, match="does not match"):
            runner.run(*args, check_complete=True)


@pytest.mark.parametrize("requested,active", [(8, 16), (16, 8), (0, 8), (-8, 8)])
def test_requested_runtime_geometry_must_match_process_group(
    tmp_path, monkeypatch, requested, active
):
    monkeypatch.setattr(runner, "get_world_size", lambda: active)
    monkeypatch.setattr(runner, "load_frozen_evaluator", lambda path: pytest.fail("loaded panel"))
    with pytest.raises(ValueError, match="world size"):
        runner.run(tmp_path, tmp_path, tmp_path, tmp_path, world_size=requested)


def test_eight_and_sixteen_ranks_cover_identical_frozen_rows_and_ep_batches(tmp_path):
    from olmo_core.data.multimodal.data_loader import MultimodalDataLoader

    class Panel:
        # Preserve a nontrivial frozen ordering and mapping to original source rows.
        indices = tuple(reversed(range(64)))
        dataset = SimpleNamespace(indices=[1000 + 7 * index for index in range(64)])

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            return (
                index,
                self.indices[index],
                self.dataset.indices[self.indices[index]],
            )

    class Collator:
        pad_sequence_length = 2560

        def __call__(self, examples):
            return examples

    by_world = {}
    for world_size in (8, 16):
        rows, ep_batches = [], {}
        for rank in range(world_size):
            loader = MultimodalDataLoader(
                Panel(),
                Collator(),
                work_dir=tmp_path,
                global_batch_size=4 * world_size * 2560,
                shuffle=False,
                dp_world_size=world_size,
                dp_rank=rank,
                fs_local_rank=0,
            )
            loader.reshuffle(1)
            assert loader.total_batches == (2 if world_size == 8 else 1)
            for batch_index, batch in enumerate(loader._iter_batches()):
                identities = runner.source_indices(SimpleNamespace(batches=loader), batch_index, 4)
                resolved = [
                    (
                        row["panel_index"],
                        row["selected_source_index"],
                        row["base_source_index"],
                    )
                    for row in identities
                ]
                assert resolved == batch and len(batch) == 4
                rows.extend(resolved)
                group = batch_index * (world_size // 8) + rank // 8
                ep_batches.setdefault(group, []).extend(resolved)
        assert sorted(row[0] for row in rows) == list(range(64))
        by_world[world_size] = (sorted(rows), ep_batches)
    assert by_world[8] == by_world[16]


def test_main_accepts_explicit_arguments_without_starting_training_for_checks(
    tmp_path, monkeypatch
):
    for name in ("checkpoint", "panel-checkpoint", "panel-file"):
        (tmp_path / name).touch()
    run = Mock()
    monkeypatch.setattr(runner, "run", run)
    prepare = Mock()
    monkeypatch.setattr(runner, "prepare_training_environment", prepare)
    monkeypatch.setattr(runner, "check_scorers", Mock())
    runner.main(
        [
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--panel-checkpoint",
            str(tmp_path / "panel-checkpoint"),
            "--panel-file",
            str(tmp_path / "panel-file"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--world-size",
            "8",
            "--check-complete",
        ]
    )
    prepare.assert_not_called()
    run.assert_called_once_with(
        tmp_path / "checkpoint",
        tmp_path / "panel-checkpoint",
        tmp_path / "panel-file",
        tmp_path / "outputs",
        dry_run=False,
        check_complete=True,
        world_size=8,
    )
