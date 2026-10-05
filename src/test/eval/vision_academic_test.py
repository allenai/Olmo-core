"""Scientific contracts and task-level restart behavior for academic vision evaluation."""

import contextlib
import copy
import io
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from olmo_core.eval import vision_academic
from olmo_core.train.train_module.transformer.multimodal_train_module import (
    MultimodalOLMoDDPTrainModule,
)


@pytest.fixture(scope="module")
def academic():
    return vision_academic


def test_standard_task_metrics(academic):
    assert academic._normalize_vqa_answer("The red,blue cats?") == "red blue cats"
    assert academic._normalize_vqa_answer("One cat, don't go!") == "1 cat don't go"
    assert academic._normalize_vqa_answer("dont") == "don't"
    assert academic._normalize_vqa_answer("1,000.") == "1000"
    assert academic._normalize_vqa_answer("red,blue?") == "red blue"
    assert academic._normalize_textvqa_answer("red,blue?") == "redblue"
    assert academic._normalize_textvqa_answer("Girl's") == "girl 's"
    # This inert lowercase contraction matches the published EvalAI implementation's
    # capital-I dictionary keys after its lowercasing pass.
    assert academic._normalize_textvqa_answer("im") == "im"

    three_matches = ["yes"] * 3 + ["no"] * 7
    assert academic._vqa_accuracy("yes", three_matches) == pytest.approx(0.9)
    assert academic._vqa_accuracy("yes", ["yes"] * 4 + ["no"] * 6) == 1.0
    assert academic._anls("sitten", ["kitten"]) == pytest.approx(5 / 6)
    assert academic._anls("x", ["kitten"]) == 0.0
    assert academic._anls("ab", ["ac"]) == 0.0
    assert academic._chartqa_relaxed_accuracy("104", "100") == 1.0
    assert academic._chartqa_relaxed_accuracy("105", "100") == 1.0
    assert academic._chartqa_relaxed_accuracy("106", "100") == 0.0
    assert academic._chartqa_relaxed_accuracy("5%", "0.05") == 1.0
    assert academic._chartqa_relaxed_accuracy("0", "0") == 1.0
    assert academic._chartqa_relaxed_accuracy("0.0", "0") == 0.0
    assert academic._chartqa_relaxed_accuracy("Blue", "blue") == 1.0


def test_multiple_choice_scoring_and_empty_option_rendering(academic):
    ai2d = academic.AcademicExample(
        task="ai2d",
        example_id="4185.png-0",
        source_position="392",
        visual=None,
        image_reference=None,
        question="Which diagram label is correct?",
        options=("D", "B", "A", "C"),
        answer_index=2,
        stratum="standard",
    )
    # The predicted index is the outer prompt label C, not the raw diagram-letter text A.
    assert academic._score_prediction(ai2d, prediction="C", predicted_index=2) == 1.0
    assert academic._score_prediction(ai2d, prediction="A", predicted_index=0) == 0.0

    a_okvqa = academic.AcademicExample(
        task="a_okvqa_mc",
        example_id="duplicate-correct-text",
        source_position="0",
        visual=None,
        image_reference=None,
        question="Where?",
        options=("road", "water", "road", "air"),
        answer_index=0,
    )
    # The official A-OKVQA MC scorer compares chosen text, so a duplicate text is correct.
    assert academic._score_prediction(a_okvqa, prediction="C", predicted_index=2) == 1.0

    prompt = academic._build_mc_prompt("Choose.", ("s", "", "b", "f"))
    assert "A. s\nB. <empty>\nC. b\nD. f" in prompt


@pytest.mark.parametrize("size", [(378, 378), (256, 1024), (1024, 256), (80, 2400)])
def test_grid_signature_matches_native_preprocessor(academic, size):
    image = Image.new("RGB", size, color=(17, 31, 47))
    _, _, actual = academic.preprocess_image_molmo2(
        image,
        dtype=torch.float32,
        device=torch.device("cpu"),
        max_crops=academic.DEFAULT_MAX_CROPS,
        is_training=False,
    )
    assert academic._molmo2_grid_signature(image) == tuple(int(value) for value in actual)


def test_ai2d_adapter_keeps_encoded_images_lazy_and_blank_positions(academic):
    encoded = io.BytesIO()
    Image.new("RGB", (8, 4), color=(1, 2, 3)).save(encoded, format="PNG")
    image_reference = {"bytes": encoded.getvalue(), "path": None}
    raw = [
        {
            "question_id": "4185.png-0",
            "image": image_reference,
            "question": "Choose.",
            "answer_texts": ["red", "blue", "", "green"],
            "correct_answer": 3,
            "has_transparent_box": True,
        }
    ]
    examples = academic._ai2d_examples_from_raw(raw)
    assert examples[0].visual is image_reference
    assert examples[0].image_reference is image_reference
    assert not isinstance(examples[0].visual, Image.Image)
    assert examples[0].options == ("red", "blue", "", "green")
    assert examples[0].answer_index == 3
    assert examples[0].stratum == "transparent"


def test_a_okvqa_adapter_preserves_known_blank_distractor(academic, monkeypatch):
    row = {
        "image": "unused.png",
        "question": "Which letter?",
        "options": ["s", "", "b", "f"],
        "answer_idx": 3,
        "metadata": {"example_id": "iTeGLgJNuQjkRqeTmKwWLs"},
    }
    monkeypatch.setitem(
        academic.academic_registry.ACADEMIC_REGISTRY,
        "a_okvqa_mc",
        SimpleNamespace(loader=lambda split: [row]),
    )
    examples = academic._load_task_examples("a_okvqa_mc")
    assert examples[0].example_id == "iTeGLgJNuQjkRqeTmKwWLs"
    assert examples[0].source_position == "0"
    assert examples[0].options == ("s", "", "b", "f")
    assert examples[0].answer_index == 3


def test_ai2d_strata_and_exact_nonoverlap_pair_deltas(academic):
    rows = []
    specifications = (
        ("standard", False, False, 1.0, 0.0, 0.0),
        ("transparent", False, True, 0.0, 1.0, 1.0),
        ("standard", True, False, 0.0, 0.0, 0.0),
    )
    for index, (
        stratum,
        recipient_overlap,
        donor_overlap,
        correct,
        shuffled,
        blank,
    ) in enumerate(specifications):
        rows.append(
            {
                "example_id": str(index),
                "stratum": stratum,
                "alignment_train_image_overlap": recipient_overlap,
                "shuffled_alignment_train_image_overlap": donor_overlap,
                "controls": {
                    "correct": {"score": correct},
                    "shuffled": {"score": shuffled},
                    "blank": {"score": blank},
                },
            }
        )
    result = academic._aggregate_task_outputs("ai2d", rows)
    metrics = result["controls"]["correct"]
    assert metrics["multiple_choice_accuracy_standard"] == pytest.approx(0.5)
    assert metrics["multiple_choice_accuracy_transparent"] == 0.0
    assert result["controls"]["shuffled"]["exact_byte_nonoverlap_multiple_choice_accuracy"] == 0.0
    deltas = result["image_control_deltas"]
    assert deltas[
        "exact_byte_nonoverlap_multiple_choice_accuracy_correct_minus_shuffled"
    ] == pytest.approx(1.0)
    assert deltas["exact_byte_nonoverlap_correct_minus_shuffled_examples"] == 1
    assert deltas[
        "exact_byte_nonoverlap_multiple_choice_accuracy_correct_minus_blank"
    ] == pytest.approx(0.0)
    assert deltas["exact_byte_nonoverlap_correct_minus_blank_examples"] == 2


def test_generation_stop_counts_distinguish_eos_and_cap(academic):
    rows = []
    for stop_reason in ("eos", "max_tokens"):
        rows.append(
            {"controls": {control: {"stop_reason": stop_reason} for control in academic.CONTROLS}}
        )
    counts = academic._generation_stop_counts("docvqa", rows)
    assert counts == {control: {"eos": 1, "max_tokens": 1} for control in academic.CONTROLS}


@pytest.mark.parametrize("multiple_choice", [True, False], ids=["mc", "free"])
@pytest.mark.parametrize(
    "outer_context",
    [contextlib.nullcontext, torch.no_grad, torch.inference_mode],
    ids=["grad", "no_grad", "inference"],
)
def test_prediction_uses_eager_grad_context_and_detaches_graphs(
    academic, monkeypatch, multiple_choice, outer_context
):
    context_events = []
    context_active = False
    feature_graphs = []
    logits_graphs = []
    forwarded_features = []
    forwarded_inputs = []

    @contextlib.contextmanager
    def set_stance(stance):
        nonlocal context_active
        assert stance == "force_eager"
        context_events.append("enter")
        context_active = True
        try:
            yield
        finally:
            context_active = False
            context_events.append("exit")

    monkeypatch.setattr(torch.compiler, "set_stance", set_stance)

    def assert_safe_context():
        assert context_active
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()

    class FakeModel:
        def encode_images(self, images, pooling):
            assert_safe_context()
            assert not images.is_inference()
            assert not pooling.is_inference()
            features = torch.ones(1, requires_grad=True) * 2
            feature_graphs.append(weakref.ref(features))
            return features

    class FakeTrainModule:
        device = torch.device("cpu")
        _eval_batch_context = MultimodalOLMoDDPTrainModule._eval_batch_context

        def __init__(self):
            self.model_parts = [FakeModel()]

        def model_forward_no_pipeline(self, input_ids, **kwargs):
            assert_safe_context()
            assert all(graph() is None for graph in feature_graphs + logits_graphs)
            assert not input_ids.is_inference()
            for key in ("token_type_ids", "position_ids", "logits_to_keep"):
                assert not kwargs[key].is_inference()
            features = kwargs["encoded_image_features"]
            assert not features.is_inference()
            assert not features.requires_grad
            assert features.grad_fn is None
            forwarded_features.append(features)
            forwarded_inputs.append(input_ids.clone())
            logits = torch.zeros((1, 1, 100))
            if multiple_choice or len(forwarded_inputs) == 1:
                logits[0, 0, 10] = 2.0
                logits[0, 0, 11] = 1.0
            else:
                logits[0, 0, 2] = 2.0
            logits = logits * torch.ones((), requires_grad=True)
            logits_graphs.append(weakref.ref(logits))
            return logits

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def decode(self, tokens, *, skip_special_tokens):
            assert skip_special_tokens
            assert tokens == [10, 2]
            return "first"

    inference = academic._NativeAcademicInference(
        FakeTrainModule(),
        FakeTokenizer(),
        SimpleNamespace(image_token_ids=frozenset({90})),
        max_sequence_length=8192,
        max_crops=academic.DEFAULT_MAX_CROPS,
        max_new_tokens=2,
        sequence_bucket_size=128,
    )
    inference._prepare_visual = lambda image: (
        torch.zeros(1),
        torch.zeros(1),
        [90],
        (14, 14, 14, 14),
    )

    def consensus_token(token):
        # The forward's output tensor is released before prediction processing, not
        # held together with its autograd graph until the following forward returns.
        assert all(graph() is None for graph in logits_graphs)
        return token

    inference._consensus_token = consensus_token
    monkeypatch.setattr(
        academic, "document_prompt_ids", lambda tokenizer, prompt, image_ids: [1, 90, 2]
    )
    monkeypatch.setattr(
        academic,
        "response_ids",
        lambda tokenizer, letter: [10 + "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(letter)],
    )
    example = academic.AcademicExample(
        task="ai2d" if multiple_choice else "docvqa",
        example_id="ai2d-0",
        source_position="0",
        visual=None,
        image_reference=None,
        question="Choose.",
        options=("first", "second") if multiple_choice else (),
        answer_index=0 if multiple_choice else None,
    )
    with outer_context():
        previous_modes = (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        output = inference.predict(example, Image.new("RGB", (2, 2)))
        assert previous_modes == (torch.is_grad_enabled(), torch.is_inference_mode_enabled())
    if multiple_choice:
        assert output["predicted_index"] == 0
        assert output["prediction"] == "A"
        assert len(forwarded_inputs) == 1
    else:
        assert output["prediction"] == "first"
        assert output["generated_token_ids"] == [10, 2]
        assert output["stop_reason"] == "eos"
        assert len(forwarded_inputs) == 2
        assert forwarded_inputs[1][0, :4].tolist() == [1, 90, 2, 10]
        assert forwarded_features[0] is forwarded_features[1]
    assert len(feature_graphs) == 1
    assert all(graph() is None for graph in feature_graphs + logits_graphs)
    assert context_events == ["enter", "exit", "enter", "exit"]
    assert output["image_grid_signature"] == [14, 14, 14, 14]


@pytest.fixture
def selected_panel(academic, tmp_path):
    examples = []
    records = []
    for index, color in enumerate(("red", "blue")):
        image = tmp_path / f"{index}.png"
        Image.new("RGB", (100, 100), color).save(image)
        example = academic.AcademicExample(
            task="docvqa",
            example_id=str(index),
            source_position=str(index),
            visual=str(image),
            image_reference=str(image),
            question="What is visible?",
            answers=(color,),
        )
        examples.append(example)
        grid = list(academic._molmo2_grid_signature(str(image)))
        records.append(
            {
                "example_id": example.example_id,
                "source_position": example.source_position,
                "annotation_sha256": academic._canonical_sha256(example.annotation()),
                "image_sha256": academic._file_sha256(image),
                "image_grid_signature": grid,
                "image_token_count": len(academic.build_image_token_ids(*grid)),
                "alignment_train_image_overlap": bool(index),
            }
        )
    for index, record in enumerate(records):
        donor = records[1 - index]
        record.update(
            {
                "shuffled_donor_id": donor["example_id"],
                "shuffled_image_sha256": donor["image_sha256"],
                "shuffled_image_grid_signature": donor["image_grid_signature"],
                "shuffled_alignment_train_image_overlap": donor["alignment_train_image_overlap"],
            }
        )
    manifest = {
        "format": "vision_alignment_external_academic_manifest",
        "selection": {"tasks": ["docvqa"], "split": "validation"},
        "controls": {"names": list(academic.CONTROLS)},
        "tasks": {
            "docvqa": {
                "records": records,
                "selection_count": 2,
                "selection_sha256": academic._canonical_sha256(records),
                "source": {},
            }
        },
    }
    return manifest, examples


def _save_manifest(path, value):
    content = {key: item for key, item in value.items() if key != "content_sha256"}
    path.write_text(
        json.dumps({**content, "content_sha256": vision_academic._canonical_sha256(content)})
    )


def test_selected_loader_preserves_rows_donors_and_checks_only_selected_images(
    academic, selected_panel, tmp_path, monkeypatch
):
    manifest, examples = selected_panel
    path = tmp_path / "manifest.json"
    _save_manifest(path, manifest)
    loaded_manifest = academic.load_manifest(path)
    unused = academic.AcademicExample(
        task="docvqa",
        example_id="unused",
        source_position="unused",
        visual=None,
        image_reference="/missing/unselected.png",
        question="unused",
        answers=("unused",),
    )
    monkeypatch.setattr(academic, "_load_task_examples", lambda task: [*examples, unused])
    loaded = academic.load_selected_examples(loaded_manifest, ["docvqa"])
    assert list(loaded["docvqa"]) == ["0", "1"]
    assert loaded["docvqa"]["0"] is examples[0]
    assert [row["shuffled_donor_id"] for row in loaded_manifest["tasks"]["docvqa"]["records"]] == [
        "1",
        "0",
    ]
    assert academic.benchmark_definition(loaded_manifest) == academic.benchmark_definition(manifest)
    Image.new("RGB", (100, 100), "green").save(examples[0].image_reference)
    with pytest.raises(ValueError, match="Selected image differs"):
        academic.load_selected_examples(loaded_manifest, ["docvqa"])


@pytest.mark.parametrize("change", ["annotation_sha256", "source_position"])
def test_selected_loader_rejects_changed_annotations(academic, selected_panel, monkeypatch, change):
    manifest, examples = selected_panel
    monkeypatch.setattr(academic, "_load_task_examples", lambda task: examples)
    manifest["tasks"]["docvqa"]["records"][0][change] = "changed"
    with pytest.raises(ValueError, match="Selected annotation differs"):
        academic.load_selected_examples(manifest, ["docvqa"])


@pytest.mark.parametrize("change", ["donor", "overlap", "checksum"])
def test_manifest_rejects_inconsistent_donors(academic, selected_panel, tmp_path, change):
    manifest, _ = selected_panel
    record = manifest["tasks"]["docvqa"]["records"][0]
    if change == "donor":
        record["shuffled_donor_id"] = "0"
    elif change == "overlap":
        record["shuffled_alignment_train_image_overlap"] = False
    if change != "checksum":
        manifest["tasks"]["docvqa"]["selection_sha256"] = academic._canonical_sha256(
            manifest["tasks"]["docvqa"]["records"]
        )
    else:
        record["source_position"] = "changed"
    path = tmp_path / "manifest.json"
    _save_manifest(path, manifest)
    with pytest.raises(ValueError):
        academic.load_manifest(path)


def _panel_result(academic, manifest, examples):
    rows = []
    for example, record in zip(examples, manifest["tasks"]["docvqa"]["records"]):
        prediction = {
            "prediction": example.answers[0],
            "predicted_index": None,
            "score": 1.0,
            "generated_token_ids": [7, 100257],
            "output_tokens": 2,
            "stop_reason": "eos",
            "image_grid_signature": record["image_grid_signature"],
            "image_token_count": record["image_token_count"],
            "image_token_ids_sha256": academic._canonical_sha256(
                academic.build_image_token_ids(
                    *record["image_grid_signature"], token_ids=academic.TOKEN_IDS
                )
            ),
        }
        rows.append(
            {
                **record,
                "question": example.question,
                "gold_answers": list(example.answers),
                "gold_answer_index": None,
                "options": [],
                "stratum": None,
                "controls": {control: dict(prediction) for control in academic.CONTROLS},
            }
        )
    return {
        "selection_count": len(rows),
        "selection_sha256": manifest["tasks"]["docvqa"]["selection_sha256"],
        "examples": rows,
        **academic._aggregate_task_outputs("docvqa", rows),
        "generation_stop_counts": academic._generation_stop_counts("docvqa", rows),
        "alignment_train_image_overlap_count": 1,
    }


def test_atomic_task_resume_and_cpu_merge(academic, selected_panel, tmp_path, monkeypatch):
    manifest, examples = selected_panel
    result = _panel_result(academic, manifest, examples)
    checkpoint = tmp_path / "step500"
    (checkpoint / "model_and_optim").mkdir(parents=True)
    (checkpoint / "model_and_optim" / ".metadata").write_bytes(b"state metadata")
    (checkpoint / ".metadata.json").write_text("{}")
    saved = {
        "dataset": {
            "tokenizer": {"identifier": academic.TOKENIZER_ID, "pad_token_id": 100277},
            "tokenizer_revision": academic.TOKENIZER_REVISION,
        },
        "model": {"vision": {}, "lm": {"vocab_size": 100352}, "image_patch_token_id": 100280},
    }
    (checkpoint / "config.json").write_text(json.dumps(saved))
    definition = academic._definition(checkpoint, manifest)
    assert definition["benchmark"] == academic.benchmark_definition(manifest)
    assert definition["execution"]["forward_context"] == "eager_grad_enabled"
    output = tmp_path / "academic.json"
    task_path = academic._task_path(output, "docvqa")
    with pytest.raises(FileNotFoundError):
        academic.merge_results(output, manifest, definition)
    academic._write_result(
        task_path,
        {
            "format": academic.RESULT_FORMAT,
            "definition": definition,
            "task": "docvqa",
            "completed": True,
            "result": result,
        },
    )
    monkeypatch.setattr(
        academic,
        "load_selected_examples",
        lambda *args: pytest.fail("Merge loaded data"),
    )
    monkeypatch.setattr(
        academic, "_load_tokenizer", lambda *args: pytest.fail("Merge loaded tokenizer")
    )
    merged = academic.merge_results(output, manifest, definition)
    assert merged["completed"]
    assert merged["tasks"] == {"docvqa": result}
    assert list(tmp_path.glob("tmp*.json")) == []
    other = copy.deepcopy(definition)
    other["execution"]["checkpoint"] = "step1000"
    with pytest.raises(ValueError, match="mismatched"):
        academic.merge_results(output, manifest, other)

    # Results without an explicit forward context cannot be resumed or mixed with
    # predictions from eager, grad-enabled execution.
    previous = academic._read_json(task_path)
    previous["definition"]["execution"].pop("forward_context")
    task_path.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="mismatched"):
        academic._load_task_result(task_path, "docvqa", manifest, definition)
    with pytest.raises(ValueError, match="mismatched"):
        academic.merge_results(output, manifest, definition)
    with pytest.raises(ValueError, match="another evaluation"):
        academic._write_result(task_path, {**previous, "definition": definition})


@pytest.mark.parametrize("change", ["score", "aggregate", "token", "order", "missing"])
def test_cached_results_reject_inconsistent_rows(academic, selected_panel, change):
    manifest, examples = selected_panel
    result = _panel_result(academic, manifest, examples)
    if change == "score":
        result["examples"][0]["controls"]["correct"]["score"] = 0.0
    elif change == "aggregate":
        result["controls"]["correct"]["anls"] = 0.0
    elif change == "token":
        result["examples"][0]["controls"]["correct"]["generated_token_ids"] = [
            100280,
            100257,
        ]
    elif change == "order":
        result["examples"].reverse()
    else:
        result["examples"].pop()
    with pytest.raises(ValueError):
        academic.validate_task_result("docvqa", result, manifest)


def test_does_not_overwrite_legacy_receipts(academic, tmp_path):
    path = tmp_path / "legacy.json"
    legacy = {
        "format": "vision_alignment_historical_panel_comparison",
        "definition": {},
    }
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="another evaluation"):
        academic._write_result(path, {"format": academic.RESULT_FORMAT, "definition": {}})
    assert json.loads(path.read_text()) == legacy


@pytest.mark.parametrize("task", vision_academic.DEFAULT_TASKS)
def test_saved_reference_rescoring(academic, task):
    reference = os.environ.get("OLMO_VISION_ACADEMIC_REFERENCE")
    manifest_path = os.environ.get("OLMO_VISION_ACADEMIC_MANIFEST")
    if reference is None or manifest_path is None:
        pytest.skip("Set OLMO_VISION_ACADEMIC_REFERENCE and OLMO_VISION_ACADEMIC_MANIFEST")
    manifest = academic.load_manifest(Path(manifest_path))
    result = json.loads(Path(reference).read_text())["tasks"][task]
    academic.validate_task_result(task, result, manifest)
    assert len(result["examples"]) == 512


def test_saved_reference_prompt_tokens(academic):
    reference = os.environ.get("OLMO_VISION_ACADEMIC_REFERENCE")
    cache = os.environ.get("OLMO_VISION_ACADEMIC_HF_CACHE")
    if reference is None or cache is None:
        pytest.skip("Set OLMO_VISION_ACADEMIC_REFERENCE and OLMO_VISION_ACADEMIC_HF_CACHE")
    tokenizer, token_ids = academic._load_tokenizer(cache, 100352)
    assert token_ids == academic.TOKEN_IDS
    tasks = json.loads(Path(reference).read_text())["tasks"]
    for result in tasks.values():
        for row in result["examples"]:
            prompt = (
                academic._build_mc_prompt(row["question"], row["options"])
                if row["options"]
                else academic._free_answer_prompt(row["question"])
            )
            image_ids = academic.build_image_token_ids(
                *row["image_grid_signature"], token_ids=token_ids
            )
            ids = academic.document_prompt_ids(tokenizer, prompt, image_ids=image_ids)
            assert all(
                prediction["input_tokens"] == len(ids) for prediction in row["controls"].values()
            )
            for prediction in row["controls"].values():
                if not row["options"]:
                    assert (
                        tokenizer.decode(
                            prediction["generated_token_ids"], skip_special_tokens=True
                        ).strip()
                        == prediction["prediction"]
                    )
