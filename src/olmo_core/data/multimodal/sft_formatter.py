"""SFT prompt formatting for Molmo2.

Defaults to the stage-2 family (``uber_model_v2`` templates + ``demo_or_style_v2`` system
prompt). Stage 1 uses a different family -- ``prompt_templates="none"`` with
``system_prompt="style_and_length_v2"`` -- which puts the bare question (for pointing, the bare
lowercased label) behind a ``"<style>:"`` tag rather than a natural-language template. Both are
selectable, because the form has to follow the checkpoint: a model trained on one and
evaluated on the other is out of distribution.
"""

from __future__ import annotations

import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .grounding import (
    MULTI_IMAGE_POINT_COUNT_PROMPTS,
    MULTI_IMAGE_POINTING_PROMPTS,
    POINT_COUNT_PROMPTS,
    POINTING_PROMPTS,
    multi_image_pointing_answer,
    normalize_points,
    pointing_answer,
)
from .multiple_choice_templates import template_mc_question

__all__ = [
    "SftFormatter",
    "DEMO_STYLES",
    "IMAGE_MC_STYLES",
    "MULTI_IMAGE_POINTING_STYLES",
    "AUX_POINTING_STYLES",
    "base_pointing_style",
]

# Subset of mm_olmo data_formatter.py DEMO_STYLES / IMAGE_MC_STYLES used by image-only-v9.
DEMO_STYLES = frozenset(
    {
        "point_count",
        "pointing",
        "cosyn_point",
        # This repo's tag for GUISyn's interaction intents (mm_olmo asks them as "pointing").
        "gui_point",
        "user_qa",
        "long_caption",
        "short_caption",
        "text_sft",
        "correction_qa",
        # NOTE: "synthetic_qa" (PixMoCapQa) and "transcript" are NOT demo styles in
        # mm_olmo (data_formatter.py:930-962) — they get a "style:" prefix.
        # mm_olmo DEMO_STYLES also carries the multi-image pointing family
        # (data_formatter.py:958-961) — no "style:" prefix under demo_or_style_v2.
        "multi_image_pointing",
        "multi_image_counting",
        "multi_image_point_then_count",
        "multi_image_count_then_point",
    }
)
IMAGE_MC_STYLES = frozenset(
    {
        "a_okvqa_mc",
        "ai2_diagram",
        "ai2_diagram_no_letter",
        "science_qa",
        "eval_multiple_choice",
        "multi_image_mc",
        "mantis_instruct_mc",
    }
)
MULTI_IMAGE_POINTING_STYLES = frozenset(
    {
        "multi_image_pointing",
        "multi_image_point_then_count",
        "multi_image_counting",
        "multi_image_count_then_point",
    }
)
# mm_olmo's marker styles for point sets that FAILED the VLM audit (``PixMoPointV2.audit_style``
# / ``PixMoCountConfigV2.audit_style``, data_formatter.py:1189, 1824-1825). They render exactly
# like their base style -- same prompt pool, same answer -- except for the style token itself,
# so the model learns these targets are less reliable without them diluting the primary
# ``pointing`` / ``point_count`` distributions. Not demo styles: they keep their ``"<style>:"``
# prefix under every system-prompt family.
AUX_POINTING_STYLES = frozenset({"aux_pointing", "aux_point_count"})
POINTING_STYLES = frozenset(
    {"pointing", "point_count", "point_then_count", "cosyn_point", "gui_point"}
    | AUX_POINTING_STYLES
)


def base_pointing_style(style: str) -> str:
    """Strip mm_olmo's ``aux_`` marker: ``aux_point_count`` is formatted as ``point_count``
    (``format_points``, data_formatter.py:1189)."""
    return style[4:] if style.startswith("aux_") else style


CAPTION_PROMPTS = (
    "Describe this image.",
    "Describe this image",
    "describe the image",
    "Write a long description of this image.",
    "caption the picture",
    "Caption",
    "caption",
    "Construct a long caption for this image",
    "Generate a caption",
    "Create a detailed caption",
    "Write a long caption",
    "Describe this image in detail",
    "Describe this",
    "describe this",
    "Caption this",
    "What can be seen in this image?",
    "What do you see in the image?",
    "Look at this photo carefully and then tell me about it in detail",
    "Write a long description of this image",
    "Tell me about this picture.",
    "Write a paragraph about this image.",
    "Look at this image carefully and then describe it in detail",
    "Generate a long caption about this image.",
)

# mm_olmo GENERAL_PROMPTS_V1["short_caption"] (data_formatter.py:71-83), verbatim.
SHORT_CAPTION_PROMPTS = (
    "Caption the image with 1 or two sentences",
    "Write a very short description of this image.",
    "Briefly describe the image.",
    "Look and this image, and then summarize it in a sentence or two.",
    "Write a brief caption describing the image",
    "Brief Caption:",
    "A short image caption:",
    "A short image description",
    "Briefly describe the content of the image.",
    "Can you give me one sentence summary of the picture?",
    "How would you describe this image in a sentence or two?",
)

CHAIN_OF_THOUGHT_PROMPTS = ("{question} Provide reasoning steps and then give the short answer.",)


def _apply_chain_of_thought_prompt(question: str) -> str:
    return CHAIN_OF_THOUGHT_PROMPTS[0].format(question=question)


@dataclass
class SftFormatter:
    """Minimal port of mm_olmo DataFormatter.

    The prompt family must match the checkpoint. The defaults are the stage-2 family used by
    ``image_only_v9``; stage 1 passes ``prompt_templates="none"`` and
    ``system_prompt="style_and_length_v2"`` to match the released ``Molmo2-4B-Pretrain``
    config, whose ``data_formatter`` records exactly those.

    Getting it wrong is quiet and costly: a stage-1 4B checkpoint trained on the templated
    form but evaluated on the terse form scored 0.706 f1 on ``pixmo_point_eval_v3_mp`` against
    0.815 for the released Pretrain checkpoint, losing most of it on abstention (zero-slice f1
    0.718 vs 0.913) because the "Please say 'There are none.'" instruction lives only in the
    stage-2 template.
    """

    select_answer: str = "best"
    seed: int = 0
    prompt_templates: str = "uber_model_v2"
    """``"uber_model_v2"`` samples a natural-language template; ``"none"`` uses the bare
    lowercased label (mm_olmo ``data_formatter.py:1759-1779``)."""
    system_prompt: str = "demo_or_style_v2"
    """``"demo_or_style_v2"`` prefixes the non-demo styles with ``"<style>:"``;
    ``"style_and_length"``/``"_v2"`` prefix every style with ``"<style>:"``."""
    p_multi_point_all_image: float = 0.5
    """Probability that a multi-image pointing question targets "all images" instead of
    a random image subset (mm_olmo stage-2 sets 0.5)."""

    #: System-prompt families that prefix the style name.
    STYLE_AND_LENGTH_FAMILIES = frozenset({"style_and_length", "style_and_length_v2"})

    def style_prefix(self, style: str) -> str:
        if self.system_prompt in self.STYLE_AND_LENGTH_FAMILIES:
            # mm_olmo tags every style under this family (data_formatter.py:1773-1820), demo
            # styles included: pointing and counting as "<style>:", everything else as
            # "<style> <n>:" with n the answer's length // 15 (plus noise) 90% of the time. The
            # number is dropped here, as it is from the caption tag (`pixmo_cap.style_tag_prompt`),
            # so a tag is the same string at training and at test time. That is exactly mm_olmo's
            # molmo3 stage-1 family (`style_and_length_v3`, data_formatter.py:1779-1794), which
            # tags every non-caption style with the bare "<style>:".
            return f"{style}:" if style else ""
        if not style or style in DEMO_STYLES or style in IMAGE_MC_STYLES:
            return ""
        return f"{style}:"

    def select_vqa_answer(self, answers, rng: np.random.RandomState) -> Optional[str]:
        """mm_olmo ``DataFormatter`` answer selection (data_formatter.py:1576-1587).

        Blank answers participate in the "best" vote (and can win) exactly like
        mm_olmo — filtering them would also shift the tie-break rng draw.
        """
        if answers is None or isinstance(answers, str):
            return answers
        if self.select_answer == "first":
            return min(answers)
        if self.select_answer == "best":
            counts = Counter(answers)
            m = max(counts.values())
            cands = [k for k, v in counts.items() if v == m]
            return cands[rng.randint(0, len(cands))]
        raise NotImplementedError(self.select_answer)

    def template_options(
        self, example: Dict[str, Any], is_training: bool, rng: np.random.RandomState
    ) -> Tuple[str, str, Dict[str, Any]]:
        labelled = "options" in example
        allow_unlabelled = True
        if labelled and "answer_idx" in example:
            idx = example["answer_idx"]
            # mm_olmo only inspects the option when the idx is a plain int
            # (data_formatter.py:1070-1073).
            if isinstance(idx, int):
                allow_unlabelled = bool(str(example["options"][idx]).strip())

        if not is_training or rng.random() < 0.1:
            if labelled:
                prefixes = string.ascii_uppercase
                option_text = "\n".join(f"{p}. {o}" for p, o in zip(prefixes, example["options"]))
                option_names = list(prefixes[: len(example["options"])])
                outputs = [f"{n}. {o}" for n, o in zip(option_names, example["options"])]
            else:
                option_text = "\n".join(example["unlabelled_options"])
                option_names = list(example["unlabelled_options"])
                outputs = list(example["unlabelled_options"])
            question = (
                example["question"] + "\nOnly return the correct answer option.\n" + option_text
            )
        else:
            question = example["question"]
            opts = example["options"] if labelled else example["unlabelled_options"]
            # mm_olmo always calls template_mc_question with its default
            # unlabelled=False, for `unlabelled_options` too (data_formatter.py:1101).
            question, option_names, outputs = template_mc_question(
                question,
                opts,
                rng,
                p_label_options=0.8 if allow_unlabelled else 1.0,
            )

        if "answer_idx" in example:
            ans_idx = example["answer_idx"]
            if not (0 <= ans_idx < len(outputs)):
                raise ValueError(f"Invalid answer idx in example: {example}")
            output = outputs[ans_idx]
        else:
            output = None
        return question, output, {"option_names": option_names}

    def format_points(self, example: Dict[str, Any]) -> str:
        import numpy as np

        pts = example.get("points")
        if pts is None:
            xy = []
        elif isinstance(pts, dict) and "x" in pts and "y" in pts:
            xy = np.asarray([pts["x"], pts["y"]], dtype=np.float64).T.reshape(-1, 2)
        elif isinstance(pts, np.ndarray):
            xy = pts
        elif pts and isinstance(pts[0], dict):
            xy = [[p["x"], p["y"]] for p in pts]
        else:
            xy = pts
        # mm_olmo (data_formatter.py:1163-1168): "label" is lowercased,
        # "label_cased" keeps its case, and "question" is the final fallback.
        if "label" in example:
            label = example["label"].lower()
        elif "label_cased" in example:
            label = example["label_cased"]
        else:
            label = example.get("question", "")
        point_scale = example["point_scale"] if "point_scale" in example else 100
        norm = normalize_points(xy, point_scale=point_scale, image_size=example.get("image_size"))
        style = base_pointing_style(example.get("style", "pointing"))
        # mm_olmo's count is always len(points) (data_formatter.py:1141) — never a
        # dataset-provided "count" field, which its formatter cannot even see.
        if style in ("point_count", "point_then_count"):
            return pointing_answer(norm, label, "point_count")
        return pointing_answer(norm, label, "pointing")

    def _format_multi_image_points(
        self, example: Dict[str, Any], rng: np.random.RandomState, is_training: bool
    ) -> Tuple[str, Optional[str]]:
        """Multi-image pointing / counting (port of mm_olmo ``format_multi_points``).

        The example carries parallel per-image lists: ``normalized_labels``, ``labels``,
        and ``points`` (each entry a list of ``{'x','y'}`` dicts or ``(x, y)`` pairs,
        at ``point_scale`` units). One label is drawn, then either "all images" or a
        random image subset is targeted (``p_multi_point_all_image``); the answer
        serializes the matching points of the targeted images with 1-based image
        indices and point ids continuing across images.

        Deviation from mm_olmo (documented): unique labels keep first-occurrence order
        (``dict.fromkeys``) instead of ``list(set(...))``, whose iteration order depends
        on ``PYTHONHASHSEED``; the rng draw count is unchanged.
        """
        style = example["style"]
        norm_labels = list(example["normalized_labels"])
        labels = list(example.get("labels", norm_labels))
        points = list(example["points"])

        unique_labels = list(dict.fromkeys(norm_labels))
        selected_norm = rng.choice(unique_labels)
        selected_label = next(
            (labels[i] for i, nl in enumerate(norm_labels) if nl == selected_norm),
            selected_norm,
        )

        all_images = list(range(len(norm_labels)))
        n_images = len(all_images)
        label_exists = any(
            nl == selected_norm and len(points[i]) > 0 for i, nl in enumerate(norm_labels)
        )

        if self.p_multi_point_all_image:
            if n_images == 1 or rng.random() < self.p_multi_point_all_image:
                selected_images = "all images"
            else:
                n_select = rng.randint(1, n_images)
                chosen = rng.choice(all_images, n_select, replace=False)
                selected_images = ", ".join(f"image_{i + 1}" for i in chosen)
        else:
            n_select = rng.randint(1, n_images) if n_images >= 1 else n_images
            chosen = rng.choice(all_images, size=n_select, replace=False)
            if n_select == n_images and rng.random() < 0.5:
                selected_images = "all images"
            else:
                selected_images = ", ".join(f"image_{i + 1}" for i in chosen)

        pool = (
            MULTI_IMAGE_POINTING_PROMPTS
            if style == "multi_image_pointing"
            else MULTI_IMAGE_POINT_COUNT_PROMPTS
        )
        if selected_images == "all images" and style == "multi_image_pointing":
            # 50% chance to phrase an all-images question with the single-image pool.
            if rng.random() < 0.5:
                question = POINTING_PROMPTS[rng.randint(len(POINTING_PROMPTS))].format(
                    label=selected_label
                )
            else:
                question = pool[rng.randint(len(pool))].format(
                    selected_images=selected_images, selected_label=selected_label
                )
        else:
            question = pool[rng.randint(len(pool))].format(
                selected_images=selected_images, selected_label=selected_label
            )

        if not is_training:
            return question, None
        if not label_exists:
            return question, "There are none."

        if selected_images == "all images":
            valid = [
                i for i in all_images if norm_labels[i] == selected_norm and len(points[i]) > 0
            ]
        else:
            chosen_idx = [int(s.split("_")[1]) - 1 for s in selected_images.split(", ")]
            valid = [
                i
                for i in chosen_idx
                if i < len(norm_labels) and norm_labels[i] == selected_norm and len(points[i]) > 0
            ]
        if not valid:
            return question, "There are none."

        scale = float(example.get("point_scale", 100))
        by_image = []
        for i in valid:
            xy = [(p["x"], p["y"]) if isinstance(p, dict) else (p[0], p[1]) for p in points[i]]
            # clip_points clamping is equivalent to the [0, 1] clamp inside
            # _scale_point after normalization.
            by_image.append((i + 1, [(x / scale, y / scale) for x, y in xy]))
        return question, multi_image_pointing_answer(by_image, selected_label, style)

    def _needs_formatted_message(self, msg: Dict[str, Any]) -> bool:
        style_str = str(msg.get("style", ""))
        return (
            msg.get("explanation") is not None
            or style_str.endswith("_exp")
            or style_str.startswith("cosyn_")
        )

    def format_branches(
        self,
        example: Dict[str, Any],
        *,
        is_training: bool = True,
        index: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[List[Tuple[str, str]]]:
        """Return the annotation branches of one example.

        Each branch is a sequential list of ``(user_text, assistant_text)`` turns:
        independent annotations (``message_list`` entries) become separate branches
        (attention-isolated subsegments), while a ``{"messages": [...]}`` conversation
        stays ONE branch so later turns attend earlier ones (mm_olmo
        ``_format_example``, data_formatter.py:2013-2019). The style prefix applies to
        a branch's first user turn only (mm_olmo data_formatter.py:2072-2075).
        """
        if rng is None:
            from .sequence_builder import example_rng

            rng = example_rng(self.seed, index)
        style = example.get("style", "")

        if "message_list" in example:
            branches: List[List[Tuple[str, str]]] = []
            for msg in example["message_list"]:
                if "messages" in msg:
                    messages = msg["messages"]
                    conv = [(messages[u], messages[u + 1]) for u in range(0, len(messages) - 1, 2)]
                    conv = [(q, a) for q, a in conv if a and str(a).strip()]
                    if conv:
                        # Style prefix on the first user turn of the conversation.
                        prefix = self.style_prefix(str(msg.get("style", style)))
                        if prefix:
                            q0, a0 = conv[0]
                            conv[0] = ((f"{prefix} " if q0 else prefix) + q0, a0)
                        branches.append(conv)
                else:
                    # mm_olmo passes only the sub-message (plus the media fields) to
                    # the formatter — not the whole top-level example.
                    sub = {
                        k: v
                        for k, v in example.items()
                        if k in ("image", "metadata", "point_scale", "clip_points", "image_size")
                    }
                    sub.update(msg)
                    branches.extend(
                        self.format_branches(sub, is_training=is_training, index=index, rng=rng)
                    )
            valid = [b for b in branches if b]
            if is_training and not valid:
                raise ValueError("No valid (question, answer) branches")
            return valid

        output: Optional[str] = None
        if style in MULTI_IMAGE_POINTING_STYLES:
            prompt, output = self._format_multi_image_points(example, rng, is_training)
            if is_training and (output is None or not str(output).strip()):
                raise ValueError("No valid output in example")
            return [[(prompt, output)]]
        elif "prompt" in example:
            # mm_olmo honors an explicit "prompt" before any templating
            # (data_formatter.py:1753-1755).
            prompt = example["prompt"]
        elif style in ("long_caption", "short_caption") and "question" not in example:
            pool = SHORT_CAPTION_PROMPTS if style == "short_caption" else CAPTION_PROMPTS
            prompt = pool[rng.randint(len(pool))]
            output = example.get("text") or example.get("caption", "")
        elif style in POINTING_STYLES:
            if "question" in example:
                prompt = example["question"]
            else:
                # mm_olmo (data_formatter.py:1851-1857): "label" keeps its original
                # case 50% of the time (one rng draw); "label_cased" is never lowered.
                if "label" in example:
                    label = example["label"]
                    if rng.random() > 0.5:
                        label = label.lower()
                else:
                    label = example["label_cased"]
                if self.prompt_templates == "none":
                    # mm_olmo data_formatter.py:1770-1779: no templating or instructions,
                    # just the lowercased label (the "<style>:" prefix is added separately).
                    prompt = (
                        example["label"].lower() if "label" in example else example["label_cased"]
                    )
                else:
                    pool = (
                        POINT_COUNT_PROMPTS
                        if style in ("point_count", "point_then_count")
                        else POINTING_PROMPTS
                    )
                    prompt = pool[rng.randint(len(pool))].format(label=label)
            if not is_training:
                output = None
            else:
                output = self.format_points(example)
        elif "question" in example and ("options" in example or "unlabelled_options" in example):
            if self.prompt_templates == "none":
                # mm_olmo lays the options out with `format_options` under this family
                # (data_formatter.py:1927-1928), not with the sampled templates below, and that
                # layout is not ported: refuse rather than train a stage-2 prompt in stage 1.
                raise NotImplementedError(
                    "multiple-choice questions have no prompt_templates='none' port "
                    f"(style {style!r})"
                )
            prompt, output, _ = self.template_options(example, is_training, rng)
        elif "question" in example:
            prompt = example["question"]
        else:
            prompt = ""

        # The chain-of-thought instruction is one of mm_olmo's templates
        # (`GENERAL_PROMPTS_V1["chain_of_thought"]`, data_formatter.py:1979-1980), so it is only
        # applied by the templated families. Under "none" the question is sent bare
        # (data_formatter.py:1933-1934): the `*_exp` style tag is what asks for the explanation.
        if "_exp" in style and prompt and self.prompt_templates != "none":
            prompt = _apply_chain_of_thought_prompt(prompt)

        if output is None and is_training:
            if "answers" in example:
                output = self.select_vqa_answer(example["answers"], rng)
            elif "answer" in example and "explanation" in example:
                output = f"{example['explanation']} Answer: {example['answer']}"
            elif "answer" in example:
                output = example["answer"]
            elif "text" in example:
                output = example["text"]

        if is_training and (output is None or not str(output).strip()):
            raise ValueError("No valid output in example")

        prefix = self.style_prefix(style)
        user_text = (f"{prefix} " if prefix and prompt else prefix) + prompt
        return [[(user_text, output)]]

    def format_turns(
        self,
        example: Dict[str, Any],
        *,
        is_training: bool = True,
        index: int = 0,
        rng: Optional[np.random.RandomState] = None,
    ) -> List[Tuple[str, str]]:
        """Flattened view of :meth:`format_branches` (single-turn branches only).

        Kept for callers that treat every turn as an independent branch; use
        :meth:`format_branches` to preserve multi-turn conversations.
        """
        branches = self.format_branches(example, is_training=is_training, index=index, rng=rng)
        return [turn for branch in branches for turn in branch]
