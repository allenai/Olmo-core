"""
Molmo2 "stage 1" caption-pretraining (reproduction of ``mm_olmo``'s captioner).

Trains the connector, vision encoder and LM on PixMoCap captions/transcripts as three
separate optimizer groups, matching mm_olmo's captioner (``ft_connector``/``ft_vit``/
``ft_llm`` all true): connector lr 2e-4 warmup 200, ViT lr 6e-6 warmup 2000, LM lr 2e-5
warmup 2000, all cosine to ``alpha_f=0.1``. Uses the float ``root_subsegments``-weighted
loss. ``--train_vit=false`` freezes the encoder and trains two groups instead. In-loop
evaluation is intentionally omitted.

Weights start from **base** checkpoints, matching mm_olmo's ``reset_with_pretrained_weights``:
the LM from the base Qwen3 backbone, the ViT from ``google/siglip2-so400m-patch14-384``, and
the connector / extra-token embeddings randomly initialised (``INIT_FROM = "scratch"``). This
is stage-1 *pretraining*. Passing ``--init_from=molmo2`` instead continues the released,
already-post-trained Molmo2 checkpoint on the stage-1 objective, which is a fine-tune —
useful for parity tests and continuation experiments, but not a stage-1 reproduction.

``--model_size`` selects the variant: ``4b`` (Qwen3-4B, the default) or ``8b`` (Qwen3-8B).

``--recipe`` selects the data mixture (see :data:`RECIPES`): ``v1`` (the default) is the released
Molmo2-4B-Pretrain mixture -- caption 0.6, pointing and counting 0.3 on the v1 sources, Tulu
text 0.1, no OCR. ``v2`` is caption 0.5, pointing and counting 0.25 on the v2 sources, OCR 0.25
and no text-only data. ``v3`` is v2 with an academic QA group, a clock-reading group and more OCR,
for 50,000 steps rather than 32,000: caption 0.33, pointing and counting 0.16, OCR 0.34, academic
QA 0.14, clocks 0.03. A recipe only sets the defaults of ``pointing_rate`` / ``nlp_rate`` /
``ocr_rate`` / ``academic_rate`` / ``clock_rate`` / ``pointing_data`` and the step count; an
explicit override of any of them still wins.

``--pointing_data`` selects the pointing/counting group: ``v1`` (the released Molmo2 pretrain's
sources, the default) or ``v2`` (mm_olmo's molmo3 stage-1 sources: the audited, image-grouped
PixMo-Points build with sub-sampled absence queries, the audited PixMo-Count build and the audited
CoSyn build, plus MolmoPoint-GUISyn's synthetic GUI screenshots). The v2 knobs are the
``pointing_v2`` / ``count_v2`` / ``gui_syn`` config fields, e.g. ``--pointing_v2.filter_audit=true``;
``--gui_pointing=false`` leaves GUISyn out.

``--ocr_rate`` (default 0) adds the OCR group, paid for by the caption group: olmOCR-mix page
transcription (rendered from PDFs, needs ``pypdfium2``), TextOCR scene text, two synthetic
English sets (NVIDIA's scattered scene-style text, needs ``h5py``, and thermal receipts), and
mm_olmo's three-level captions of text-rich figures. The user turn of every OCR source is a
style tag with no question. The rate is split evenly between transcription and figure captions,
then by sqrt(size) within each. Every default source is a train split only; the scene-text tars
whose split cannot be verified are selectable but not default. ``--ocr_sources=[...]`` picks the
sources (see :mod:`olmo_core.data.multimodal.mixtures.ocr`); the ``olmocr`` / ``ocr_tars`` /
``text_rich`` config fields are the three source templates, e.g. ``--olmocr.languages=null``,
``nvidia_synth`` / ``receipts`` configure the synthetic sets, and ``--ocr_data_root`` relocates
the tar tree.

``--academic_rate`` (default 0; 0.14 in ``v3``) adds stage 2's large synthetic QA sets, also paid
for by the caption group: CoSyn's seven single-image categories, DVQA, FigureQA and PlotQA, each
question behind its source's style tag (``dv_qa: <question>``). The training set of a benchmark
the stage-2 checkpoints are evaluated on is never a stage-1 source, nor is TallyQA, whose images
overlap the VQAv2 eval's. ``--academic_sources=[...]`` picks the sources (see
:mod:`olmo_core.data.multimodal.mixtures.stage1_academic`).

``--clock_rate`` (default 0; 0.03 in ``v3``) adds PixMo-Clocks as a group of its own, also paid
for by the caption group: synthetic clock faces, with ``clocks: What time is being shown?`` as the
user turn and the time as the answer.

Run without arguments for usage. Quick local smoke test on synthetic data::

    torchrun --nproc-per-node=1 src/scripts/train/Molmo2-Stage1.py train smoke \\
        --dataset.dataset_path=synthetic --trainer.max_duration.value=5 \\
        --trainer.max_duration.unit=steps

.. note::
    Single-GPU, DDP, and FSDP/HSDP data parallelism are supported. TP/CP/PP/EP of the
    multimodal model are out of scope.
"""

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Sequence, Tuple, cast

from olmo_core.config import Config, DType
from olmo_core.data.multimodal import (
    CoSynPointDatasetConfig,
    GuiSynDatasetConfig,
    MixtureDataLoader,
    MultimodalCollatorConfig,
    MultimodalDataLoader,
    NvidiaSynthOcrDatasetConfig,
    OcrCaptionTarsDatasetConfig,
    OlmOcrMixDatasetConfig,
    PixMoCapDatasetConfig,
    PixMoCountDatasetConfig,
    PixMoCountV2DatasetConfig,
    PixMoPointsDatasetConfig,
    PixMoPointsV2DatasetConfig,
    SyntheticReceiptsDatasetConfig,
    TextRichCaptionDatasetConfig,
    Tulu4DatasetConfig,
)
from olmo_core.data.multimodal.mixtures.ocr import (
    DEFAULT_OCR_SOURCES,
    DUPLICATE_OLMOCR_SOURCES,
    OCR_SOURCE_NAMES,
    SPLIT_UNVERIFIED_SOURCES,
    build_ocr_source,
    ocr_task,
    ocr_task_shares,
    ocr_weighting_sizes,
)
from olmo_core.data.multimodal.mixtures.stage1_academic import (
    CLOCKS_SOURCE,
    DEFAULT_ACADEMIC_SOURCES,
    STAGE1_ACADEMIC_SOURCE_NAMES,
    STAGE2_EVAL_TRAIN_SETS,
    academic_weighting_sizes,
    build_stage1_academic_source,
    build_stage1_clocks_source,
)
from olmo_core.data.multimodal.olmocr import canonical_split
from olmo_core.data.multimodal.paths import OE_ENCODER_DATA, PIXMO_DATASETS
from olmo_core.data.multimodal.pixmo_points import COSYN_POINT_V2_PATH
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.distributed.utils import get_rank, get_world_size
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.internal.common import (
    build_launch_config,
    get_beaker_username,
    get_root_dir,
)
from olmo_core.launch.beaker import BeakerEnvVar, BeakerLaunchConfig
from olmo_core.nn.transformer.config import TransformerActivationCheckpointingMode
from olmo_core.nn.vision import MultimodalLM, MultimodalLMConfig
from olmo_core.optim import (
    AdamWConfig,
    CosWithWarmup,
    OptimGroupOverride,
    PerGroupScheduler,
)
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_cli_environment,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    BeakerCallback,
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
    UnshardedModelExportCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    MultimodalTransformerTrainModuleConfig,
    TransformerActivationCheckpointingConfig,
    TransformerDataParallelConfig,
)
from olmo_core.utils import get_default_device, seed_all

log = logging.getLogger(__name__)

#######################
#### CONFIGURATION ####
#######################

# Model size. Selects the architecture factory, the base LM to initialise from, and the
# released checkpoint used by `--init_from=molmo2`. Override with `--model_size=8b`.
MODEL_SIZE = "4b"


@dataclass(frozen=True)
class ModelSizeSpec:
    """Weight sources and RoPE bases for one Molmo2 variant."""

    factory: Callable[..., MultimodalLMConfig]
    """Native architecture factory, e.g. :meth:`MultimodalLMConfig.molmo2_4B`."""

    molmo2_id: str
    """Released checkpoint, used by ``--init_from=molmo2``."""

    scratch_lm_id: str
    """Base Qwen3 backbone, used by ``--init_from=scratch``."""

    scratch_rope_theta: int
    """RoPE base of ``scratch_lm_id``'s weights."""

    molmo2_rope_theta: int
    """RoPE base of ``molmo2_id``'s weights."""

    def rope_theta(self, init_from: str) -> int:
        return self.scratch_rope_theta if init_from == "scratch" else self.molmo2_rope_theta


# Per-variant weight sources, taken from the released pretrain configs at
# /weka/oe-training-default/mm-olmo/released-models-molmo2-1225/Molmo2-{4B,8B}-Pretrain/.
# `rope_theta` follows the weight source, and the two variants used different backbones.
MODEL_SIZES = {
    "4b": ModelSizeSpec(
        factory=MultimodalLMConfig.molmo2_4B,
        molmo2_id="allenai/Molmo2-4B",
        # The released 4B pretrain ran `train_captioner.py qwen3_4b_instruct`, i.e. the
        # *instruct* backbone (`Qwen/Qwen3-4B-Instruct-2507`), whose RoPE base is 5e6 — which
        # is where the released Molmo2-4B's 5e6 comes from. Its tokenizer is identical to
        # Molmo2's for everything the data pipeline uses (BPE, eos_token_id 151645, and the
        # user-turn chat template).
        scratch_lm_id="Qwen/Qwen3-4B-Instruct-2507",
        scratch_rope_theta=5_000_000,
        molmo2_rope_theta=5_000_000,
    ),
    "8b": ModelSizeSpec(
        factory=MultimodalLMConfig.molmo2_8B,
        molmo2_id="allenai/Molmo2-8B",
        # The released 8B pretrain ran `train_captioner.py qwen3_8b` — the *base* backbone,
        # RoPE base 1e6, untied. Hence the 4B/8B asymmetry: different backbones, not a
        # Molmo2-side change.
        scratch_lm_id="Qwen/Qwen3-8B",
        scratch_rope_theta=1_000_000,
        molmo2_rope_theta=1_000_000,
    ),
}

# Weight init. "scratch" is mm_olmo's true stage-1 start and the default: base Qwen3 LM
# + base SigLIP2 ViT + randomly-initialised connector / new-token embeddings. "molmo2"
# instead *continues* the released (already post-trained) Molmo2 checkpoint on the stage-1
# objective — that is a fine-tune, not stage-1 pretraining, so it is opt-in only
# (`--init_from=molmo2`), kept for parity tests and continuation experiments.
INIT_FROM = "scratch"
INIT_FROM_CHOICES = ("scratch", "molmo2")
_BOOLS = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
SCRATCH_VIT_ID = "google/siglip2-so400m-patch14-384"  # shared by every Molmo2 variant
NEW_EMBEDDING_INIT_STD = 0.02  # mm_olmo `new_embedding_init_range`
SEQUENCE_LENGTH = 2560  # fixed pad length; the released runs passed `--seq_len=2560`
USE_FLEX_ATTN = True  # fused FlexAttention backend for the multimodal masks (~+8% MFU)
PACK_SEQUENCES = True  # pack several examples per sequence
# mm_olmo `packing: {mode: dynamic_solver, buffer_size: 48, text_weight: 1.0,
# image_weight: 1.0}` — a 2D knapsack over (text tokens, image crops) rather than the
# token-only next-fit we used before. The crop budget is the second knapsack dimension: an
# 8-crop example costs at most 9 crops (1 global + 8 high-res) and ~1348 image tokens, so at
# seq 2560 the token budget binds first and this mainly bounds worst-case ViT/collator
# memory. Set below a single example's 9 crops it would force that example into its own
# mostly-padding pack.
PACK_MAX_CROPS = 3 * (1 + 8)
PACK_BUFFER_SIZE = 48
PACK_IMAGE_WEIGHT = 1.0
COMPILE_MODEL = True  # torch.compile the LM (fuses pointwise ops; one-time compile warmup)
DATA_PREFETCH_WORKERS = 4  # background threads preprocessing examples (0 = synchronous)
MAX_CROPS = 8

# mm_olmo applies 0.1 dropout to the residual stream of RESPONSE tokens only (prompt and
# image tokens get 0.0), via a per-token drop mask in its LM block (`Dropout(mask_p=...)`).
# Implemented as `ResidualStream(masked_dropout=...)`; the mask is derived from `loss_masks`.
RESPONSE_RESIDUAL_DROPOUT = 0.1

# Instance-based batching, expressed in tokens for the token-based Trainer.
#
# `GLOBAL_BATCH_INSTANCES` matches mm_olmo's `global_train_batch_size=128` — the
# optimization-relevant quantity. 128 also divides every DP world size we run (8/16/32),
# which the previous default of 8 did not: at 16 GPUs it failed the
# `global % (microbatch x dp_world_size) == 0` check.
#
# mm_olmo uses `device_train_microbatch_size=4`; we keep 1. That is purely a
# memory/throughput knob — gradient accumulation makes the two mathematically identical for
# the same global batch — and microbatch >1 at this sequence length OOM'd without activation
# checkpointing (see the `ac_config` follow-up).
GLOBAL_BATCH_INSTANCES = 128
# mm_olmo's `device_train_microbatch_size: 4`, which its `activation_checkpointing: true`
# pays for. NOTE 128 % (4 x dp_world_size) must be 0, so this supports world sizes up to 32
# (the released 4B ran 16 GPUs, the 8B 32); at 64 use microbatch 2.
RANK_MICROBATCH_INSTANCES = 4
GLOBAL_BATCH_SIZE = GLOBAL_BATCH_INSTANCES * SEQUENCE_LENGTH
RANK_MICROBATCH_SIZE = RANK_MICROBATCH_INSTANCES * SEQUENCE_LENGTH

# Per-component LRs / warmups (mm_olmo train_captioner.py).
CONNECTOR_LR = 2e-4
LLM_LR = 2e-5
CONNECTOR_WARMUP = 200
LLM_WARMUP = 2000
# Train the ViT as well (mm_olmo captioner `ft_vit=True`), giving it its own optimizer group.
# mm_olmo's vit_betas/vit_eps/vit_weight_decay equal the LLM's, so the group only overrides
# the learning rate and inherits betas (0.9, 0.95) / eps 1e-6 / wd 0.0 from `AdamWConfig`.
# Set False to freeze the encoder and train connector + LM only.
TRAIN_VIT = True
VIT_LR = 6e-6
VIT_WARMUP = 2000
# Hold the pretrained token embeddings fixed, matching mm_olmo's `ft_embedding="lm_head"`
# default: only the 128 image-special rows (`embeddings.extra_weight`) are learned. Because
# the table is split, this is a plain whole-tensor freeze. For a *tied* backbone (Molmo2-4B)
# the base block is also the LM head, so both roles are frozen; for an untied one
# (Molmo2-8B) only the input table is, and the head keeps training — exactly mm_olmo.
FREEZE_BASE_EMBEDDINGS = True
ALPHA_F = 0.1

# Data: the canonical PixMoCap "cap" dataset (HF DatasetDict, load_from_disk). Override as needed.
DATASET_PATH = f"{PIXMO_DATASETS}/cap"
MAX_STEPS = 32000

# Stage-1 mixture rates (mm_olmo train_captioner --pointing/--nlp). Caption gets the
# Prompt family for the pointing/counting data, from the released Molmo2-4B-Pretrain
# `data_formatter` (`prompt_templates: none`, `system_prompt: style_and_length_v2`): the question
# is the bare lowercased label behind a `"<style>:"` prefix, not a natural-language template.
# The formatter defaults to the stage-2 family, which trains a model that is then out of
# distribution for pointing evals -- worth ~11 f1 on pixmo_point_eval_v3_mp, most of it
# abstention, because the "Please say 'There are none.'" instruction only exists in the
# stage-2 template.
# Caption loss weight. The released Molmo2-4B-Pretrain leaves every source at 1.0, which puts
# captions at ~77.5% of the sum(CE*w)/sum(w) loss mass; 1.25 lifts that to ~81%. Measured against
# two-seed baselines: dense_caption avg 57.271 -> 57.808 (+0.537, seed spread 0.085) and
# consistency 69.973 -> 70.701 (released is 70.745), for pointing costs of ~0.009-0.013 f1 at
# seed spreads of 0.008-0.011. Override with `--caption_message_weight=1.0` to reproduce the
# released weighting exactly.
CAPTION_MESSAGE_WEIGHT = 1.25

POINTING_DATASET_KWARGS = {
    "prompt_templates": "none",
    "system_prompt": "style_and_length_v2",
    # The pointing/counting dataset classes default to `root_subsegments`, which scales an
    # example's loss by 1/sqrt(n_labels). The released run leaves this unset for every dataset
    # (`mm_preprocessor.loss_token_weighting: None`, `message_weight: None` on all four pointing
    # entries), so it weights every response token equally. Our pointing rows carry ~12.5 labels
    # per example and counting ~2.8, so the default silently gave that data ~3.5x / ~1.7x less
    # gradient weight per token than captions -- for exactly the reason the caption dataset's
    # own comment gives: the factor does not cancel out of the global sum(CE*w)/sum(w) divisor
    # when branch counts differ across examples.
    "loss_token_weighting": "none",
}

# remainder (1 - POINTING_RATE - NLP_RATE - OCR_RATE - ACADEMIC_RATE - CLOCK_RATE). Set them all to
# 0.0 for a caption-only run.
POINTING_RATE = 0.30
NLP_RATE = 0.10
# The OCR group (`olmo_core.data.multimodal.mixtures.ocr`): general OCR data only, i.e. an image
# in and its text (or a description of a text-rich figure) out, with no question in the prompt.
# It is mm_olmo's two molmo3 stage-1 OCR groups (`train_molmo3_stage1._base_mixture`, 0.075 each)
# -- olmOCR-mix page transcription and the three-level figure captions -- plus TextOCR scene
# text and two synthetic English transcription sets: NVIDIA OCR-Synthetic (`synth_ocr`) and
# thermal receipts (`receipt_ocr`). The rate is split evenly between the two tasks,
# transcription and figure captions, as mm_olmo's two groups are, then by sqrt(size) within a
# task (mm_olmo's `root_size_factor`), with a synthetic source weighted as if no larger than the
# largest real source of its task (NVIDIA's 1.46M images count as olmOCR-mix documents' 219k).
# Paid for out of the caption group. Off by default so the default run stays the released
# Molmo2 pretrain mixture; `--ocr_rate=0.15` enables it at mm_olmo's total.
# `DEFAULT_OCR_SOURCES` holds train splits only. It leaves out the `s2pdf` / `iabooks` tars, which
# are the same pages as olmOCR-mix documents / books, and the HierText / COCO-Text / UberText
# tars, whose split cannot be verified. olmOCR-mix pages are rendered from PDFs at load time,
# which needs `pypdfium2` (installed by the launch `post_setup` below).
OCR_RATE = 0.0
OCR_SOURCES = DEFAULT_OCR_SOURCES
# The academic QA group (`olmo_core.data.multimodal.mixtures.stage1_academic`): stage 2's large
# synthetic QA sets, trained in stage 1 as well so a stage 1 mixed into text midtraining has more
# distinct images than PixMo-Cap's. CoSyn's seven single-image categories (357k images), DVQA
# (200k), PlotQA (157k) and FigureQA (100k), weighted at stage 2's row caps so the group's
# sqrt(size) split gives the three templated chart sets 19% of it. Each question is the bare
# question behind its source's style tag. Never a source: the training set of a benchmark the
# stage-2 checkpoints are evaluated on (VQAv2, TextVQA, ChartQA, DocVQA, InfographicVQA, AI2D,
# A-OKVQA), or TallyQA, whose images overlap the VQAv2 eval's. Paid for out of the caption group;
# off unless given, and 0.14 in the v3 recipe.
ACADEMIC_RATE = 0.0
ACADEMIC_SOURCES = DEFAULT_ACADEMIC_SOURCES
# The clock-reading group (`mixtures.stage1_academic.build_stage1_clocks_source`): PixMo-Clocks'
# 800k synthetic clock faces, a group of its own rather than an academic source because it is one
# narrow skill with one fixed question. Its own rate caps how much of it a run sees whatever the
# QA group's size. Its user turn keeps the question, `clocks: What time is being shown?`: stage 1
# may become the only place the model learns to read clocks, so the question text is trained too.
# Paid for out of the caption group; off unless given, and 0.03 in the v3 recipe.
CLOCK_RATE = 0.0

# Which sources `POINTING_RATE` buys.
#   "v1": the released Molmo2 pretrain's group (mm_olmo train_captioner.py `--pointing`):
#         pixmo_points_train, pixmo_count_train, pixmo_points_high_freq_train, cosyn_point,
#         split within the group by sqrt(size).
#   "v2": mm_olmo's molmo3 stage-1 group (launch_scripts/train_molmo3_stage1.py
#         `_base_mixture`): the audited, image-grouped PixMo-Points build with absence queries
#         (`PixMoPointsV2DatasetConfig`), the audited PixMo-Count build
#         (`PixMoCountV2DatasetConfig`) and the audited CoSyn build (`cosyn_point_v2`, mm_olmo's
#         `CoSynPointConfigV2` without its masks), split *linearly* by size (mm_olmo
#         `size_weighted=1`). The one member of that group with no port is `CocoTrain`, which
#         is a SEGMENTATION source, not detection-as-pointing: under `style="pixmo"` it emits
#         `pixmo_seg` and its messages carry `bboxes` + RLE `segmentations` with no points
#         (mm_olmo academic_datasets.py:2175-2189, data_formatter.py:965-968). olmo-core has no
#         segmentation path, so it stays unported -- as do PixMoPointV2's own mask branches.
#         On top of mm_olmo's group, v2 adds MolmoPoint-GUISyn (`gui_syn`, mm_olmo's
#         `molmo2_syn_point`, which mm_olmo trains only in its GUI-only mixtures): synthetic
#         desktop / mobile / web screenshots, one point per sampled UI element, asked by an
#         interaction intent (`gui_point:`) or by the element's name (`pointing:`). It joins
#         the linear split by image count; `--gui_pointing=false` leaves it out.
# The v2 knobs are the `pointing_v2` / `count_v2` / `gui_syn` config fields
# (`--pointing_v2.<field>=...`).
POINTING_DATA = "v1"
POINTING_DATA_CHOICES = ("v1", "v2")
# mm_olmo `_base_mixture` settings for the v2 sources. Audit-failed point sets are kept but
# rendered behind the `aux_*` marker styles (set `filter_audit=true` to drop them instead;
# both work). The absence queries are SUB-SAMPLED -- 2 easy negatives per image and a quarter
# of the paired (hard) negatives per epoch: training on all of them makes the model prone to
# refusing to point. `v2_paired_negatives` follows the dataset class default (mm_olmo's stage-1
# arms flip it per trainer).
POINTING_V2_AUDIT_STYLE = ("aux_point_count", "aux_pointing")
POINTING_V2_FILTER_AUDIT = False
POINTING_V2_N_EASY_NEGATIVES = 2
POINTING_V2_P_PAIRED_NEGATIVES = 0.25
# The audited CoSyn build's failed questions (18% of them) are kept behind `aux_cosyn_point:`.
# mm_olmo tags them `aux_pointing:`; a tag of their own keeps `aux_pointing:` followed by an
# object name, as `pointing:` is. Its agent masks feed segmentation, which is not trained here.
POINTING_V2_COSYN_AUDIT_STYLE = "aux_cosyn_point"
# Whether the v2 pointing group includes MolmoPoint-GUISyn (`gui_syn`). Its intents are English
# requests, so they get their own `gui_point:` tag, as CoSyn's do. Desktop screenshots carry ~89
# elements each, so `gui_syn.max_elements` (16) are sampled per image and epoch; they are
# downscaled to the stage-1 crop budget (`MAX_CROPS`), so ~12-pixel targets on 1920x1080
# desktop shots lose detail.
GUI_POINTING = True

# Data recipes: the group rates and the pointing sources, selected with `--recipe`. The caption
# group gets the remainder, 1 - pointing_rate - nlp_rate - ocr_rate - academic_rate - clock_rate. A
# recipe only sets defaults: `--recipe=v2 --ocr_rate=0.2` is v2 with a 0.2 OCR group (and caption
# at 0.55).
#   "v1": the released Molmo2-4B-Pretrain mixture: caption 0.6, pointing and counting 0.3 (v1
#         sources), Tulu text 0.1, no OCR.
#   "v2": caption 0.5, pointing and counting 0.25 (v2 sources: audited PixMo-Points / PixMo-Count
#         + CoSyn, plus GUISyn GUI screenshots), OCR 0.25 (the default OCR sources, see
#         `OCR_RATE`), no text-only data.
#   "v3": v2 plus the academic QA group (`ACADEMIC_RATE`) and the clock group (`CLOCK_RATE`), with
#         more OCR, for 50,000 steps (`RECIPE_MAX_STEPS`): caption 0.33, pointing and counting
#         0.16, OCR 0.34, academic QA 0.14, clocks 0.03. The OCR rate is what one pass over the
#         OCR group takes in 50k steps: from 80 sampled examples per source, packed by this
#         script's 2D-knapsack packer at 2,560 tokens, a run is 1.41 examples per sequence, 9.0M
#         examples, 3.07M of them OCR for its 3.09M rows. Within the group the split is still by
#         task, then sqrt(size) with the synthetic sets capped (`_ocr_fractions`), so that pass
#         is uneven: the figure captions ~1 pass, olmOCR documents 2.1, the small olmOCR subsets,
#         TextOCR and the receipts 7-10, NVIDIA's 1.46M synthetic images 0.32. Caption and
#         pointing keep about v2's passes (4.2 over PixMo-Cap, v2 4.0; 4.0 over the pointing
#         sources with GUISyn, v2 4.0, and 4.4 for both with `--gui_pointing=false`); the
#         academic group gets 1.55 passes and the clock group 271k examples (0.34 passes). The
#         expected weighted loss splits caption / pointing / OCR / academic as 39.2% / 12.8% /
#         38.9% / 9.1% (v2: 55.1% / 18.5% / 26.5% / 0), clocks 0.08%, measured without GUISyn.
RECIPES = {
    "v1": dict(
        pointing_rate=POINTING_RATE,
        nlp_rate=NLP_RATE,
        ocr_rate=OCR_RATE,
        academic_rate=ACADEMIC_RATE,
        clock_rate=CLOCK_RATE,
        pointing_data="v1",
    ),
    "v2": dict(
        pointing_rate=0.25,
        nlp_rate=0.0,
        ocr_rate=0.25,
        academic_rate=0.0,
        clock_rate=0.0,
        pointing_data="v2",
    ),
    "v3": dict(
        pointing_rate=0.16,
        nlp_rate=0.0,
        ocr_rate=0.34,
        academic_rate=0.14,
        clock_rate=0.03,
        pointing_data="v2",
    ),
}
RECIPE = "v1"
# Training steps a recipe runs for, where it differs from `MAX_STEPS`: v3's OCR rate is set to cover
# the OCR group once in 50k steps. An explicit `--trainer.max_duration.value` still wins.
RECIPE_MAX_STEPS = {"v3": 50_000}

# Beaker.
BEAKER_CLUSTER = "ai2/jupiter"
NUM_NODES = 1
BEAKER_WORKSPACE = "ai2/OLMo-core"
BEAKER_BUDGET = "ai2/oe-other"

# Logging. Set WANDB_PROJECT to None to disable W&B (requires the WANDB_API_KEY secret
# in the Beaker workspace). Metrics always go to the console regardless.
# WANDB_ENTITY=None uses the API key's default entity (personal account), avoiding 403s
# from writing to a team the key lacks access to; set it to a team you can write to.
WANDB_PROJECT: Optional[str] = "molmo2-stage1"
WANDB_ENTITY: Optional[str] = None

###########################
#### END CONFIGURATION ####
###########################


@dataclass
class ExperimentConfig(Config):
    launch: BeakerLaunchConfig
    model: MultimodalLMConfig
    dataset: PixMoCapDatasetConfig
    collator: MultimodalCollatorConfig
    train_module: MultimodalTransformerTrainModuleConfig
    trainer: TrainerConfig
    pointing_v2: PixMoPointsV2DatasetConfig
    """The audited PixMo-Points source; used when ``pointing_data == "v2"``."""
    count_v2: PixMoCountV2DatasetConfig
    """The audited PixMo-Count source; used when ``pointing_data == "v2"``."""
    gui_syn: GuiSynDatasetConfig
    """The MolmoPoint-GUISyn GUI-pointing source; used when ``pointing_data == "v2"`` and
    ``gui_pointing``."""
    olmocr: OlmOcrMixDatasetConfig
    """Template for the olmOCR-mix OCR sources (``subset`` is set per source); used when
    ``ocr_rate > 0``."""
    ocr_tars: OcrCaptionTarsDatasetConfig
    """Template for the caption-tars OCR sources; used when ``ocr_rate > 0``. One template
    serves every tar source, so ``dataset_path`` / ``style`` / ``strip_text_tags`` are set per
    source from the registry and setting them here is refused -- relocate the whole tree with
    ``--ocr_data_root`` instead."""
    text_rich: TextRichCaptionDatasetConfig
    """Template for the figure-caption OCR sources (``category`` is set per source); used when
    ``ocr_rate > 0``."""
    nvidia_synth: NvidiaSynthOcrDatasetConfig
    """The ``nvidia_synth_en`` OCR source; used when it is in ``ocr_sources`` and
    ``ocr_rate > 0``."""
    receipts: SyntheticReceiptsDatasetConfig
    """The ``synth_receipts_en`` OCR source; used when it is in ``ocr_sources`` and
    ``ocr_rate > 0``."""
    model_size: str = MODEL_SIZE
    """``"4b"`` or ``"8b"`` — selects the architecture, the base LM to initialise from, and
    the released checkpoint used by ``--init_from=molmo2``."""
    data_seed: int = 95818  # mm_olmo `data.seed`
    init_seed: int = 6198  # mm_olmo `seed`
    global_batch_size: int = GLOBAL_BATCH_SIZE
    """Global batch in *tokens* (= global instances × seq len). Override to scale the batch;
    pair with ``--train_module.rank_microbatch_size`` to set sequences/forward (GEMM size)."""
    recipe: str = RECIPE
    """The data recipe the rates below were defaulted from (:data:`RECIPES`); recorded for
    provenance. Pick it with ``--recipe``, which is read before the merge."""
    pointing_rate: float = POINTING_RATE
    """Fraction of mixture samples from pointing/counting sources (mm_olmo ``--pointing``)."""
    nlp_rate: float = NLP_RATE
    """Fraction of mixture samples from Tulu4 NLP SFT (mm_olmo ``--nlp``)."""
    pointing_data: str = POINTING_DATA
    """``"v1"`` (released Molmo2 pretrain sources) or ``"v2"`` (mm_olmo molmo3 stage-1 sources:
    audited points + sub-sampled absence queries); see :data:`POINTING_DATA`."""
    gui_pointing: bool = GUI_POINTING
    """Include MolmoPoint-GUISyn (``gui_syn``) in the v2 pointing group; see
    :data:`GUI_POINTING`. No effect with ``pointing_data == "v1"``."""
    ocr_rate: float = OCR_RATE
    """Fraction of mixture samples from the OCR group; see :data:`OCR_RATE`."""
    ocr_sources: Tuple[str, ...] = OCR_SOURCES
    """OCR sources in the group, each a separate dataset (names from
    :data:`olmo_core.data.multimodal.mixtures.ocr.OCR_SOURCE_NAMES`)."""
    ocr_data_root: str = OE_ENCODER_DATA
    """Root of the oe-encoder tar tree; each caption-tars source appends its own subdirectory
    (:data:`olmo_core.data.multimodal.mixtures.ocr.OCR_TAR_SOURCES`)."""
    academic_rate: float = ACADEMIC_RATE
    """Fraction of mixture samples from the academic QA group; see :data:`ACADEMIC_RATE`."""
    academic_sources: Tuple[str, ...] = ACADEMIC_SOURCES
    """Academic QA sources in the group (names from
    :data:`olmo_core.data.multimodal.mixtures.stage1_academic.STAGE1_ACADEMIC_SOURCE_NAMES`)."""
    clock_rate: float = CLOCK_RATE
    """Fraction of mixture samples from the PixMo-Clocks group; see :data:`CLOCK_RATE`."""
    train_vit: bool = TRAIN_VIT
    caption_message_weight: float = CAPTION_MESSAGE_WEIGHT
    """Multiplier on the caption source's loss tokens; 1.0 reproduces the released
    weighting. Read before ``merge`` runs, so it is declared here only to be accepted as a
    top-level override."""
    """Train the vision encoder in its own optimizer group (mm_olmo ``ft_vit``). When False
    the encoder is frozen and kept in eval mode."""

    freeze_base_embeddings: bool = FREEZE_BASE_EMBEDDINGS
    """Freeze the pretrained token-embedding block, training only the extra image-token rows
    (mm_olmo ``ft_embedding="lm_head"``)."""

    pack_sequences: bool = PACK_SEQUENCES
    """Pack several short examples per sequence (mm_olmo dynamic packer)."""

    init_from: str = INIT_FROM
    """``"molmo2"`` (released Molmo2-4B weights) or ``"scratch"`` (mm_olmo stage-1 init:
    base Qwen3-4B + base SigLIP2 + random connector/new embeddings)."""

    # Data-loader knobs. Exposed as fields (rather than left as module constants) so they can
    # be swept from the CLI: data loading was measured at ~11% of step time, and the packer
    # parameters are the other lever on examples/sec.
    data_prefetch_workers: int = DATA_PREFETCH_WORKERS
    """Background threads preprocessing examples (0 = synchronous)."""

    pack_max_crops: int = PACK_MAX_CROPS
    """Image-crop budget per pack — the knapsack's second dimension."""

    pack_buffer_size: int = PACK_BUFFER_SIZE
    """Examples buffered before the 2D knapsack picks a pack (mm_olmo ``buffer_size``)."""

    pack_image_weight: float = PACK_IMAGE_WEIGHT
    """Objective weight per image crop in the knapsack (mm_olmo ``image_weight``)."""


def _build_model_config(model_size: str, init_from: str) -> MultimodalLMConfig:
    """Build the Molmo2-4B :class:`MultimodalLMConfig` natively (no weights, no HF read).

    ``MultimodalLMConfig.molmo2_4B`` is byte-identical to
    ``molmo2_config_from_hf_config(AutoConfig.from_pretrained(MODEL_ID))``, so a
    from-scratch run needs nothing from the released Molmo2 repo.

    The one architecture field that depends on the weight source is the RoPE base: the
    released Molmo2-4B uses ``5e6``, but base Qwen3-4B — whose weights ``"scratch"`` loads
    — was trained with ``1e6`` (mm_olmo's stage-1 ``QWEN3_4B`` also uses ``1e6``). Using
    the released value with base weights would put them on the wrong rotary base.
    """
    spec = MODEL_SIZES[model_size]
    return spec.factory(
        rope_theta=spec.rope_theta(init_from),
        response_residual_dropout=RESPONSE_RESIDUAL_DROPOUT,
    )


def _read_override(overrides: List[str], key: str, default: str) -> str:
    """Read a top-level scalar out of the raw overrides.

    The model config depends on ``model_size`` and ``init_from`` (architecture and RoPE
    base), and it is built before :meth:`Config.merge` runs, so those overrides cannot be
    read off the merged config. Later occurrences win, matching ``merge``.
    """
    value = default
    # `Config.merge` normalizes hyphens to underscores (`_clean_opt`), so `--model-size=8b` and
    # `--model_size=8b` are the same override to it. Comparing the raw name here would miss the
    # dashed spelling and silently build the config from the default while `merge` applied the
    # requested value to the top-level field -- a divergence with no error.
    for override in overrides:
        name, _, raw = override.lstrip("-").partition("=")
        if name.replace("-", "_") == key and raw:
            value = raw
    return value


def _read_bool_override(overrides: List[str], key: str, default: bool) -> bool:
    """Read a boolean top-level scalar out of the raw overrides."""
    raw = _read_override(overrides, key, str(default)).strip().lower()
    if raw not in _BOOLS:
        raise OLMoConfigurationError(f"{key}={raw!r} is not a boolean")
    return _BOOLS[raw]


def _read_float_override(overrides: List[str], key: str, default: float) -> float:
    """Read a float top-level scalar out of the raw overrides."""
    raw = _read_override(overrides, key, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        raise OLMoConfigurationError(f"{key}={raw!r} is not a float") from None


def resolve_recipe(overrides: List[str]) -> Tuple[str, dict]:
    """The ``--recipe`` named in the raw overrides and the field defaults it sets.

    Read before :meth:`Config.merge`, so the recipe's rates become the config's starting values
    and an explicit ``--pointing_rate`` / ``--nlp_rate`` / ``--ocr_rate`` / ``--academic_rate`` /
    ``--clock_rate`` / ``--pointing_data`` still overrides them.

    :raises OLMoConfigurationError: If the recipe is not one of :data:`RECIPES`.
    """
    recipe = _read_override(overrides, "recipe", RECIPE).strip().lower()
    if recipe not in RECIPES:
        raise OLMoConfigurationError(f"recipe={recipe!r} is not one of {tuple(RECIPES)}")
    return recipe, dict(RECIPES[recipe])


def _resolve_model_spec(overrides: List[str]) -> Tuple[str, str]:
    """Resolve ``(model_size, init_from)`` from the raw overrides, validating both."""
    model_size = _read_override(overrides, "model_size", MODEL_SIZE).lower()
    init_from = _read_override(overrides, "init_from", INIT_FROM)
    if model_size not in MODEL_SIZES:
        raise OLMoConfigurationError(
            f"model_size={model_size!r} is not one of {tuple(MODEL_SIZES)}"
        )
    if init_from not in INIT_FROM_CHOICES:
        raise OLMoConfigurationError(f"init_from={init_from!r} is not one of {INIT_FROM_CHOICES}")
    return model_size, init_from


def validate_data_config(config) -> None:
    """Check the data-mixture fields of a merged config.

    Separate from :func:`build_config`, which also resolves the Beaker launch and so cannot run
    without cluster access: this only reads ``recipe``, ``pointing_data``, the five rates,
    ``ocr_sources``, ``olmocr``, ``ocr_tars``, ``text_rich`` and ``academic_sources``.

    :raises OLMoConfigurationError: If a field is invalid, or an override would be silently
        ignored, or a source would read anything but its train split, or an academic source is the
        training set of a stage-2 eval benchmark.
    """
    if config.recipe not in RECIPES:
        raise OLMoConfigurationError(f"recipe={config.recipe!r} is not one of {tuple(RECIPES)}")
    if config.pointing_data not in POINTING_DATA_CHOICES:
        raise OLMoConfigurationError(
            f"pointing_data={config.pointing_data!r} is not one of {POINTING_DATA_CHOICES}"
        )
    if config.ocr_rate > 0 and not config.ocr_sources:
        raise OLMoConfigurationError("ocr_rate > 0 needs at least one entry in ocr_sources")
    # This script only trains, so a source may only ever read its train split. olmOCR-mix ships
    # a held-out `eval` parquet per subset that the dataset class can read for evaluation; it
    # must not be reachable from a training run by an override.
    if canonical_split(config.olmocr.split) != "train":
        raise OLMoConfigurationError(
            f"olmocr.split={config.olmocr.split!r}: a training run reads only the train split "
            "(the olmOCR-mix `eval` parquets are held-out data)"
        )
    unknown = [n for n in config.ocr_sources if n not in OCR_SOURCE_NAMES]
    if unknown:
        raise OLMoConfigurationError(
            f"Unknown ocr_sources {unknown}; expected names from {OCR_SOURCE_NAMES}"
        )
    if len(set(config.ocr_sources)) != len(config.ocr_sources):
        raise OLMoConfigurationError(f"ocr_sources has duplicates: {config.ocr_sources}")
    # `build_ocr_source` sets these per source, so a value here would be accepted, saved into the
    # run config and then ignored. Fail instead of pretending it took effect.
    for template, default, fields in (
        (
            "ocr_tars",
            OcrCaptionTarsDatasetConfig(),
            (
                ("dataset_path", "use --ocr_data_root to relocate the whole tar tree"),
                ("style", "the style is fixed per source by mixtures.ocr.OCR_TAR_SOURCES"),
                ("strip_text_tags", "set per source by mixtures.ocr.OCR_TAR_SOURCES"),
            ),
        ),
        ("olmocr", OlmOcrMixDatasetConfig(), (("subset", "pick subsets with --ocr_sources"),)),
        (
            "text_rich",
            TextRichCaptionDatasetConfig(),
            (("category", "pick categories with --ocr_sources"),),
        ),
    ):
        for field, hint in fields:
            if getattr(getattr(config, template), field) != getattr(default, field):
                raise OLMoConfigurationError(
                    f"--{template}.{field} is set per OCR source and would be ignored here; {hint}"
                )
    unverified = [n for n in config.ocr_sources if n in SPLIT_UNVERIFIED_SOURCES]
    if unverified and config.ocr_rate > 0:
        log.warning(
            "ocr_sources includes %s, whose tars cannot be shown to hold only a training split "
            "(see mixtures.ocr); they are not in the default group for that reason.",
            unverified,
        )
    for tar_name, mix_name in DUPLICATE_OLMOCR_SOURCES.items():
        if tar_name in config.ocr_sources and mix_name in config.ocr_sources:
            log.warning(
                "ocr_sources has both %s and %s, which are the same pages rendered by two "
                "pipelines: those pages will be sampled twice.",
                tar_name,
                mix_name,
            )
    if config.academic_rate > 0 and not config.academic_sources:
        raise OLMoConfigurationError(
            "academic_rate > 0 needs at least one entry in academic_sources"
        )
    # Checked whatever the rate, so a refused set cannot sit in a saved config waiting for a rate.
    held_out = [n for n in config.academic_sources if n in STAGE2_EVAL_TRAIN_SETS]
    if held_out:
        raise OLMoConfigurationError(
            f"academic_sources {held_out} are training sets of stage-2 eval benchmarks "
            f"({sorted({STAGE2_EVAL_TRAIN_SETS[n] for n in held_out})}); they stay stage-2 data"
        )
    if CLOCKS_SOURCE in config.academic_sources:
        raise OLMoConfigurationError(
            f"{CLOCKS_SOURCE} is not an academic source: it is its own group, set with --clock_rate"
        )
    unknown = [n for n in config.academic_sources if n not in STAGE1_ACADEMIC_SOURCE_NAMES]
    if unknown:
        raise OLMoConfigurationError(
            f"Unknown academic_sources {unknown}; expected names from "
            f"{STAGE1_ACADEMIC_SOURCE_NAMES}"
        )
    if len(set(config.academic_sources)) != len(config.academic_sources):
        raise OLMoConfigurationError(f"academic_sources has duplicates: {config.academic_sources}")
    rates = (
        config.pointing_rate,
        config.nlp_rate,
        config.ocr_rate,
        config.academic_rate,
        config.clock_rate,
    )
    if any(r < 0 for r in rates):
        raise OLMoConfigurationError(f"group rates must be >= 0, got {rates}")
    if sum(rates) > 1.0:
        raise OLMoConfigurationError(
            "pointing_rate + nlp_rate + ocr_rate + academic_rate + clock_rate exceeds 1: nothing "
            "is left for the caption source"
        )


def build_config(script: str, run_name: str, overrides: List[str]) -> ExperimentConfig:
    root_dir = get_root_dir(BEAKER_CLUSTER)
    beaker_user = get_beaker_username()
    assert beaker_user is not None

    model_size, init_from = _resolve_model_spec(overrides)
    recipe, recipe_fields = resolve_recipe(overrides)
    # Resolved pre-merge because it shapes `freeze_params`, the optimizer groups and the
    # per-group scheduler, all of which are built before `Config.merge` runs.
    train_vit = _read_bool_override(overrides, "train_vit", TRAIN_VIT)
    caption_message_weight = _read_float_override(
        overrides, "caption_message_weight", CAPTION_MESSAGE_WEIGHT
    )
    freeze_base_embeddings = _read_bool_override(
        overrides, "freeze_base_embeddings", FREEZE_BASE_EMBEDDINGS
    )
    model_config = _build_model_config(model_size, init_from)

    dataset_config = PixMoCapDatasetConfig(
        dataset_path=DATASET_PATH,
        mode="transcript_and_caption",
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        # mm_olmo's captioner leaves `loss_token_weighting` at its "none" default, so every
        # response token is weighted equally. `root_subsegments` (1/sqrt(n_branches)) does not
        # cancel out of the global `sum(CE*w)/sum(w)` divisor when branch counts differ across
        # examples, so it would re-weight caption vs pointing vs NLP relative to mm_olmo.
        loss_token_weighting="none",
        # Captions carry 1.25x weight. The released Molmo2-4B-Pretrain leaves every source at
        # 1.0, which puts captions at ~77.5% of the sum(CE*w)/sum(w) loss mass; 1.25 lifts that
        # to ~81%. Measured on two-seed baselines (n=2 per arm, seed spread in parentheses):
        #   dense_caption avg   57.271 -> 57.808  (+0.537, spread 0.085)
        #   consistency         69.973 -> 70.701  (+0.728, spread 0.024) -- released is 70.745
        #   pixmo_points f1      0.7938 ->  0.7847 (-0.009, spread 0.008)
        #   sa_co f1             0.5537 ->  0.5408 (-0.013, spread 0.011)
        # i.e. a caption gain ~6x the noise floor for pointing costs at ~1x it.
        message_weight=caption_message_weight,
        seed=95818,
    )

    # The v2 pointing sources (mm_olmo `_base_mixture`); only built when `pointing_data == "v2"`.
    # Unlike the v1 classes, which take their prompt family from POINTING_DATASET_KWARGS, these
    # have the stage-1 family built in: only the loss weighting is passed through.
    pointing_v2_config = PixMoPointsV2DatasetConfig(
        p_paired_negatives=POINTING_V2_P_PAIRED_NEGATIVES,
        n_easy_samples=POINTING_V2_N_EASY_NEGATIVES,
        audit_style=POINTING_V2_AUDIT_STYLE,
        filter_audit=POINTING_V2_FILTER_AUDIT,
        max_crops=MAX_CROPS,
        loss_token_weighting=POINTING_DATASET_KWARGS["loss_token_weighting"],
    )
    count_v2_config = PixMoCountV2DatasetConfig(
        audit_style=POINTING_V2_AUDIT_STYLE,
        filter_audit=POINTING_V2_FILTER_AUDIT,
        max_crops=MAX_CROPS,
        loss_token_weighting=POINTING_DATASET_KWARGS["loss_token_weighting"],
    )
    gui_syn_config = GuiSynDatasetConfig(
        max_crops=MAX_CROPS,
        loss_token_weighting=POINTING_DATASET_KWARGS["loss_token_weighting"],
    )
    # OCR source templates (`build_ocr_source` fills in the per-source fields); only built when
    # `ocr_rate > 0`. Every response token weighted equally, like the caption source; the user
    # turn is the bare `<style>:` tag (`olmocr:` / `textocr:` / `synth_ocr:` / `receipt_ocr:` /
    # `fig_caption_{high,mid,low}:`), as in mm_olmo's molmo3 stage 1. Long pages are
    # tail-truncated to the sequence length.
    olmocr_config = OlmOcrMixDatasetConfig(
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="none",
    )
    ocr_tars_config = OcrCaptionTarsDatasetConfig(
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="none",
    )
    # mm_olmo's `figure_ocr` group: three captions per image, as three branches.
    text_rich_config = TextRichCaptionDatasetConfig(
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="none",
    )
    # The synthetic English transcription sets (see `synthetic_ocr`).
    nvidia_synth_config = NvidiaSynthOcrDatasetConfig(
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="none",
    )
    receipts_config = SyntheticReceiptsDatasetConfig(
        max_crops=MAX_CROPS,
        max_sequence_length=SEQUENCE_LENGTH,
        loss_token_weighting="none",
    )

    # Pad token: Molmo2/Qwen2.5 EOS (151643). Fixed-length padding so every batch has a
    # constant token count for the token-based Trainer.
    collator_config = MultimodalCollatorConfig(
        pad_token_id=151643,
        label_ignore_index=-100,
        pad_sequence_length=SEQUENCE_LENGTH,
    )

    train_module_config = MultimodalTransformerTrainModuleConfig(
        rank_microbatch_size=RANK_MICROBATCH_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=LLM_LR,
            betas=(0.9, 0.95),
            eps=1e-6,
            weight_decay=0.0,
            group_overrides=[
                OptimGroupOverride(
                    params=["connector.*"],
                    opts=dict(lr=CONNECTOR_LR, weight_decay=0.0, scheduler_name="connector"),
                ),
                # mm_olmo's third component group (`ft_vit=True`, vit_learning_rate=6e-6).
                *(
                    [
                        OptimGroupOverride(
                            params=["vision.*"],
                            opts=dict(lr=VIT_LR, weight_decay=0.0, scheduler_name="vision"),
                        )
                    ]
                    if train_vit
                    else []
                ),
            ],
        ),
        # An empty list keeps the encoder trainable *and* in train mode — the train module
        # only forces `vision.eval()` when `vision.*` is frozen.
        # mm_olmo: `activation_checkpointing: true` with `llm.activation_checkpoint:
        # whole_layer` — every block checkpointed. This is what makes microbatch 4 fit.
        # (vision/connector AC are already on by default in the train module, matching
        # the released config's vit `activation_checkpointing: true` and
        # `connector_activation_checkpointing: true`.)
        ac_config=TransformerActivationCheckpointingConfig(
            mode=TransformerActivationCheckpointingMode.full
        ),
        freeze_params=(
            ([] if train_vit else ["vision.*"])
            # The extra image-token rows live in `embeddings.extra_weight`, so freezing
            # `embeddings.weight` leaves them trainable.
            + (["lm.embeddings.weight"] if freeze_base_embeddings else [])
        ),
        z_loss_multiplier=1e-4,
        max_grad_norm=1.0,
        compile_model=COMPILE_MODEL,
        autocast_precision=DType.bfloat16,
        scheduler=PerGroupScheduler(
            schedulers={
                "connector": CosWithWarmup(warmup=CONNECTOR_WARMUP, alpha_f=ALPHA_F),
                **(
                    {"vision": CosWithWarmup(warmup=VIT_WARMUP, alpha_f=ALPHA_F)}
                    if train_vit
                    else {}
                ),
            },
            default=CosWithWarmup(warmup=LLM_WARMUP, alpha_f=ALPHA_F),
        ),
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            # mm_olmo's released pretrain used `fsdp.precision: float` — fp32 params,
            # gradients and buffers, with bf16 only from the `amp_bf16` autocast below.
            param_dtype=DType.float32,
            reduce_dtype=DType.float32,
        ),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{root_dir}/checkpoints/{beaker_user.lower()}/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=5,
            cancel_check_interval=5,
            max_duration=Duration.steps(RECIPE_MAX_STEPS.get(recipe, MAX_STEPS)),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            # Synchronous checkpointing: avoids the async checkpoint thread pool whose
            # teardown raced/failed on this cluster ("cannot schedule new futures after
            # interpreter shutdown"). Saves block briefly but complete reliably.
            # The last 5 permanent checkpoints are kept (steps 24k-32k of a 32k run), each with
            # its DCP state and the unsharded export, about 68 GB apiece. The v2 run scored
            # within about a point across 24k-32k, so they span the checkpoints worth evaluating.
            CheckpointerCallback(
                save_interval=2000,
                ephemeral_save_interval=500,
                save_async=False,
                max_checkpoints=5,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=run_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                enabled=WANDB_PROJECT is not None,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        # Every permanent checkpoint also gets `model.safetensors` + `olmo_core_config.json`, so
        # evals read one file instead of the 256-shard DCP state (olmo-eval issue #379).
        .with_callback("unsharded_export", UnshardedModelExportCallback())
        .with_callback("garbage_collector", GarbageCollectorCallback())
        .with_callback("beaker", BeakerCallback())
    )  # NOTE: no in-loop eval callbacks (out of scope for stage 1).

    launch_config = build_launch_config(
        name=run_name,
        root_dir=root_dir,
        cmd=[script, "train", run_name, *overrides],
        cluster=BEAKER_CLUSTER,
        workspace=BEAKER_WORKSPACE,
        budget=BEAKER_BUDGET,
        num_nodes=NUM_NODES,
    )
    # Stage-1 reads data and writes checkpoints on weka, so no S3 / GCS secrets are required.
    launch_config.aws_config_secret = None
    launch_config.aws_credentials_secret = None
    launch_config.google_credentials_secret = None
    # Only request env secrets that exist in the (debug) workspace; drop optional ones
    # (COMET / R2 / WEKA / SLACK) that aren't provisioned there.
    launch_config.env_secrets = [
        s for s in launch_config.env_secrets if s.name in ("BEAKER_TOKEN", "WANDB_API_KEY")
    ]
    # The pointing/counting Arrow datasets on weka were saved with `datasets >= 4`, whose
    # `List` feature type the image's older `datasets` can't deserialize. Upgrade after the
    # package install (olmo-core does not pin `datasets`, so this is not clobbered).
    # `pypdfium2` renders the olmOCR-mix PDF pages and `h5py` reads the NVIDIA synthetic OCR
    # files at load time (`--ocr_rate > 0`); small self-contained wheels, so they are installed
    # unconditionally.
    launch_config.post_setup = "pip install -U 'datasets>=4,<6' pypdfium2 h5py"
    # Optionally use the fused FlexAttention backend for the multimodal masks (~+8% MFU on
    # the stage-1 mixture vs the dense `torch` backend; see USE_FLEX_ATTN).
    if USE_FLEX_ATTN:
        launch_config.env_vars = list(launch_config.env_vars) + [
            BeakerEnvVar(name="OLMO2_FLEX_ATTN", value="1")
        ]

    config = ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        collator=collator_config,
        train_module=train_module_config,
        trainer=trainer_config,
        launch=launch_config,
        pointing_v2=pointing_v2_config,
        count_v2=count_v2_config,
        gui_syn=gui_syn_config,
        recipe=recipe,
        **recipe_fields,
        olmocr=olmocr_config,
        ocr_tars=ocr_tars_config,
        text_rich=text_rich_config,
        nvidia_synth=nvidia_synth_config,
        receipts=receipts_config,
    ).merge(overrides)

    validate_data_config(config)

    # `_resolve_model_spec` already validated the pre-merge values; re-check the merged
    # config so a stray `--model_size`/`--init_from`/`--train_vit` cannot slip through, and
    # fail here rather than after a Beaker job has queued, started and downloaded weights.
    resolved = (model_size, init_from, train_vit, freeze_base_embeddings)
    merged = (
        config.model_size,
        config.init_from,
        config.train_vit,
        config.freeze_base_embeddings,
    )
    if merged != resolved:
        raise OLMoConfigurationError(
            f"model_size / init_from / train_vit / freeze_base_embeddings changed during merge: "
            f"{resolved} -> {merged}"
        )

    return config


def _load_tokenizer(model_size: str, init_from: str):
    """Load the tokenizer for this init mode.

    A ``"scratch"`` run uses base ``Qwen/Qwen3-4B``'s tokenizer — what mm_olmo's stage 1
    uses — so nothing is read from the released Molmo2 repo. It is a drop-in for training:
    identical BPE (``encode`` matches exactly), identical ``eos_token_id`` (151645), and
    identical output for the only chat-template calls the data pipeline makes (user-turn +
    generation prompt, see ``data/multimodal/qwen3_layout.py``). Qwen3's template differs
    only for *assistant* messages, which that pipeline never passes. The image-special token
    IDs come from ``nn/vision/molmo2_tokens.py`` constants, not from the tokenizer, so their
    absence from the base vocab does not affect training (they are unmapped when *decoding*,
    which only matters for debugging).
    """
    from transformers import AutoTokenizer

    spec = MODEL_SIZES[model_size]
    if init_from == "scratch":
        return AutoTokenizer.from_pretrained(spec.scratch_lm_id)
    return AutoTokenizer.from_pretrained(spec.molmo2_id, trust_remote_code=True)


def _init_weights_from_hf(
    model: MultimodalLM, model_cfg: MultimodalLMConfig, molmo2_id: str
) -> None:
    """Load converted HF Molmo2 weights into the (meta-initialised) model.

    :param molmo2_id: The released checkpoint to continue from, e.g. ``allenai/Molmo2-4B``.
    """
    from transformers import AutoModelForImageTextToText

    from olmo_core.nn.vision.molmo2_loader import (
        ensure_default_rope_registered,
        molmo2_hf_state_dict_to_multimodal_lm,
        reinit_rope_buffers,
        retie_word_embeddings,
    )

    ensure_default_rope_registered()
    log.info(f"Loading HF weights from {molmo2_id} ...")
    hf = AutoModelForImageTextToText.from_pretrained(molmo2_id, trust_remote_code=True)
    reinit_rope_buffers(hf)
    converted = molmo2_hf_state_dict_to_multimodal_lm(hf.state_dict(), model_cfg)
    del hf
    model.to_empty(device=get_default_device())
    model.load_state_dict(converted, strict=False)
    # `to_empty` silently un-ties tied word embeddings (Molmo2-4B is tied; the 8B is not);
    # restore the share so training updates the head and the embedding table as one
    # parameter, like mm_olmo. A no-op for untied configs.
    retie_word_embeddings(model)
    del converted


def _init_weights_from_scratch(
    model: MultimodalLM, model_cfg: MultimodalLMConfig, scratch_lm_id: str
) -> None:
    """mm_olmo's true stage-1 init (``reset_with_pretrained_weights``):

    * LM  <- the base Qwen3 backbone, via the generic HF->olmo-core converter;
      the 128 extra-token embedding rows are ``N(0, 0.02)`` (mm_olmo
      ``new_embedding_init_range``) and the LM head is tied to the embeddings when the
      config is tied (Molmo2-4B; the 8B and Qwen3-8B are untied).
    * vision <- base ``google/siglip2-so400m-patch14-384`` (blocks 0..24), shared by every
      released Molmo2 variant.
    * connector <- random (``reset_parameters``: ``N(0, 0.02)`` weights, zero biases —
      matches mm_olmo's pooling + projector init).

    :param scratch_lm_id: The base LM to initialise from, e.g. ``Qwen/Qwen3-4B``.
    """
    import torch
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    from olmo_core.nn.hf import convert_state_from_hf
    from olmo_core.nn.vision import siglip_state_dict_to_vision_encoder
    from olmo_core.nn.vision.molmo2_loader import retie_word_embeddings

    log.info(f"[scratch init] Loading base LM weights from {scratch_lm_id} ...")
    hf_lm = AutoModelForCausalLM.from_pretrained(scratch_lm_id, dtype=torch.float32)
    lm_state = convert_state_from_hf(hf_lm.config, hf_lm.state_dict(), model_type="qwen3")
    del hf_lm

    # The base checkpoint's vocab must match our base block exactly; the extra image-token
    # rows are a separate parameter, initialised here from mm_olmo's `new_embedding_init_range`.
    emb = lm_state["embeddings.weight"]
    if emb.shape[0] != model_cfg.lm.vocab_size:
        raise RuntimeError(
            f"Base LM vocab {emb.shape[0]} != target base vocab {model_cfg.lm.vocab_size}"
        )
    n_extra = model_cfg.lm.n_extra_vocab
    gen = torch.Generator().manual_seed(0)  # same rows on every rank
    extra_rows = torch.empty(n_extra, emb.shape[1], dtype=emb.dtype)
    extra_rows.normal_(std=NEW_EMBEDDING_INIT_STD, generator=gen)

    converted = {f"lm.{k}": v for k, v in lm_state.items()}
    converted["lm.embeddings.extra_weight"] = extra_rows
    # Both our LM head and the base checkpoint's span the base vocab, so it maps straight
    # across: no padding, and for a tied config the head simply *is* the base block
    # (`retie_word_embeddings` below restores the share that `to_empty` broke).
    if model_cfg.lm.tie_word_embeddings:
        converted["lm.lm_head.w_out.weight"] = emb
    del lm_state

    log.info(f"[scratch init] Loading base ViT weights from {SCRATCH_VIT_ID} ...")
    hf_vit = SiglipVisionModel.from_pretrained(SCRATCH_VIT_ID, dtype=torch.float32)
    converted.update(
        siglip_state_dict_to_vision_encoder(
            hf_vit.state_dict(), n_blocks=len(model.vision.blocks), prefix="vision."
        )
    )
    del hf_vit

    model.to_empty(device=get_default_device())
    missing, unexpected = model.load_state_dict(converted, strict=False)
    del converted
    # Everything except the connector must have been covered by the two base checkpoints.
    non_connector_missing = [k for k in missing if not k.startswith("connector.")]
    if non_connector_missing or unexpected:
        raise RuntimeError(
            f"[scratch init] unexpected state-dict coverage: missing={non_connector_missing[:8]} "
            f"unexpected={list(unexpected)[:8]}"
        )
    # `to_empty` above un-ties tied word embeddings, so the head and the embedding table
    # would otherwise train as two independent parameters. Restore the share so they move
    # together, matching mm_olmo's `weight_tying=True` (no-op for untied configs).
    retie_word_embeddings(model)
    log.info("[scratch init] Randomly initialising the connector ...")
    model.connector.reset_parameters()


def _size_fractions(sizes: Sequence[int], rule: str, names: Optional[Sequence[str]] = None):
    """Split one group's rate among its sources from their sizes (mm_olmo SubMixture math):
    ``"sqrt"`` is ``root_size_factor=None``, ``"linear"`` is ``size_weighted=1``.

    An empty source is rejected rather than weighted. Size-proportional weights give it 0, and
    :class:`~olmo_core.data.multimodal.MixtureDataLoader` then skips it silently, so a mistyped
    path or an over-strict filter would train the remaining sources at quietly renormalized
    rates with nothing in the log; all-empty additionally divides by zero into NaN weights.

    :raises OLMoConfigurationError: If any size is <= 0, or ``rule`` is unknown.
    """
    import numpy as np

    empty = [
        (names[i] if names is not None else str(i)) for i, s in enumerate(sizes) if int(s) <= 0
    ]
    if empty:
        raise OLMoConfigurationError(
            f"mixture source(s) {empty} have no examples: check their data paths and filters "
            "(a size-0 source is never sampled, so it would silently drop out of the mixture)"
        )
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    if rule == "sqrt":
        frac = np.sqrt(sizes_arr)
    elif rule == "linear":
        frac = sizes_arr
    else:
        raise OLMoConfigurationError(f"unknown size rule {rule!r}")
    return frac / frac.sum()


def _ocr_fractions(names: Sequence[str], sizes: Sequence[int]):
    """How the OCR group's rate is split among its sources.

    Two levels, like mm_olmo's two OCR groups. The rate is first divided between the tasks,
    transcription and figure captions (``mixtures.ocr.OCR_TASK_SHARES``); each task's share is
    then divided among its sources by sqrt(size). A single flat split would be decided by the
    figure captions' much larger row counts. A synthetic source's size is capped at the largest
    real source of its task first (``mixtures.ocr.ocr_weighting_sizes``).
    """
    import numpy as np

    shares = ocr_task_shares(names)
    tasks = [ocr_task(n) for n in names]
    sizes = ocr_weighting_sizes(names, sizes)
    out = np.zeros(len(names), dtype=np.float64)
    for task in dict.fromkeys(tasks):
        idx = [i for i, t in enumerate(tasks) if t == task]
        within = _size_fractions([sizes[i] for i in idx], "sqrt", [names[i] for i in idx])
        for i, f in zip(idx, within):
            out[i] = shares[i] * float(f)
    return out


def _pointing_group_fractions(
    sizes: Sequence[int], pointing_data: str, names: Optional[Sequence[str]] = None
):
    """How the pointing group's rate is split among its sources, from their sizes.

    ``"v1"`` follows mm_olmo's captioner (``root_size_factor=None``: sqrt of the size);
    ``"v2"`` follows mm_olmo's molmo3 stage 1 (``size_weighted=1``: linear in the size).

    Under ``"v2"`` the split is linear, so each source's *row count* sets its share directly.
    Ours counts rows with a trainable POINTING annotation; mm_olmo's ``_base_mixture`` leaves
    ``include_masks=True``, so its filter also admits mask-only annotations and its count is
    larger. We do not port masks, so the pointing-only count is the right denominator here --
    but it does mean this group's internal split is not numerically mm_olmo's.
    """
    if pointing_data == "v1":
        return _size_fractions(sizes, "sqrt", names)
    if pointing_data == "v2":
        return _size_fractions(sizes, "linear", names)
    raise OLMoConfigurationError(
        f"pointing_data={pointing_data!r} is not one of {POINTING_DATA_CHOICES}"
    )


def _academic_fractions(names: Sequence[str], sizes: Sequence[int]):
    """How the academic group's rate is split among its sources: by sqrt(size), with the templated
    chart sets' sizes capped (``mixtures.stage1_academic.academic_weighting_sizes``).
    """
    return _size_fractions(academic_weighting_sizes(names, sizes), "sqrt", names)


def _v2_pointing_sources(config: ExperimentConfig) -> List[Tuple[str, Config]]:
    """The v2 pointing group's ``(name, dataset config)`` pairs: mm_olmo train_molmo3_stage1
    `_base_mixture`'s pointing group, minus `CocoTrain` and every segmentation branch (the v2
    sources are ported points-only), plus GUISyn when ``gui_pointing``."""
    sources: List[Tuple[str, Config]] = [
        ("pixmo_points_v2", config.pointing_v2),
        ("pixmo_count_v2", config.count_v2),
        (
            "cosyn_point_v2",
            CoSynPointDatasetConfig(
                dataset_path=COSYN_POINT_V2_PATH,
                audit_style=POINTING_V2_COSYN_AUDIT_STYLE,
                max_crops=MAX_CROPS,
                **POINTING_DATASET_KWARGS,
            ),
        ),
    ]
    if config.gui_pointing:
        sources.append(("gui_syn", config.gui_syn))
    return sources


def _build_mixture_sources(tokenizer, config: ExperimentConfig):
    """Build the caption + pointing + NLP + OCR + academic + clock sources, their sampling weights
    (mm_olmo SubMixture) and their names: caption gets ``1 - pointing_rate - nlp_rate - ocr_rate -
    academic_rate - clock_rate``; the pointing group shares ``pointing_rate`` (split per
    :func:`_pointing_group_fractions`); NLP gets ``nlp_rate``; the OCR sources share ``ocr_rate``
    (:func:`_ocr_fractions`); the academic sources share ``academic_rate``
    (:func:`_academic_fractions`); PixMo-Clocks gets ``clock_rate``."""
    p, n, o, a = config.pointing_rate, config.nlp_rate, config.ocr_rate, config.academic_rate
    c = config.clock_rate
    datasets: List = [config.dataset.build(tokenizer)]  # caption
    weights: List[float] = [max(1.0 - p - n - o - a - c, 0.0)]
    names: List[str] = ["pixmo_cap"]

    if p > 0:
        if config.pointing_data == "v1":
            pointing_names = [
                "pixmo_points_train",
                "pixmo_count_train",
                "pixmo_points_high_freq_train",
                "cosyn_point",
            ]
            pointing = [
                PixMoPointsDatasetConfig(
                    kind="basic", max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS
                ).build(tokenizer),
                PixMoCountDatasetConfig(max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS).build(
                    tokenizer
                ),
                PixMoPointsDatasetConfig(
                    kind="high_frequency", max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS
                ).build(tokenizer),
                CoSynPointDatasetConfig(max_crops=MAX_CROPS, **POINTING_DATASET_KWARGS).build(
                    tokenizer
                ),
            ]
        elif config.pointing_data == "v2":
            sources = _v2_pointing_sources(config)
            pointing_names = [name for name, _ in sources]
            pointing = [source.build(tokenizer) for _, source in sources]
        else:
            raise OLMoConfigurationError(
                f"pointing_data={config.pointing_data!r} is not one of {POINTING_DATA_CHOICES}"
            )
        frac = _pointing_group_fractions(
            [len(d) for d in pointing], config.pointing_data, pointing_names
        )
        datasets += pointing
        weights += [p * float(f) for f in frac]
        names += pointing_names

    if n > 0:
        datasets.append(Tulu4DatasetConfig().build(tokenizer))
        weights.append(n)
        names.append("tulu4")

    if o > 0:
        ocr = [
            build_ocr_source(
                name,
                tokenizer,
                olmocr=config.olmocr,
                tars=config.ocr_tars,
                text_rich=config.text_rich,
                nvidia_synth=config.nvidia_synth,
                receipts=config.receipts,
                data_root=config.ocr_data_root,
            )
            for name in config.ocr_sources
        ]
        frac = _ocr_fractions(list(config.ocr_sources), [len(d) for d in ocr])
        datasets += ocr
        weights += [o * float(f) for f in frac]
        names += list(config.ocr_sources)

    if a > 0:
        academic = [
            build_stage1_academic_source(
                name, tokenizer, max_crops=MAX_CROPS, seed=config.data_seed
            )
            for name in config.academic_sources
        ]
        frac = _academic_fractions(list(config.academic_sources), [len(d) for d in academic])
        datasets += academic
        weights += [a * float(f) for f in frac]
        names += list(config.academic_sources)

    if c > 0:
        datasets.append(
            build_stage1_clocks_source(tokenizer, max_crops=MAX_CROPS, seed=config.data_seed)
        )
        weights.append(c)
        names.append(CLOCKS_SOURCE)

    log.info(
        "Mixture sources / sizes / weights: %s",
        [(name, len(d), round(w, 4)) for name, d, w in zip(names, datasets, weights)],
    )
    return datasets, weights, names


def train(config: ExperimentConfig):
    seed_all(config.init_seed)

    tokenizer = _load_tokenizer(config.model_size, config.init_from)

    model = config.model.build(init_device="meta")
    spec = MODEL_SIZES[config.model_size]
    if config.init_from == "scratch":
        _init_weights_from_scratch(model, config.model, spec.scratch_lm_id)
    elif config.init_from == "molmo2":
        _init_weights_from_hf(model, config.model, spec.molmo2_id)
    else:  # unreachable via build_config, which validates; guards direct train() calls
        raise OLMoConfigurationError(
            f"init_from={config.init_from!r} is not one of {INIT_FROM_CHOICES}"
        )

    train_module = config.train_module.build(model)

    collator = config.collator.build()
    # Derive the data-parallel world size / rank from the train module's DP process
    # group so each rank reads its own shard (must match the trainer's DP degree).
    dp_pg = train_module.dp_process_group
    dp_world_size, dp_rank = get_world_size(dp_pg), get_rank(dp_pg)

    if (
        config.pointing_rate > 0
        or config.nlp_rate > 0
        or config.ocr_rate > 0
        or config.academic_rate > 0
        or config.clock_rate > 0
    ):
        datasets, weights, names = _build_mixture_sources(tokenizer, config)
        data_loader = MixtureDataLoader(
            datasets,
            weights,
            collator,
            dataset_names=names,
            work_dir=config.trainer.save_folder,
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            pack=config.pack_sequences,
            pack_max_crops=config.pack_max_crops if config.pack_sequences else None,
            pack_buffer_size=config.pack_buffer_size,
            pack_image_weight=config.pack_image_weight,
            prefetch_workers=config.data_prefetch_workers,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )
    else:
        data_loader = MultimodalDataLoader(
            config.dataset.build(tokenizer),
            collator,
            work_dir=config.trainer.save_folder,
            global_batch_size=config.global_batch_size,
            seed=config.data_seed,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
        )

    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def launch(config: ExperimentConfig):
    config.launch.launch(follow=True)


if __name__ == "__main__":
    usage = f"""
Usage
=====

› python {sys.argv[0]} [dry_run|launch|train] RUN_NAME [OVERRIDES...]

  * dry_run: Print out the final config after applying overrides and exit.
  * launch:  Launch the script on Beaker as a batch job for training.
  * train:   Run training locally (usually under torchrun).

Examples
========

Print the config:
› python {sys.argv[0]} dry_run molmo2-stage1

The v2 recipe (caption 0.5, v2 pointing and counting 0.25, OCR 0.25, no text-only data):
› python {sys.argv[0]} launch molmo2-stage1-v2 --recipe=v2

8B from-scratch run:
› python {sys.argv[0]} launch molmo2-stage1-8b --model_size=8b

Audited (v2) pointing sources, dropping audit-failed points instead of marking them:
› python {sys.argv[0]} launch molmo2-stage1-v2pts --pointing_data=v2 --pointing_v2.filter_audit=true

OCR group at 15% (olmOCR-mix + TextOCR transcription, and three-level figure captions):
› python {sys.argv[0]} launch molmo2-stage1-ocr --ocr_rate=0.15
Only the olmOCR-mix page transcription sources:
› python {sys.argv[0]} launch molmo2-stage1-olmocr --ocr_rate=0.075 \
      --ocr_sources=[olmocr_documents,olmocr_books,olmocr_loc_transcripts,olmocr_national_archives]

The v3 recipe (50k steps; caption 0.33, pointing 0.16, OCR 0.34, academic QA 0.14, clocks 0.03):
› python {sys.argv[0]} launch molmo2-stage1-v3 --recipe=v3

Local synthetic smoke test:
› torchrun --nproc-per-node=1 {sys.argv[0]} train smoke \\
      --dataset.dataset_path=synthetic --trainer.max_duration.value=5

Launch on Beaker:
› python {sys.argv[0]} launch molmo2-stage1 --launch.num_nodes=1
    """.strip()

    if len(sys.argv) < 3:
        print(usage)
        sys.exit(1)

    script, cmd, run_name, *overrides = sys.argv

    if cmd == "train":
        # Use a generous process-group timeout (gloo + NCCL). The default 15 min was the
        # exact watchdog timeout that aborted runs when a rank lagged on a collective
        # during checkpointing / bookkeeping (and W&B network stalls can add latency).
        prepare_training_environment(timeout=timedelta(minutes=60))
    else:
        prepare_cli_environment()

    config = build_config(script, run_name, overrides)
    log.info(config)

    if cmd == "train":
        train(config)
        teardown_training_environment()
    elif cmd == "launch":
        launch(config)
    elif cmd == "dry_run":
        pass
    else:
        print(usage)
        sys.exit(1)
