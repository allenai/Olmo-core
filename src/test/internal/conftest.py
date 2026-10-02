import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from olmo_core.data import NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.multimodal.alignment import MultimodalMixtureConfig
from olmo_core.data.multimodal.pixmo_cap import PixMoCapDatasetConfig
from olmo_core.internal import vision_alignment
from olmo_core.internal.experiment import CliContext, SubCmd
from olmo_core.internal.vision_alignment_data import (
    ALIGNMENT_LOSS_TARGETS,
    ALIGNMENT_MEAN_LOSS_WEIGHTS,
)
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.ddp.block import OLMoDDPTransformerBlockConfig
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.moe.v2.ep_config import ExpertParallelConfig
from olmo_core.nn.moe.v2.router import MoERouterConfigV2
from olmo_core.nn.transformer import OLMoDDPModelConfig
from olmo_core.nn.vision import (
    Molmo2TokenIds,
    MultimodalLMConfig,
    VisionConnectorConfig,
    VisionEncoderConfig,
)


@pytest.fixture
def alignment_recipe(tmp_path, monkeypatch):
    base = tmp_path / "pretraining"
    base.mkdir()
    tokenizer = TokenizerConfig.dolma2()
    lm = OLMoDDPModelConfig(
        d_model=64,
        vocab_size=100352,
        n_layers=2,
        block=OLMoDDPTransformerBlockConfig(sequence_mixer=AttentionConfig(n_heads=4)),
        lm_head=LMHeadConfig(),
    )
    dataset = NumpyFSLDatasetConfig(
        tokenizer=tokenizer, sequence_length=8192, paths=["s3://pretraining/tokens.npy"]
    )
    (base / "config.json").write_text(
        json.dumps({"model": lm.as_config_dict(), "dataset": dataset.as_config_dict()})
    )
    token_ids = Molmo2TokenIds(
        im_start_id=100278,
        im_end_id=100279,
        im_patch_id=100280,
        im_col_id=100281,
        low_res_im_start_id=100282,
        image_placeholder_id=100283,
        im_end_turn_id=100264,
    )
    monkeypatch.setattr(
        MultimodalMixtureConfig, "build_tokenizer", Mock(return_value=(tokenizer, token_ids))
    )
    load_hf = Mock(return_value=object())
    monkeypatch.setattr(vision_alignment, "load_molmo2_hf_vision_config", load_hf)

    def make_multimodal(hf_config, language_model, image_patch_token_id):
        vision = VisionEncoderConfig()
        return MultimodalLMConfig(
            lm=language_model,
            vision=vision,
            connector=VisionConnectorConfig.from_vision_encoder(
                vision, output_dim=language_model.d_model
            ),
            image_patch_token_id=image_patch_token_id,
        )

    monkeypatch.setattr(vision_alignment, "multimodal_config_from_molmo2_vision", make_multimodal)
    constructed_sources = []

    def visual_sources(phase, sequence_length, artifact_root, split="train"):
        sources = {
            name: PixMoCapDatasetConfig(
                dataset_path=f"{artifact_root}/{name}",
                split=split,
                max_sequence_length=sequence_length,
            )
            for name in ALIGNMENT_LOSS_TARGETS[phase]
            if name != "native_text_replay"
        }
        constructed_sources.append(sources)
        return sources

    monkeypatch.setattr(vision_alignment, "build_visual_sources", visual_sources)

    def build(phase="bridge", parent=None, overrides=(), include_means=True):
        args = [
            f"--recipe.phase={phase}",
            f"--recipe.artifact_root={tmp_path}/artifacts",
            f"--recipe.output_root={tmp_path}/runs",
            f"--recipe.work_dir={tmp_path}/cache",
        ]
        if parent is None:
            args.append(f"--recipe.pretraining_checkpoint={base}")
        else:
            args.append(f"--recipe.parent_checkpoint={parent}")
        if include_means:
            args.extend(
                f"--dataset.mean_loss_weight.{name}={mean}"
                for name, mean in ALIGNMENT_MEAN_LOSS_WEIGHTS[phase].items()
            )
        return vision_alignment.build_config(
            CliContext(
                script="src/scripts/train/Vision-Align.py",
                cmd=SubCmd.dry_run,
                run_name=f"alignment-{phase}",
                cluster="local",
                overrides=[*args, *overrides],
            )
        )

    def save(config):
        checkpoint = tmp_path / f"{config.recipe.phase}-checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps(config.as_config_dict()))
        return checkpoint

    def set_router_coefficients(default_weight):
        path = base / "config.json"
        saved = json.loads(path.read_text())
        model = OLMoDDPModelConfig.from_dict(saved["model"])
        model.n_layers = 4
        model.block.routed_experts_router = MoERouterConfigV2(
            d_model=model.d_model,
            num_experts=12,
            top_k=3,
            lb_loss_weight=default_weight,
            lb_loss_granularity="instance",
            z_loss_weight=0.003,
        )
        model.block.ep = ExpertParallelConfig(capacity_factor=1.25)
        model.block_overrides = {index: model.block.copy() for index in (1, 2, 3)}
        for index, coefficient in ((1, None), (2, 0.0), (3, 0.07)):
            model.block_overrides[index].routed_experts_router.lb_loss_weight = coefficient
        saved["model"] = model.as_config_dict()
        path.write_text(json.dumps(saved))
        return model

    return SimpleNamespace(
        build=build,
        save=save,
        set_router_coefficients=set_router_coefficients,
        base=base,
        load_hf=load_hf,
        token_ids=token_ids,
        constructed_sources=constructed_sources,
    )
