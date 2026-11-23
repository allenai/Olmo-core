"""Boundary-masked interpolation and additive CPT vs the NormalizedLeakySFT control.

Readout: contradiction F1 at 2K–128K and loss/throughput diagnostics. Same one-epoch
corpus, batch, optimizer, and base checkpoint. Compute and gradient scale are not
matched: boundary detection adds CPU work; addition keeps the SFT coefficient one.
"""
from _qwen35_contradiction_common import build_leaky_experiment
from sweep_settings import NORMALIZED_ARMS

VARIANTS = {'boundary': 'BoundaryMaskedNormalizedLeakySFT', 'additive': 'AdditiveLeakySFT'}


def build_variant_experiment(context, *, arm, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown LeakySFT variant: {variant}')
    if context.overrides:
        raise ValueError('Variant sweep uses the fixed control recipe; CLI overrides are disabled')
    config = build_leaky_experiment(context, arm=arm, normalized=True)
    alpha = NORMALIZED_ARMS[arm]
    if variant == 'additive':
        config.train_module.normalized_leaky_sft_alpha = None
        config.train_module.additive_leaky_sft_alpha = alpha
    else:
        config.train_module.cpt_claim_boundary_tokenizer = 'Qwen/Qwen3.5-0.8B'
    config.trainer.callbacks['wandb'].group = VARIANTS[variant] + '-contradiction-xlong-filter32k-1ep'
    return config
