import copy

import pytest
import torch

from olmo_core.data.collator import DataCollator, PaddingDirection
from olmo_core.data.composable import ComposableDataLoader, PackingInstanceSource
from olmo_core.data.composable.landmark_packing_instance_source import LandmarkPackingInstanceSource
from olmo_core.data.composable.token_source import InMemoryDocumentSource
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.data import utils


def test_shift_boundaries_structural_and_instance_masks():
    batch = dict(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1]*8]),
                 label_mask=torch.tensor([[False, False, True, False, False, True, True, False], [True]*8]),
                 loss_valid_mask=torch.tensor([[True, True, True, False, True, True, False, False], [True]*8]),
                 doc_lens=torch.tensor([[4, 4], [8, 0]]), instance_mask=torch.tensor([True, False]))
    old = copy.deepcopy(batch)
    labels, output = utils.get_leaky_labels(batch)
    assert labels.tolist() == [[2, 3, -100, -100, 6, -100, -100, -100], [-100]*8]
    assert output.tolist() == [[False, True, False, False, True, False, False, False], [False]*8]
    for key in old:
        torch.testing.assert_close(batch[key], old[key])


def test_missing_validity_fails_closed():
    with pytest.raises(ValueError, match="loss_valid_mask"):
        utils.get_leaky_labels(dict(input_ids=torch.tensor([[1, 2, 3]]),
                                   label_mask=torch.tensor([[False, True, True]])))


@pytest.mark.parametrize("direction", [PaddingDirection.left, PaddingDirection.right])
def test_collator_preserves_validity_and_masks_padding(direction):
    items = [dict(input_ids=[1, 2, 3], label_mask=[False, True, True], loss_valid_mask=[True]*3),
             dict(input_ids=[4, 5], label_mask=[False, True], loss_valid_mask=[True]*2)]
    batch = DataCollator(pad_token_id=9, pad_direction=direction)(items)
    assert batch["loss_valid_mask"].dtype == torch.bool
    expected = [False, True, True] if direction == PaddingDirection.left else [True, True, False]
    assert batch["loss_valid_mask"][1].tolist() == expected
    labels, _ = utils.get_leaky_labels(batch)
    # Never train a prediction from a left-padding position into the first real token.
    if direction == PaddingDirection.left:
        assert labels[1, 0].item() == -100


@pytest.mark.parametrize("kind", ["dense", "landmark"])
@pytest.mark.parametrize("exclude_predictors", [False, True])
def test_packer_and_loader_preserve_real_special_ids(tmp_path, kind, exclude_predictors):
    tok = TokenizerConfig(vocab_size=100, eos_token_id=99, pad_token_id=98)
    docs = InMemoryDocumentSource(tokens=[50, 98, 2, 99, 7, 99], tokenizer=tok, work_dir=tmp_path)
    if kind == "dense":
        source = PackingInstanceSource(docs, sequence_length=16, tokenizer=tok, work_dir=tmp_path)
    else:
        source = LandmarkPackingInstanceSource(docs, sequence_length=16, mem_freq=3,
                    mem_id=50, pad_id=98, exclude_landmark_predictors=exclude_predictors, work_dir=tmp_path)
    inst = source[0]
    mask = list(inst["loss_valid_mask"])
    ids = list(inst["input_ids"])
    if kind == "dense":
        assert mask == [True]*6+[False]*10
    else:
        assert mask == [True, True, True, False, not exclude_predictors, False, False, False,
                        True, True, False, False, False, False, False, False]
    assert ids[0] == 50 and mask[0]  # naturally occurring landmark ID
    assert ids[1] == 98 and mask[1]  # naturally occurring pad ID
    loader = ComposableDataLoader(source, tokenizer=tok, collator=DataCollator(pad_token_id=98),
             global_batch_size=16, seed=1, target_device_type="cpu", work_dir=tmp_path/"loader",
             generate_doc_lengths=True)
    loader.reshuffle(1)
    batch = next(iter(loader))
    assert batch["loss_valid_mask"][0].tolist() == mask


def test_attention_mask_excludes_targets_and_padding_predictors():
    batch = dict(input_ids=torch.tensor([[9, 1, 2, 3, 9]]),
                 loss_valid_mask=torch.ones(1, 5, dtype=torch.bool),
                 label_mask=torch.ones(1, 5, dtype=torch.bool),
                 attention_mask=torch.tensor([[0, 1, 1, 1, 0]]))
    labels, output = utils.get_leaky_labels(batch)
    assert labels.tolist() == [[-100, 2, 3, -100, -100]]
    assert output.tolist() == [[False, True, True, False, False]]
