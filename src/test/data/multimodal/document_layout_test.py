import numpy as np

from olmo_core.data.multimodal.document_layout import (
    branch_context_ids,
    document_prompt_ids,
    image_prefix_ids,
    message_ids,
    response_ids,
)
from olmo_core.data.multimodal.rng import make_random_state
from olmo_core.nn.vision import Molmo2TokenIds


class _Tokenizer:
    eos_token_id = 7

    def __init__(self):
        self.encoded = []

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        self.encoded.append(text)
        return list(text.encode())


def test_s002_document_layout_has_eos_boundary_and_no_role_headers():
    tokenizer = _Tokenizer()
    token_ids = Molmo2TokenIds(
        im_start_id=20,
        im_end_id=21,
        im_patch_id=22,
        im_col_id=23,
        low_res_im_start_id=24,
        image_placeholder_id=25,
        im_end_turn_id=26,
    )
    prefix = image_prefix_ids(tokenizer, np.asarray([1, 1, 1, 1]), token_ids=token_ids)

    assert prefix == [7, 24, 22, 21, 20, 22, 23, 21]
    branch_context_ids(tokenizer, "Describe this image.")
    response_ids(tokenizer, "A cat sits here.")
    assert tokenizer.encoded == ["Describe this image.", " A cat sits here."]
    assert all("<|im_start|>" not in text for text in tokenizer.encoded)


def test_message_format_none_adds_space_only_after_first_message():
    tokenizer = _Tokenizer()
    message_ids(tokenizer, "first", first=True)
    message_ids(tokenizer, "second", first=False)
    assert tokenizer.encoded == ["first", " second"]


def test_s002_inference_prompt_matches_training_prefix_and_response_tokens():
    tokenizer = _Tokenizer()
    prompt = document_prompt_ids(tokenizer, "Choose A or B", image_ids=[20, 21])
    option = response_ids(tokenizer, "A")

    assert prompt == [7, 20, 21, *list(b"Choose A or B")]
    assert option == list(b" A")
    assert tokenizer.encoded == ["Choose A or B", " A"]


def test_molmo_random_state_is_coordinate_deterministic():
    first = make_random_state(12, 3).randint(0, 2**31, size=8)
    same = make_random_state(12, 3).randint(0, 2**31, size=8)
    other_epoch = make_random_state(12, 4).randint(0, 2**31, size=8)
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other_epoch)
