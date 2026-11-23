"""CPT-only boundary detection for numbered-claim contradiction prompts."""
import re

import torch

_CLAIM_HEADER = re.compile(r"(?m)^Claim [1-9][0-9]*:")


def claim_boundary_mask(input_ids, output_mask, valid_mask, tokenizer):
    """Return an unshifted target mask marking the first token of each claim header.

    Headers belong to their claims; preceding newlines belong to the preceding
    segment. Only the transition into the first header token is excluded. No
    attention mask, answer target, or within-claim target is changed. Full examples
    are expected (as in the contradiction BFD recipe). Token offsets are accepted
    only after an exact decode/encode round trip, including special/padding tokens.
    """
    if input_ids.shape != output_mask.shape or input_ids.shape != valid_mask.shape:
        raise ValueError("claim boundary masks must have the same shape as input_ids")
    result = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, ids in enumerate(input_ids.detach().cpu().tolist()):
        eligible = (valid_mask[row].bool() & ~output_mask[row].bool()).cpu()
        if not eligible.any():
            continue
        text = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if encoded['input_ids'] != ids:
            raise ValueError("claim boundary tokenizer round-trip changed input IDs")
        starts = {start: i for i, (start, end) in enumerate(encoded['offset_mapping']) if end > start}
        found = []
        for match in _CLAIM_HEADER.finditer(text):
            if match.start() not in starts:
                raise ValueError("claim boundary is not aligned with a token start")
            index = starts[match.start()]
            if eligible[index]:
                found.append(index)
        if not found:
            raise ValueError("no claim headers found in valid non-output tokens")
        result[row, found] = True
    return result
