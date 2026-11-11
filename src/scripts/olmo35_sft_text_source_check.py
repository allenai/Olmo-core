#!/usr/bin/env python3
"""Check the multimodal trainer's text source on the real data of the current text SFT recipe.

The text SFT cache (``bf82a8fab8-6068a350-rowaligned``, 449,487 packed instances of 65,536
tokens) plus the 10% long-context replay dir the GPU text SFT runs used (58,480 rows, one
document plus EOS each) must pack, through :class:`SftNumpyCacheConfig`, into the instances
those runs indexed: 449,487 for the cache alone and 507,967 with the replay dir (the GPU runs'
loader indices were all below 507,967). Also validates the replay dir as open-instruct does and
reports its tokens and loss tokens, and spot-checks instances from both parts::

    PYTHONPATH=src python src/scripts/olmo35_sft_text_source_check.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CACHE = (
    "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c12/a7/cache-neptune/numpy_sft/"
    "bf82a8fab8-6068a350-rowaligned"
)
REPLAY = "/weka/olmo-3p5-post-checkpoints/abhishekr/h182/replay-b1c12r"
EXPECTED_CACHE_INSTANCES = 449_487
EXPECTED_WITH_REPLAY = 507_967
EOS = 100257


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cache", default=CACHE)
    parser.add_argument("--replay", default=REPLAY)
    args = parser.parse_args()
    from olmo_core.data.multimodal.sft_numpy_cache import (
        SftNumpyCacheConfig,
        check_extra_numpy_dir,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    receipt: dict = {"cache": args.cache, "replay": args.replay}
    t0 = time.time()
    cache_only = SftNumpyCacheConfig(numpy_dir=args.cache, work_dir=str(out / "work-cache"))
    receipt["replay_dir"] = check_extra_numpy_dir(
        args.replay,
        token_dtype=cache_only.dataset_config().get_dtype(),
        vocab_size=cache_only.tokenizer.vocab_size,
        eos_token_id=cache_only.tokenizer.eos_token_id,
    )
    a = cache_only.build()
    b = SftNumpyCacheConfig(
        numpy_dir=args.cache, work_dir=str(out / "work-replay"), extra_numpy_dirs=[args.replay]
    ).build()
    receipt["instances_cache"] = len(a)
    receipt["instances_with_replay"] = len(b)
    rng = np.random.RandomState(0)
    same = all(
        np.array_equal(a[i]["input_ids"], b[i]["input_ids"])
        for i in rng.choice(len(a), 64, replace=False).tolist()
    )
    receipt["cache_instances_unchanged_by_replay"] = bool(same)
    replay_checks = []
    for i in rng.choice(np.arange(len(a), len(b)), 16, replace=False).tolist():
        ex = b[i]
        docs = ex["document_ids"]
        trained = []
        for d in np.unique(docs):
            m = docs == d
            ids = ex["input_ids"][m]
            # A replay document trains every position but its last (whose target is outside).
            trained.append(bool(ids[-1] == EOS and (ex["loss_masks"][m][:-1] > 0).all()))
        replay_checks.append({"instance": i, "documents": len(trained), "all_trained": trained})
    receipt["replay_instances_spot_check"] = replay_checks
    receipt["passes"] = {
        "cache_instances": len(a) == EXPECTED_CACHE_INSTANCES,
        "with_replay_instances": len(b) == EXPECTED_WITH_REPLAY,
        "cache_unchanged": bool(same),
        "replay_documents_fully_trained": all(all(c["all_trained"]) for c in replay_checks),
    }
    receipt["seconds"] = round(time.time() - t0, 1)
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
