"""Measure the actual 256K-corpus -> <=32K selection and packing on a Weka host.

This is CPU-only. Run before training; do not infer a step count from the old ladder.
"""
import argparse
import hashlib
import json
from pathlib import Path

from _qwen35_contradiction_common import (
    BASE_CHECKPOINT,
    DATA_ROOT,
    SEQUENCE_LENGTH,
    build_leaky_experiment,
)
from sweep_settings import CLUSTER, GPUS
from olmo_core.data.utils import get_leaky_labels
from olmo_core.internal.experiment import CliContext, SubCmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    config = build_leaky_experiment(
        CliContext(
            script=__file__,
            cmd=SubCmd.prep,
            run_name="leakysft-xlong-filter32k-prep",
            cluster=CLUSTER,
            overrides=[],
        ),
        arm="a00",
    )
    metadata_path = Path(DATA_ROOT) / "metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    if not (Path(BASE_CHECKPOINT) / ".metadata").is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    packed = config.dataset[0].build(config.data_loader.work_dir)
    counts = dict(retained_examples=0, excluded_examples=0, retained_tokens=0, excluded_tokens=0)
    longest = 0
    for source in packed.sources:
        for start, end in source.get_document_offsets():
            size = int(end - start)
            longest = max(longest, size)
            group = "retained" if size <= SEQUENCE_LENGTH else "excluded"
            counts[f"{group}_examples"] += 1
            counts[f"{group}_tokens"] += size
    if not counts["retained_examples"] or not len(packed):
        raise ValueError("No examples remain after the 32K filter")
    loader = config.data_loader.build(packed)
    loader.reshuffle(epoch=1)
    assert len(loader) * GPUS == len(packed), "epoch must consume every packed window"
    batch = loader.get_mock_batch()
    labels, output = get_leaky_labels(batch)
    assert (labels != -100).any() and output.any()
    report = dict(
        data_root=DATA_ROOT,
        max_example_tokens=SEQUENCE_LENGTH,
        filter="retain complete tokenized examples <=32768 tokens; exclude, never truncate",
        metadata=json.loads(metadata_bytes),
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        source_fingerprints=[source.fingerprint for source in packed.sources],
        packed_fingerprint=packed.fingerprint,
        longest_original_example=longest,
        packed_windows=len(packed),
        steps_per_epoch=len(loader),
        epochs=1,
        dropped_tail_windows=0,
        base_checkpoint=BASE_CHECKPOINT,
        **counts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
