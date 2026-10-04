"""Freeze one resolved source allocation for all three MT arms."""

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass

import olmoe3_adaptive_mt_plan as p
from olmoe3_lr_sweep_watch import atomic_json

from olmo_core.data.source_mixture import (
    SourceMixtureDataset, SourceMixtureDatasetConfig, SourceMixtureOutcome, SourcePathTokens,
)
from olmo_core.io import get_bytes_range, get_file_size


@dataclass
class FrozenMTMixture(SourceMixtureDatasetConfig):
    """Resolve globs once, and persist the exact Hamilton allocation and source order."""

    def build(self, *, npdtype, sequence_length):
        folder = p.AUTO / "data"
        folder.mkdir(parents=True, exist_ok=True)
        recipe = dict(config=self.as_dict(json_safe=True, exclude_private_fields=True), dtype=str(npdtype), sequence=sequence_length)
        recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
        with (folder / "mixture.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            file = folder / "mixture.json"
            if not file.exists():
                mixture = super().build(npdtype=npdtype, sequence_length=sequence_length)
                inventory, samples = {}, []
                for source in mixture.sources:
                    paths = list(dict.fromkeys(str(x.path) for x in source.path_tokens))
                    for path in paths:
                        inventory[path] = get_file_size(path)
                    for i in sorted({0, len(paths) // 2, len(paths) - 1}):
                        path = paths[i]
                        size = inventory[path]
                        for start in sorted({0, max(0, size - 65536)}):
                            count = min(size, 65536)
                            samples.append(dict(path=path, start=start, count=count,
                                sha256=hashlib.sha256(get_bytes_range(path, start, count)).hexdigest()))
                atomic_json(file, dict(recipe=recipe, recipe_sha256=recipe_hash,
                    sources=[asdict(s) for s in mixture.sources], inventory=inventory,
                    samples=samples, sample_scope="head/tail of first/middle/last file per source"))
            row = json.loads(file.read_text())
            assert row["recipe_sha256"] == recipe_hash
        return SourceMixtureDataset(sources=[
            SourceMixtureOutcome(name=s["name"], path_tokens=[SourcePathTokens(**t) for t in s["path_tokens"]])
            for s in row["sources"]
        ])


def components(common):
    import olmoe3_hero_mt as mt

    mt.BATCH, mt.REQUESTED_TOKENS, mt.SEED = p.BATCH, p.REQUESTED_TOKENS, p.SEED
    data = mt.data_components(common)
    config = data.dataset.source_mixture_config
    data.dataset.source_mixture_config = FrozenMTMixture(**{
        k: getattr(config, k) for k in config.__dataclass_fields__
    })
    return data


def verify_inputs():
    """Recheck metadata and bounded content samples once per arm's startup."""
    row = json.loads((p.AUTO / "data/mixture.json").read_text())
    for path, size in row["inventory"].items():
        assert get_file_size(path) == size, path
    for sample in row["samples"]:
        data = get_bytes_range(sample["path"], sample["start"], sample["count"])
        assert hashlib.sha256(data).hexdigest() == sample["sha256"], sample["path"]
    return hashlib.sha256((p.AUTO / "data/mixture.json").read_bytes()).hexdigest()
