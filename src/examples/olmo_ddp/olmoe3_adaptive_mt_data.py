"""Freeze one resolved source allocation for all three MT arms."""

import fcntl
import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import SimpleNamespace

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
                assert os.environ.get("ADAPTIVE_MT_PREPARE") == "1", "Run CPU data preparation first"
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


def freeze_config(config):
    """Copy constructor fields only; Config also defines inherited ClassVars."""
    return FrozenMTMixture(**{f.name: getattr(config, f.name) for f in fields(config) if f.init})


def components(common):
    # This path deliberately avoids importing GPU kernels on the CPU prep worker.
    import olmo_core
    from olmo_core.data import InstanceFilterConfig, NumpyDataLoaderConfig, NumpyFSLDatasetConfig
    from olmo_core.data.source_mixture import SourceMixtureList

    mix = Path(olmo_core.__file__).parent / "data/source_mixtures/OLMo3-32B-midtraining-modelnamefilter.yaml"
    assert hashlib.sha256(mix.read_bytes()).hexdigest() == "1ed52c81c3f33fb864157f8e01ed0d0aff24548149c2ca4376182578181c25ba"
    sources = SourceMixtureList.from_yaml(str(mix))
    sources.validate()
    return SimpleNamespace(
        dataset=NumpyFSLDatasetConfig.from_src_mix(
            src_mix=FrozenMTMixture(source_list=sources, requested_tokens=p.REQUESTED_TOKENS,
                global_batch_size=p.BATCH, processes=16, seed=p.SEED),
            tokenizer=common.tokenizer, work_dir=common.work_dir, sequence_length=8192,
            instance_filter_config=InstanceFilterConfig()),
        data_loader=NumpyDataLoaderConfig(global_batch_size=p.BATCH, seed=p.SEED,
            num_workers=8, prefetch_factor=8, num_threads=4),
    )


def verify_inputs():
    """Recheck metadata and bounded content samples once per arm's startup."""
    row = json.loads((p.AUTO / "data/mixture.json").read_text())
    def size_check(item):
        path, size = item
        # Bypass the persistent metadata cache for the actual drift check.
        assert get_file_size.__wrapped__(path) == size, path

    def sample_check(sample):
        data = get_bytes_range(sample["path"], sample["start"], sample["count"])
        assert hashlib.sha256(data).hexdigest() == sample["sha256"], sample["path"]

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(size_check, row["inventory"].items()))
        list(pool.map(sample_check, row["samples"]))
    return hashlib.sha256((p.AUTO / "data/mixture.json").read_bytes()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_prepared():
    """Cheap local startup check; remote corpus validation happens before allocation."""
    row = json.loads((p.AUTO / "data/prepared.json").read_text())
    assert row["passed"] and row["steps"] == p.END and row["seed"] == p.SEED
    assert row["manifest_sha256"] == sha256_file(p.AUTO / "data/mixture.json")
    assert row["work_dir"] == str(p.DATA_WORK)
    for path, info in row["files"].items():
        assert Path(path).is_relative_to(p.DATA_WORK)
        assert Path(path).stat().st_size == info["bytes"]
        assert sha256_file(path) == info["sha256"], path
    return dict(passed=True, manifest_sha256=row["manifest_sha256"],
                prepared_sha256=sha256_file(p.AUTO / "data/prepared.json"))


def prepare():
    """Finish source allocation, dataset indices and epoch order without allocated GPUs."""
    from olmo_core.data import TokenizerConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    folder = p.AUTO / "data"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / "prepared.json").exists():
            verify_prepared()
            print("ADAPTIVE_MT_DATA_PREPARED_ALREADY", flush=True)
            return
        os.environ["ADAPTIVE_MT_PREPARE"] = "1"
        config = components(SimpleNamespace(tokenizer=TokenizerConfig.dolma2(), work_dir=str(p.DATA_WORK)))
        dataset = config.dataset.build()
        print("ADAPTIVE_MT_MIXTURE_FROZEN", flush=True)
        manifest_sha256 = verify_inputs()
        config.data_loader.num_workers = 0
        config.data_loader.prefetch_factor = None
        config.data_loader.target_device_type = "cpu"
        loader = config.data_loader.build(dataset)
        loader.reshuffle(epoch=1)
        assert loader.total_batches == p.END
        files = {str(f): dict(bytes=f.stat().st_size, sha256=sha256_file(f))
                 for f in sorted(p.DATA_WORK.rglob("*.npy"))}
        assert files
        atomic_json(folder / "prepared.json", dict(passed=True, steps=p.END, seed=p.SEED,
            work_dir=str(p.DATA_WORK), manifest_sha256=manifest_sha256, files=files,
            dataset_fingerprint=dataset.fingerprint, commit=os.environ.get("GIT_REF")))
        verify_prepared()
        print("ADAPTIVE_MT_DATA_PREPARED", len(files), flush=True)
