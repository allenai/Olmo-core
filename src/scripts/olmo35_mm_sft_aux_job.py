"""Launch a one-GPU helper job for the multimodal SFT gates (HF reference CE, export) on holmes.

    python olmo35_mm_sft_aux_job.py NAME "SHELL COMMAND" [--description TEXT]
"""

import argparse
import json
import re
import subprocess
import tempfile

IMAGE = "akshitab/olmo-core-tch2130cu130-fa4-rma-2026-09-11"
HF = "/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/hf_cache"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("command")
    p.add_argument("--description", default="")
    p.add_argument("--gpus", type=int, default=1)
    a = p.parse_args()
    task = {
        "name": a.name,
        "image": {"beaker": IMAGE},
        "command": ["/bin/bash", "-c"],
        "arguments": [a.command],
        "envVars": [
            {"name": "HF_HUB_CACHE", "value": f"{HF}/hub"},
            {"name": "HF_HOME", "value": f"{HF}/huggingface"},
            {"name": "HF_HUB_OFFLINE", "value": "1"},
            {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
        ],
        "datasets": [
            {"mountPath": f"/weka/{b}", "source": {"weka": b}}
            for b in (
                "oe-training-default",
                "olmo-3p5-mid-checkpoints",
                "olmo-3p5-checkpoints",
                "olmo-3p5-post-checkpoints",
            )
        ],
        "result": {"path": "/output"},
        "resources": {"gpuCount": a.gpus, "sharedMemory": "64GiB"},
        "context": {"priority": "urgent", "preemptible": False},
        "constraints": {"cluster": ["ai2/holmes"]},
        "timeout": "3h",
    }
    spec = {
        "version": "v2",
        "budget": "ai2/oe-other",
        "description": a.description or a.name,
        "tasks": [task],
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(spec, f)
    r = subprocess.run(
        ["beaker", "experiment", "create", f.name, "--workspace", "ai2/olmo-instruct", "--name", a.name],
        capture_output=True,
        text=True,
    )
    print((re.findall(r"01[A-Z0-9]{24}", r.stdout + r.stderr) or [r.stderr[-400:]])[0])


if __name__ == "__main__":
    main()
