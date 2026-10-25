"""
Peak-memory attribution from a ``torch.cuda.memory._snapshot()``: replay the allocator trace to
the moment of peak allocated memory and print the live blocks at that moment grouped by their
allocation site (first frame inside ``olmo_core``, else the innermost frame).
"""

from __future__ import annotations

import collections
from typing import Any, Dict


_CPP_SKIP = ("unwind", "CapturedTraceback", "gather(", "c10::", "Allocator", "at::detail::empty", "at::empty", "at::_ops::empty", "at::native::empty",
             "torch::empty", "at::TensorBase", "at::Tensor::", "std::", "_start", "__libc", "PyObject", "_Py", "Py_")


def _cpp_site(frames) -> str:
    """A backward allocation has no Python frame: name the autograd node (``...Backward``) or the first op frame."""
    first = None
    for f in frames or []:
        n = f.get("name", "") or ""
        if "Backward" in n or "autograd::generated" in n:
            return "C++ " + n[:110]
        if first is None and n and not any(k in n for k in _CPP_SKIP):
            first = n
    return "C++ " + (first or "?")[:110]


def _site(frames) -> str:
    for f in frames or []:
        fn = f.get("filename", "")
        if "olmo_core" in fn or "memexpress" in fn:
            return f"{fn.split('/')[-1]}:{f.get('line')} {f.get('name')}"
    if frames and all(f.get("filename", "??") in ("??", "") or f.get("filename", "").endswith((".cpp", ".h", ".cu"))
                      or ".so" in f.get("filename", "") for f in frames[:3]):
        return _cpp_site(frames)
    if frames:
        f = frames[0]
        return f"{f.get('filename', '?').split('/')[-1]}:{f.get('line')} {f.get('name')}"
    return "?"


def summarize_peak(snap: Dict[str, Any], top: int = 16) -> None:
    """Print the live set at the peak of the recorded trace, grouped by site."""
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank != 0:
        return
    live: Dict[int, tuple] = {}
    frames_of: Dict[int, list] = {}
    cur = peak = 0
    peak_live: Dict[int, tuple] = {}
    for trace in snap.get("device_traces", []):
        for ev in trace:
            act, addr, size = ev.get("action"), ev.get("addr"), ev.get("size", 0)
            if act == "alloc":
                live[addr] = (size, _site(ev.get("frames")))
                frames_of[addr] = ev.get("frames") or []
                cur += size
                if cur > peak:
                    peak = cur
                    peak_live = dict(live)
            elif act in ("free_completed",):
                if addr in live:
                    cur -= live.pop(addr)[0]
        break  # device 0 only
    by = collections.Counter()
    for size, site in peak_live.values():
        by[site] += size
    sample = {}
    for addr, (size, site) in peak_live.items():
        if site.startswith("C++") and (site not in sample or size > sample[site][0]):
            sample[site] = (size, addr)
    print(f"[mem-snapshot] peak live {peak / 2**30:.2f} GB in {len(peak_live)} blocks; top sites:", flush=True)
    for site, size in by.most_common(top):
        print(f"[mem-snapshot]   {size / 2**30:6.2f} GB  {site}", flush=True)
    # the largest C++-only groups: one sample allocation's call stack (innermost first), to name the backward kernel
    for site, size in [kv for kv in by.most_common(top) if kv[0].startswith("C++")][:3]:
        fr = frames_of.get(sample[site][1], [])
        names = [f.get("name", "?")[:90] for f in fr if not any(k in (f.get("name") or "") for k in _CPP_SKIP)][:14]
        print(f"[mem-snapshot]   stack of {site[:60]} ({size / 2**30:.2f} GB): " + " <- ".join(names), flush=True)
