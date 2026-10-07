"""
Helper process that rasterises olmOCR-mix PDF pages for :mod:`.olmocr`.

pypdfium2 is not thread-safe, so a training process can only render one page at a time in its
loader threads. :class:`~olmo_core.data.multimodal.olmocr.PdfRenderPool` instead runs a few
copies of this module as child processes (``python <this file>``) and hands each page to an idle
one over a pipe, so pages render in parallel and the loader threads only wait on I/O.

This file is deliberately self-contained (standard library, ``pypdfium2`` and ``PIL`` only) and
is executed *by path*, not imported as part of the package: a ``multiprocessing`` child of the
trainer would import ``olmo_core`` (and ``torch``, ~1 GB and tens of seconds per worker) and
re-run the training script as ``__mp_main__``; a worker started this way imports pdfium in a few
tens of milliseconds and stays small.

Protocol (one request at a time per worker): the parent writes one JSON line
``{"pdf_path": str, "target": int}`` to stdin; the worker answers with one JSON header line,
``{"ok": true, "width": W, "height": H, "size": N}`` followed by exactly ``N`` bytes of
interleaved 8-bit RGB pixels, or ``{"ok": false, "type": str, "error": str}``. EOF on stdin ends
the worker.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Tuple


def render_rgb(pdf_path: str, target_longest_image_dim: int) -> Tuple[int, int, bytes]:
    """Rasterise a single-page PDF so its longest side is ``target_longest_image_dim`` pixels.

    This is the one pypdfium2 call the olmOCR source makes, in-process or in a helper: the
    render scale is ``target / longest_mediabox_dim_in_points`` (olmOCR's per-page DPI rule).

    :returns: ``(width, height, pixels)`` with ``pixels`` the image's interleaved 8-bit RGB bytes
        (``PIL.Image.frombytes("RGB", (width, height), pixels)`` rebuilds it).
    :raises RuntimeError: If the PDF has more than one page.
    """
    import pypdfium2

    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        if len(pdf) != 1:
            raise RuntimeError(f"{pdf_path}: expected a pre-split single-page PDF, got {len(pdf)}")
        page = pdf[0]
        longest_dim = max(page.get_size())  # (width, height) in PDF points
        scale = target_longest_image_dim / longest_dim
        image = page.render(scale=scale).to_pil().convert("RGB")
        width, height = image.size
        return width, height, image.tobytes()
    finally:
        pdf.close()


def serve(reader, writer) -> None:
    """Answer render requests from ``reader`` on ``writer`` until EOF (see the module docstring)."""
    for line in reader:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            width, height, pixels = render_rgb(str(request["pdf_path"]), int(request["target"]))
        except BaseException as e:  # noqa: B036 -- report, do not die: the parent retries
            header = {"ok": False, "type": type(e).__name__, "error": str(e)}
            writer.write(json.dumps(header).encode("utf-8") + b"\n")
        else:
            header = {"ok": True, "width": width, "height": height, "size": len(pixels)}
            writer.write(json.dumps(header).encode("utf-8") + b"\n")
            writer.write(pixels)
        writer.flush()


def main() -> None:
    """Serve on stdin/stdout. Anything a library prints to fd 1 would corrupt the byte stream,
    so the protocol takes a private copy of stdout and fd 1 is redirected to stderr."""
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    serve(sys.stdin.buffer, out)


if __name__ == "__main__":
    main()
