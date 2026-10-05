"""PixMo clocks dataset formatting (port of mm_olmo PixMoClocks)."""

from __future__ import annotations

from os.path import join
from typing import Any, Dict, Union

import numpy as np
from PIL import Image, ImageOps

from .paths import PIXMO_DATASETS

__all__ = ["format_pixmo_clocks_row"]


def _open_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def format_pixmo_clocks_row(
    row: Dict[str, Any], rng: np.random.RandomState, *, aug: bool = True
) -> Dict[str, Any]:
    """Format one PixMo clocks JSONL row into an mm_olmo-compatible example dict."""
    # torchvision is imported here, not at module load: the academic registry imports this module,
    # and so does every stage-1 run, clocks or not. An image whose torchvision does not match its
    # torch (e.g. torch upgraded to 2.13 by the package install, torchvision left at 0.25) then only
    # fails a run that actually trains on clocks.
    import torchvision
    import torchvision.transforms.functional as VF
    from torchvision.transforms.functional import InterpolationMode, affine

    time_format = row["time_format"]
    shows_seconds = row["shows_seconds"]
    hour, minute, second = [int(row[k]) for k in ("hour", "minute", "second")]
    hour_str: Union[str, int]
    if hour == 0:
        hour_str = "12"
        am_pm = "AM"
    elif hour > 12:
        am_pm = "PM"
        hour_str = hour - 12
    else:
        hour_str = hour
        am_pm = "AM"
    hour_str = str(hour_str)
    minute_str = str(minute)
    if len(minute_str) == 1:
        minute_str = "0" + minute_str
    second_str = str(second)
    if len(second_str) == 1:
        second_str = "0" + second_str

    if time_format == "The time is not shown":
        text = "The time is not shown in the image."
        hour, minute, second = -1, -1, -1
    else:
        if not shows_seconds:
            second = -1
        if time_format == "12 hour clock (without AM/PM)" and shows_seconds:
            if hour >= 12:
                hour = hour - 12
            time = "".join([hour_str, ":", minute_str, ":", second_str])
        elif time_format == "12 hour clock (with AM/PM)" and shows_seconds:
            time = "".join([hour_str, ":", minute_str, ":", second_str, " ", am_pm])
        elif time_format == "12 hour clock (with AM/PM)" and not shows_seconds:
            time = "".join([hour_str, ":", minute_str, " ", am_pm])
        elif time_format == "12 hour clock (without AM/PM)" and not shows_seconds:
            if hour >= 12:
                hour = hour - 12
            time = "".join([hour_str, ":", minute_str])
        else:
            raise RuntimeError(time_format)
        text = "".join(["The time shown is ", time])

    image = _open_image(join(PIXMO_DATASETS, "clocks", "images", row["image"]))
    image = image.crop((0, 0, image.width, image.height - 120))

    if aug:
        sel = rng.random()
        if sel < 0.1:
            shear_x = 0.0
            shear_y = 0.0
            rotation = 0.0
        elif sel < 0.5:
            shear_x = rng.uniform(-10, 10)
            shear_y = rng.uniform(-10, 10)
            rotation = rng.uniform(-25, 25)
        else:
            if rng.random() > 0.5:
                shear_x = rng.uniform(-30, 30)
                shear_y = rng.uniform(-30, 30)
            else:
                shear_x = rng.uniform(-10, 10)
                shear_y = rng.uniform(-10, 10)
            rot_rng = rng.random()
            if rot_rng < 0.2:
                rotation = rng.uniform(-25, 25)
            elif rot_rng < 0.6:
                rotation = rng.uniform(-80, 80)
            else:
                rotation = rng.uniform(-180, 180)

        if rng.random() > 0.5:
            scale = rng.uniform(0.3, 2)
        else:
            scale = rng.uniform(0.3, 1)

        image = torchvision.transforms.Pad([200, 200, 200, 200], fill=255)(image)
        shear_y, shear_x = 0, 0
        image = affine(
            image,
            rotation,
            translate=[0, 0],
            scale=scale,
            shear=[shear_x, shear_y],
            interpolation=InterpolationMode.BILINEAR,
            fill=255,
        )

        bbox = ImageOps.invert(image).getbbox()
        image = image.crop(bbox)

        height, width = image.height, image.width
        if rng.random() < 0.2:
            h_pad = rng.randint(0, height // 2, (2,), dtype=np.int32)
            w_pad = rng.randint(0, width // 2, (2,), dtype=np.int32)
        else:
            h_pad = rng.randint(0, height * 2, (2,), dtype=np.int32)
            w_pad = rng.randint(0, width * 2, (2,), dtype=np.int32)
        image = torchvision.transforms.Pad([h_pad[0], w_pad[0], h_pad[1], w_pad[1]], fill=255)(
            image
        )

        image = VF.adjust_hue(image, rng.uniform(-0.05, 0.05))
        image = VF.adjust_brightness(image, rng.uniform(0.85, 1.2))
        image = VF.adjust_saturation(image, rng.uniform(0.8, 1.2))
        image = VF.adjust_contrast(image, rng.uniform(0.8, 1.2))

    return {
        "image": np.array(image),
        "prompt": "What time is being shown?",
        "text": text,
        "metadata": {"hour": hour, "second": second, "minute": minute},
        "style": "clocks",
    }
