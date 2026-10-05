"""Train one vision-alignment phase (``--recipe.phase=bridge|perception|joint``).

See docs/source/guides/vision_alignment.md for checkpoint handoffs and data-source overrides.
"""

from olmo_core.internal.vision_alignment import main

if __name__ == "__main__":
    main()
