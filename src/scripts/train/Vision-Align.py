"""Run all vision-alignment stages, or select one with ``--recipe.phase``.

See docs/source/guides/vision_alignment.md for checkpoint handoffs and data-source overrides.
"""

from olmo_core.internal.vision_alignment_pipeline import main

if __name__ == "__main__":
    main()
