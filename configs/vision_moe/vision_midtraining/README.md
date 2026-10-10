# Mixed midtraining

Use `src/scripts/train/Mixed-Midtraining.py` with the standard internal experiment CLI
after joint vision alignment:

```bash
python src/scripts/train/Mixed-Midtraining.py launch mixed-midtraining ai2/holmes \
  --recipe.parent_checkpoint=/path/to/alignment-joint/stepN
```

Defaults are 8K context, 128 global sequences, MB2/pack64 and an unfrozen vision encoder.
The 61-source text mixture and eight visual source groups target 90% text / 10% vision
expected supervised-loss weight. Training uses a 50B-position schedule with 200 warmup steps;
no router-input RMS repair is applied. Set `--recipe.text_loss_share=1.0` for text-only
training (T100).

See the [guide](../../../docs/source/guides/mixed_midtraining.md) for execution, sources,
calibration, checkpoint handoffs and standalone evaluation. Short-run validation does not
establish full-budget stability or quality; text and vision benchmarks run separately.
[visual_calibration_v1.json](visual_calibration_v1.json) records the default bounded
visual calibration; changed sources or serialization need matching means.

Default data and output paths require Ai2 infrastructure. Preserve source deployments
required by serialized config/callback classes when resuming existing checkpoints.
