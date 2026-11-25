# Official public training scripts

Please check the config carefully before attempting to run them. You may need to adjust hyperparameters based on your hardware.

## Usage

Each Python training script in this directory has the same CLI, and they're intended to be launched directly with `torchrun` or, for Beaker users, through Olmo-core Beaker launch CLI: `python -m olmo_core.launch.beaker`.
The scripts themselves take several required arguments as well as any number of config overrides in dot-notation.
Run a script with the `--help` flag to see which arguments are required, and run with the `--dry-run` flag to see the full config that will be used.
To override a field in the config such as the `data_loader`'s `prefetch_factor`, you could add the option `--data_loader.prefetch_factor=4` to your command-line options.

## Loading checkpoints from Hugging Face

Scripts that continue training from a released checkpoint read it from the public
[`allenai/ai2-llm`](https://huggingface.co/buckets/allenai/ai2-llm) Hugging Face storage bucket, e.g.
`https://huggingface.co/buckets/allenai/ai2-llm/resolve/checkpoints/OLMo25/step1413814/`.
The bucket is public, so no credentials are required.

Hugging Face does, however, [rate limit requests](https://huggingface.co/docs/hub/rate-limits), and
anonymous requests get the lowest quota, counted **per IP address** and therefore shared with anyone
else on your network. Loading a distributed checkpoint issues many range requests from every rank at
once, so a large checkpoint may exhaust that quota and fail with `429 Too Many Requests`.

**Without a token,** you are encouraged to download the checkpoint once and train from the local copy instead of streaming
it on every run:

```bash
hf buckets sync hf://buckets/allenai/ai2-llm/checkpoints/OLMo25/step1413814 ./step1413814
```

Then point the script at it with `--trainer.load_path=./step1413814`. See the
[storage bucket docs](https://huggingface.co/docs/hub/storage-buckets) for other ways to download.

**With a token,** requests count against your own account's higher quota instead of the shared
anonymous one. Create a token from your [access token settings](https://huggingface.co/settings/tokens)
and export it as `HF_TOKEN`:

```bash
export HF_TOKEN=hf_...
```

The token must be a **fine-grained** token; which permissions you give it doesn't matter, since the
bucket is public. Unscoped `read` and `write` tokens are rejected with a `403` when reading the
bucket. See [user access tokens](https://huggingface.co/docs/hub/security-tokens) for the difference
between token types.
