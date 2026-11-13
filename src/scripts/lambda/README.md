## Logging in

SSH into one of the login nodes.
Then start or attach your own tmux session:

```bash
USERNAME=petew
tmux new-session -A -s $USERNAME
```

Check that `/data/ai2/bin/` is in your `PATH`, and if not add it:

```bash
export PATH="/data/ai2/bin:$PATH"
```

Export your WANDB token to the `WANDB_API_KEY` env variable, and your Ai2/Beaker username to the `USERNAME` env variable:

```bash
export WANDB_API_KEY=XXXX
export USERNAME=petew
```

Optionally, export a `SLACK_WEBHOOK_URL` for alerting.

```bash
export SLACK_WEBHOOK_URL=XXXX
```

## Initial setup

We all log in as the same user, so we need to be careful to use unique repo directories and virtual environment, which we'll identify by our Ai2/Beaker username (set to the `USERNAME` env var).

```bash
# Clone repo to unique directory.
cd "/data/ai2/$USERNAME"
git clone https://github.com/allenai/Olmo-core.git OLMo-core
cd OLMo-core

# Create unique virtual environment.
uv venv --python=3.12 /data/ai2/uv/OLMo-core-$USERNAME
source "/data/ai2/uv/OLMo-core-${USERNAME}/bin/activate"
uv pip install --torch-backend=cu129 numpy torch==2.8.0 torchvision torchaudio torchao==0.15.0
uv pip install flash-attn --no-build-isolation
uv pip install "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention.git@0abbe028dfb5f033b35eb6da6fc6924accb0dc7a"
uv pip install -e '.[all]'
```

You may also need these version constraints in order for evaluation to work:

```bash
# fix pyarrow issue in evalutor:
uv pip install -U "datasets>=2.20.0"
# fox torchmetrics issue in evaluator:
uv pip install -U "huggingface-hub>=0.34.0,<1.0"
```

## Run a test job

Submit a job through SLURM with:

```bash
./src/scripts/lambda/launch.sh ./src/scripts/lambda/slurm-test-job.sbatch ai2-test-001 1
```

The first argument to the `launch.sh` is the sbatch script to run.
The second is a name to assign to the run.
The third is how many nodes to use.
This will print out the job ID, wait for it to start, and then stream the logs.

## Running your own script

Copy the test sbatch script `slurm-test-job.sbatch` and modifying to your needs.
Then launch it with the `./src/scripts/lambda/launch.sh` script in the same way as we did in the previous section.

## Running a job with retries

For long training jobs that should be automatically resubmitted after failures, use the `launch_with_retries.sh` script
instead of `launch.sh`. It takes the same arguments but will continually resubmit the job until it completes successfully,
and send Slack notifications on failures if the `SLACK_WEBHOOK_URL` environment variable is set.
