# All-in-one for researchers

This guide is intended for researchers who are new to Olmo-core and would like to experiment with OLMo models or ablate new architectures or data recipes.
We will show you:

- How to launch your first experiment with a small transformer model on Beaker, or locally with `torchrun`.
  This example will be a good foundation to build your own projects on top of.
- How to customize different components of the training loop, such as the model, data loader, optimizer, etc.
- How to scale up to larger models while maintaining high MFU.
- How to troubleshoot common issues.

If you're looking to reproduce an official OLMo training run we still recommend starting with this tutorial to get familiar with the library.
Then you can look in [`src/scripts/official/`](https://github.com/allenai/Olmo-core/tree/main/src/scripts/official) to find public versions of the training scripts,
or [`src/scripts/train/`](https://github.com/allenai/Olmo-core/tree/main/src/scripts/train) for the original Ai2 internal versions (Beaker access required to run them).

```{attention}
If you run into any issues with tutorial, don't hesitate to [open an issue on GitHub](https://github.com/allenai/Olmo-core/issues/new/choose) or reach out on Slack in the [#olmo-core-users](https://allenai.slack.com/archives/C08AU86NMCM) channel.
```

## Setup

### Fork, clone, install

For rapid experimentation we recommend forking Olmo-core for your project instead of installing it as a dependency.
So start by [creating a fork](https://github.com/allenai/Olmo-core/fork) if you haven't already, and then cloning your fork to the computer where you'll be doing the development.

Next you should create or activate a Python virtual environment with a Python version of at least 3.10.
We recommend using [uv](https://docs.astral.sh/uv/) for that, but any other virtual environment system will suffice as well, including conda.

Now once you've `cd`-ed into the root directory of your clone of Olmo-core *and* activated your virtual environment, install [PyTorch](https://pytorch.org) according the directions specific to your operating system and hardware (a CPU-only distribution is fine for local development).
And finally, install Olmo-core in editable mode by running

```
pip install -e '.[all]'
```

or an equivalent `uv` command, such as `uv pip install -e '.[all]'` or `uv sync --all-extras`.

### Beaker

If you'll be using [Beaker](https://beaker.allen.ai) to run experiments you should also [install and configure the Beaker CLI](https://beaker-docs.apps.allenai.org/start/install.html),
which will allow the Olmo-core launch module ({mod}`olmo_core.launch.beaker`) to authenticate with Beaker on your behalf.

It's also a good idea to create a dedicated Beaker workspace at this point for your project.
You can do that by running:

```
beaker workspace create --org=ai2 WORKSPACE_NAME
```

You should then set the budget account for that workspace if you know the one you should be using:

```
beaker workspace set-budget ai2/WORKSPACE_NAME ai2/BUDGET_NAME
```

We also recommend making this new workspace your default for now to avoid accidentally using a different one:

```
beaker config set default_workspace ai2/WORKSPACE_NAME
```

We have official Beaker images for Olmo-core that include all dependencies.
The most up-to-date versions are defined in the {class}`~olmo_core.launch.beaker.OLMoCoreBeakerImage` enum,
and a complete list can be found in the [Olmo-core workspace](https://beaker.allen.ai/orgs/ai2/workspaces/OLMo-core/images).

If you need to build a custom image, see the instructions below.

### Docker

We maintain a [Dockerfile](https://github.com/allenai/Olmo-core/blob/main/src/Dockerfile) for building official images with all of Olmo-core's dependencies.
You can build one yourself by running `make docker-image` from the repository root.
See the [Makefile](https://github.com/allenai/Olmo-core/blob/main/Makefile) for all the different build arguments that you can modify.

## Run your first experiment

We'll start by launching a short language model pretraining run with a small transformer (271M params) on a subset of c4.
This will only take a few minutes on as little as 2 NVIDIA 40GB A100s.

We'll be using the script [`src/examples/llm/train.py`](https://github.com/allenai/Olmo-core/blob/main/src/examples/llm/train.py),
which is intended to be run with `torchrun`, either directly or indirectly through Beaker or something like Slurm.
But before we actually launch the training run, let's look at how the key components/hyperparameters of the run are defined.

### Defining a config

Near the top of the script you'll find a custom config dataclass:

```{literalinclude} ../../../src/examples/llm/train.py
:language: py
:lineno-match:
:start-after: '# docs: start-define-config'
:end-before: '    # docs: end-define-config'
```

*The structure of the config class is arbitrary*, and creating one isn't strictly necessary to use Olmo-core, but it has several benefits:
1. First, it gives us a good way to keep track of all the hyperparameters of each experiment. Since the config inherits from Olmo-core's {class}`~olmo_core.config.Config` baseclass, it comes with useful methods to serialize it to JSON which, for example, could be uploaded to Weights & Biases or saved to the run's checkpoint directory.
2. Second, it gives us a command-line argument parser that maps args directly to fields in the config for free due to the use of the {meth}`Config.merge() <olmo_core.config.Config.merge>` method on this line:

   ```{literalinclude} ../../../src/examples/llm/train.py
   :language: py
   :lineno-match:
   :start-after: '    # docs: start-config-merge'
   :end-before: '    # docs: end-config-merge'
   ```

   This allows us to override fields in the config at runtime via command-line options, even nested fields using dot-notation.
   For example, you could add
   ```
   --data_loader.prefetch_factor=4
   ```
   to the command to update the `prefetch_factor` field within the `data_loader` part of the config.

The components of the config class are then used to construct the model, data loader, trainer, etc, as you can see here:

```{literalinclude} ../../../src/examples/llm/train.py
:language: py
:lineno-match:
:start-after: '    # docs: start-build-components'
:end-before: '    # docs: end-build-components'
```

The script also includes a dry-run mode that you can use to validate that your overrides are applied correctly.
Let's try that right now by passing in the `--dry-run` flag.
Also note the script expects one positional argument, the name of the run (this is used in multiple parts of the config so we require it before the overrides).

Here's the full dry-run command:

```fish
python src/examples/llm/train.py tutorial-run-01 --dry-run
```

And now let's try that again while overriding a few config options:

```fish
python src/examples/llm/train.py tutorial-run-01 --dry-run \
  --data_loader.prefetch_factor=4 \
  --trainer.callbacks.wandb.enabled=true
```

### Launching the run

Now that we know how to change settings on the fly we're ready to launch the run.
And in order to get it running as fast as possible we're going to turn off a few features that we'd normally want on, such as checkpointing and in-loop evals.
We're also going to tell the trainer to stop at step 100.
So the overrides we'll pass in are:

- `--trainer.callbacks.lm_evaluator.enabled=false` to disable the in-loop perplexity evaluator.
- `--trainer.callbacks.downstream_evaluator.enabled=false` to disable the in-loop downstream task evaluator.
- `--trainer.no_checkpoints` to disable checkpointing.
- `--trainer.hard_stop='{value: 100, unit: steps}'` to stop at step 100.

```{tip}
Notice the value we set for `--trainer.hard_stop` is a JSON/YAML mapping. This will get deserialized into a {class}`~olmo_core.train.Duration` instance.
```

#### Launching on Beaker

For Beaker users, you can either use [beaker-gantry](https://github.com/allenai/beaker-gantry) or Olmo-core's own lightweight, gantry-like CLI.
In this case we'll use the latter, which is in the module `olmo_core.launch.beaker`.
Try running
```fish
python -m olmo_core.launch.beaker --help
```
to get familiar with the options.

To actually launch this test run on Beaker, run this command:

```fish
python -m olmo_core.launch.beaker \
  --gpus=2 \
  --weka=oe-training-default \
  --shared-filesystem \
  -- src/examples/llm/train.py \
    tutorial-run-01 \
    --save-folder="/weka/oe-training-default/$USER/tutorial-run-01" \
    --work-dir="/weka/oe-training-default/$USER/dataset-cache" \
    --trainer.callbacks.lm_evaluator.enabled=false \
    --trainer.callbacks.downstream_evaluator.enabled=false \
    --trainer.no_checkpoints \
    --trainer.hard_stop='{value: 100, unit: steps}'
```

If the launch is successful it will print a link to the Beaker workload and then stream the logs to your terminal for the duration of the run.

Some things to note:
- We tell the launch module to request 2 GPUs and mount the weka bucket `oe-training-default` to the container at `/weka/oe-training-default`.
  This allows us to save checkpoints to weka and also gives us access to a copy of the data on weka, which will be much faster to read than streaming over HTTP.
- We also add the flag `--shared-filesystem` since we'll be using weka for the `--save-folder` and `--work-dir`, which tells Olmo-core that each rank has access to the same filesystem that these directories are in.
- Everything after the bare double dashes (`--`) is the command that the launch module will actually run on Beaker (if you've used gantry this should look familiar).
- We set the `--save-folder` and `--work-dir` options to paths on weka, using our username as part of each path to avoid collisions.

#### Launching locally with torchrun

For non-Beaker users, the script can be run directly with `torchrun`. Assuming you have 2 GPUs available, the command would be:

```fish
torchrun --nproc-per-node=2 src/examples/llm/train.py \
  tutorial-run-01 \
  --save-folder=/tmp/tutorial-run-01 \
  --work-dir=/tmp/dataset-cache \
  --trainer.callbacks.lm_evaluator.enabled=false \
  --trainer.callbacks.downstream_evaluator.enabled=false \
  --trainer.no_checkpoints \
  --trainer.hard_stop='{value: 100, unit: steps}'
```

### Changing the model and other components

Now that you've run your first experiment and have a way to test changes, let's look at how to customize different components of the training loop.

```{tip}
If you're going to change any code in the script itself we recommend copying and renaming the it to a new directory of your choice, such as one in `src/scripts/train/`.
```

The first thing you probably want to change is the model.
The script already lets you choose between a number of different preset model configs with the `--model-factory` argument.
The valid choices for this argument come from the various classmethods on the {class}`~olmo_core.nn.transformer.TransformerConfig`.
However, you could also just replace the following lines with a manually constructed `TransformerConfig` instance.
For example, to hard-code in an OLMo2 1B model, replace these lines

```{literalinclude} ../../../src/examples/llm/train.py
:language: py
:lineno-match:
:start-after: '    # docs: start-model-config'
:end-before: '    # docs: end-model-config'
```
with something like this:
```python
    model_config = TransformerConfig.olmo2_1B(
        vocab_size=tokenizer_config.padded_vocab_size()
    )
```

Keep in mind that as you change the model size and architecture you'll need to adjust hyperparameters like the learning rate, and potentially performance settings such as the micro-batch size (`--train_module.rank_microbatch_size`).
See the [scaling](#scaling) section below for more on that.

Similarly the optimizer, dataset, and other components can be changed by modifying their corresponding part of the config.
See the [Going deeper](#going-deeper) section below for more on customization.

### Fine-tuning pretrained models

This script and the {class}`~olmo_core.train.Trainer` in general can be used for fine-tuning just as well as pretraining.
There's just two additional steps you need to take:
1. You need to convert the pretrained weights into a format that the `Trainer` expects.
  See [this HF conversion guide](../examples/huggingface.rst) for an example of converting weights from HuggingFace into the right format.
2. Then you need to tell the `Trainer` to load those weights at the beginning of your run.
  Our example script already does this when the `--load-path` option is set, as you can see here:
  ```{literalinclude} ../../../src/examples/llm/train.py
  :language: py
  :lineno-match:
  :start-after: '    # docs: start-load-path'
  :end-before: '    # docs: end-load-path'
  ```

## Going deeper

The {class}`~olmo_core.train.Trainer` class is a general-purpose trainer in it can be adapted to pretty much any deep learning task by providing a custom {class}`~olmo_core.train.train_module.TrainModule` as the {data}`~olmo_core.train.Trainer.train_module` argument.

### TrainModule (model and optimizer)

A {class}`~olmo_core.train.train_module.TrainModule` abstracts away the model, optimizer, and checkpointing details from the trainer.

The example script we used above made use of the {class}`~olmo_core.train.train_module.TransformerTrainModule` implementation that's designed specifically for training any {class}`olmo_core.nn.transformer.Transformer` type model on text data.
So if that sounds like your use-case, the `TransformerTrainModule` will probably work just fine out-of-the-box for you.
Otherwise you should look at the source code for the {class}`~olmo_core.train.train_module.BasicTrainModule`, as that's a good starting point for writing your own.

### Callbacks

The behavior of the training loop can also be customized through the trainer's rich callback API.
A callback is just a subclass of the base {class}`~olmo_core.train.callbacks.Callback` class, and you can add any number of callbacks to the trainer via the {data}`~olmo_core.train.Trainer.callbacks` argument (a mapping of callback name to callback instance), or by using the {meth}`Trainer.add_callback() <olmo_core.train.Trainer.add_callback>` method.

Olmo-core comes with a number of helpful callbacks that you can find in the {mod}`olmo_core.train.callbacks` module, such as the {class}`~olmo_core.train.callbacks.WandBCallback` for logging training metrics to Weights & Biases.

Several of these callbacks are considered mandatory and are automatically added to the trainer unless you provide them on your own. These include:
- a {class}`~olmo_core.train.callbacks.ConsoleLoggerCallback` for logging progress to the terminal,
- a {class}`~olmo_core.train.callbacks.SpeedMonitorCallback` for recording throughput metrics,
- a {class}`~olmo_core.train.callbacks.GarbageCollectorCallback` for manually managing Python garbage collection, and
- a {class}`~olmo_core.train.callbacks.CheckpointerCallback` for periodically writing checkpoints.

### Data loader

Data loading can be customized by providing a custom {class}`~olmo_core.data.data_loader.DataLoaderBase` implementation as the {data}`~olmo_core.train.Trainer.data_loader` argument to the trainer.
The API supports both mapped- (known length) and iterable-style (unknown length) datasets.
It's the user's responsibility to ensure that the data loader is compatible with distributed training if using more than one GPU.
See the [data loading guide](./data_loading.rst) for more details.

## Scaling

Scaling transformers is a complex topic.
A complete guide on the matter is well beyond the scope of this document, so this section will be focused on settings that can be applied to the {class}`~olmo_core.train.train_module.TransformerTrainModule` specifically, as we anticipate most readers will be using that.

But for more in-depth information on the topic we recommend checking out the [Scaling Book](https://jax-ml.github.io/scaling-book).

At the time of writing, the `TransformerTrainModule` supports 3 dimensions of parallelism for dense models, namely data parallelism through FSDP or DDP, tensor parallelism (TP), and context parallelism (CP), as well as expert parallelism (EP) for MoEs.
There's also experimental support for pipeline parallelism (PP) in the {class}`~olmo_core.train.train_module.TransformerPipelineTrainModule`,
but if you follow these general guidelines you should be able to train up to 70B parameter dense models at a reasonable MFU without pipelining.

### Guidelines

- For models with 1B or more parameters you should use FSDP instead of DDP.
  This can be configured by setting the {class}`dp_config <olmo_core.train.train_module.TransformerDataParallelConfig>` option as follows:
  ```python
  TransformerTrainModule(
      dp_config=TransformerDataParallelConfig(name="fsdp", param_dtype="bfloat16"),
      ...
  )
  ```
  Equivalently you can set the `dp_config` via command-line overrides like this:
  ```
  --train_module.dp_config='{name: fsdp, param_dtype: bfloat16}'
  ```
  Depending on the size of your model, the number of nodes you're training on, and the data center bandwidth, you may also want to try HSDP instead of FSDP:
  ```python
  TransformerTrainModule(
      dp_config=TransformerDataParallelConfig(name="hsdp", param_dtype="bfloat16"),
      ...
  )
  ```
- Don't use TP, CP, or activation checkpointing (AC) unless you get OOMs with a rank micro-batch size of a single instance, and always use the biggest micro-batch size you can fit.
- When you can't fit a single-instance micro-batch, try enabling a minimal amount of activation checkpointing first.
  A good strategy is to set the {class}`ac_config <olmo_core.train.train_module.TransformerActivationCheckpointingConfig>` like this
  ```python
  TransformerTrainModule(
      ac_config=TransformerActivationCheckpointingConfig(
          mode="budget",
          activation_memory_budget=0.90,
      ),
      ...
  )
  ```
  and find the highest budget that will fit without crashing or producing memory allocation warnings.
  Equivalently you can set the `ac_config` via command-line overrides like this:
  ```
  --train_module.ac_config='{mode: budget, activation_memory_budget: 0.90}'
  ```
- Always use `torch.compile` (set `compile_model=True`, or `--train_module.compile_model=true` from the command-line). Not only will this make your model run faster, but it typically reduces peak CUDA memory usage as well.

## Troubleshooting

Sooner or later you're likely to run into issues with your training runs, especially when adding custom components, so here are a few tips to help you troubleshoot them.

### Low-level kernel errors

Due to the [asynchronous execution](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution) of CUDA kernels, the stack traces that are reported at the Python level when a kernel fails are often misleading.
To get a more informative stacktrace you can force synchronous kernel execution by setting the environment variable `CUDA_LAUNCH_BLOCKING=1`.
The `olmo_core.launch.beaker` module we used in the example above comes with a flag (`--debug`) that will automatically set this for you in the Beaker job.

### CUDA OOM errors

When you run into CUDA out-of-memory (OOM) errors, the first thing you should try to do is reduce the training micro-batch size (see the {data}`~olmo_core.train.train_module.TransformerTrainModule.rank_microbatch_size` argument to the {class}`~olmo_core.train.train_module.TransformerTrainModule` for example).
If that's already as small as it can be, consider other options such as activation checkpointing.
See the [scaling](#scaling) section for more ideas.

### Poor throughput or MFU

When you're trying to pinpoint a bottleneck in your training loop, it's a good idea to first look at the time spent loading each batch from your data loader,
which is a metric that's logged by the {class}`~olmo_core.train.callbacks.SpeedMonitorCallback`.

If data loading is not the issue, consider using the {class}`~olmo_core.train.callbacks.ProfilerCallback` to get a trace which can be viewed with [Perfetto UI](https://ui.perfetto.dev/).
And see the [scaling](#scaling) section for more ideas.

### Other bugs

For other bugs unrelated to CUDA, it's always a good idea to try to isolate the code that causes the issue.
If you can reproduce the issue in a small standalone script, it will be much easier to debug and fix.
And if that bug can be reproduced from a single process you could run it with a debugging like [pdb](https://docs.python.org/3/library/pdb.html).

Issues that only manifest in a distributed setting can be harder to debug.
Consider writing a distributed unit test with the {func}`~olmo_core.testing.run_distributed_test` helper function.
