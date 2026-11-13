# Official public training scripts

Please check the config carefully before attempting to run them. You may need to adjust hyperparameters based on your hardware.

## Usage

Each Python training script in this directory has the same CLI, and they're intended to be launched directly with `torchrun` or, for Beaker users, through Olmo-core Beaker launch CLI: `python -m olmo_core.launch.beaker`.
The scripts themselves take several required arguments as well as any number of config overrides in dot-notation.
Run a script with the `--help` flag to see which arguments are required, and run with the `--dry-run` flag to see the full config that will be used.
To override a field in the config such as the `data_loader`'s `prefetch_factor`, you could add the option `--data_loader.prefetch_factor=4` to your command-line options.
