"""AdditiveLeakySFT a07; controlled contradiction comparison in _variant_common.py."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _variant_common import build_variant_experiment
from olmo_core.internal.experiment import main

def build_experiment_config(context):
    return build_variant_experiment(context,arm='a07',variant='additive')

if __name__=='__main__':main(config_builder=build_experiment_config)
