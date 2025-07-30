import os
import yaml
import re
from pathlib import Path
from types import SimpleNamespace

import mlx_lm.lora as lora

"""
Much of this copied over from the mlx_lm.lora module.
"""

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)

CONFIG_DEFAULTS = {
    "model": "mlx_model",
    "train": False,
    "fine_tune_type": "lora",
    "optimizer": "adam",
    "optimizer_config": {
        "adam": {},
        "adamw": {},
    },
    "data": "data/",
    "seed": 0,
    "num_layers": 16,
    "batch_size": 4,
    "iters": 1000,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "adapter_path": "adapters",
    "save_every": 100,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "config": None,
    "grad_checkpoint": False,
    "lr_schedule": None,
    "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
    "mask_prompt": False,
    "wandb": None,
}


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    config = None
    config_path = Path(__file__).parent / "lora_config.yaml"

    print("Loading configuration file", config_path)
    with open(config_path, "r") as file:
        config = yaml.load(file, yaml_loader)

    # Update defaults for unspecified parameters
    for k, v in CONFIG_DEFAULTS.items():
        if config.get(k, None) is None:
            config[k] = v

    lora.run(SimpleNamespace(config))
