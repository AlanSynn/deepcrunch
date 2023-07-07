import argparse
import os
from deepcrunch.core import ModelWrapper, TrainerWrapper
from deepcrunch.utilities import set_seed

###############################################
#
# ENVIRONMENT VARIABLES
#
###############################################

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NCCL_P2P_LEVEL"] = "NVL"

###############################################
#
# PARSING ARGUMENTS
#
###############################################

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="deepcrunch")

    parser.add_argument(
        "--experiment",
        type=str,
        default="deepcrunch",
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
    )

    args = parser.parse_args(args=args)
    args.experiment_name = f"{args.experiment}-{args.dataset}"

    return args

def default_config(args=None):
    args.accelerator = "cpu"
    args.world_size = 1
    logger = None
    return logger, args

###############################################
#
# MAIN
#
###############################################

def main():
    set_seed()

    args = parse_args()
    print(
        "----------------------------------------------------------------------------------------------------",
        "\n",
        args,
        "\n",
        "----------------------------------------------------------------------------------------------------",
    )

    if args.debug:
        args, logger = default_config(args)
    else:
        wandb.init(project=args.experiment_name)

    wandb.config.update(args)
    logger = SafeWandbLogger()

    model = ModelWrapper(model)
    trainer = TrainerWrapper(args, logger)
    trainer.fit(model)

if __name__ == "__main__":
    main()
