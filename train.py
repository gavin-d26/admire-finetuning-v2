import argparse

from models.clip.train import train_clip
from models.clip.configs import clip_configs
from models.pixtral.train import train_pixtral
from models.pixtral.configs import pixtral_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "-model_name",
        choices=["clip", "pixtral"],
        required=True,
        help="Name of the model to train",
    )
    args = parser.parse_args()

    if args.model_name == "clip":
        train_clip(clip_configs)
    elif args.model_name == "pixtral":
        train_pixtral(pixtral_configs)
