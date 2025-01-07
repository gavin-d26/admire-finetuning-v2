import os
import torch
import lightning as pl
from lightning.pytorch.strategies import ModelParallelStrategy
from .pixtral import Pixtral12B
from .configs import pixtral_configs


def unshard_and_save_lora(checkpoint_path: str, output_dir: str):
    """
    Unshard a distributed checkpoint and save the LoRA weights.

    Args:
        checkpoint_path: Path to the sharded checkpoint directory
        output_dir: Directory to save the unsharded LoRA weights
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    print(f"Will save LoRA weights to {output_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model with same config used during training
    model = Pixtral12B(pixtral_configs)

    # Convert distributed checkpoint to single checkpoint
    print("Converting distributed checkpoint to single checkpoint...")
    checkpoint = pl.Trainer.from_checkpoint(
        checkpoint_path=checkpoint_path, map_location="cpu"  # Load on CPU to avoid OOM
    )

    # Load the weights into the model
    print("Loading weights into model...")
    model.load_state_dict(checkpoint.state_dict())

    # Save only the LoRA weights
    print("Saving LoRA weights...")
    model.pixtral.language_model.save_pretrained(output_dir)

    print(f"Successfully saved LoRA weights to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Unshard checkpoint and save LoRA weights"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the sharded checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the unsharded LoRA weights",
    )

    args = parser.parse_args()
    unshard_and_save_lora(args.checkpoint_path, args.output_dir)
