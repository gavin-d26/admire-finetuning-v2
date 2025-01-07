# This script should train and evaluate CLIP end-to-end on the synthetic dataset.

import argparse
import os
import torch
import torch.nn.functional as F
import lightning as pl
from datasets import load_dataset
from .clip import CLIP
from .datatools import _preprocess_samples_for_clip
from .configs import clip_configs


def train_clip(clip_configs):

    # set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = clip_configs["gpus_devices"]

    # set the seed
    pl.seed_everything(42)

    # Load the dataset
    dataset = load_dataset(clip_configs["dataset_url"])

    # preprocess dataset
    dataset = dataset.map(
        _preprocess_samples_for_clip,
        batched=True,
        batch_size=30,
        num_proc=clip_configs["num_proc"],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "pixel_values"]
    )

    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=clip_configs["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=clip_configs["num_proc"],
    )

    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=clip_configs["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=clip_configs["num_proc"],
    )

    # Initialize the model
    model = CLIP(clip_configs)

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=clip_configs["max_epochs"],
        devices=1,  # number of GPUs to use
        accelerator=clip_configs["accelerator"],
        #  logger=pl.loggers.TensorBoardLogger("logs/", name="clip"),
        enable_checkpointing=True,
        precision=clip_configs["precision"],
        #  default_root_dir="checkpoints/clip",
        # fast_dev_run=True,
        # limit_train_batches=1,
    )

    trainer.test(model, test_loader)

    # Train the model
    trainer.fit(model, train_loader)

    trainer.test(model, test_loader)

    return model


if __name__ == "__main__":
    train_clip(clip_configs)
