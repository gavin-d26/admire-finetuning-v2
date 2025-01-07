# This script should train and evaluate CLIP end-to-end on the synthetic dataset.

import os
from functools import partial
import torch
import torch.nn.functional as F
import lightning as pl
from lightning.pytorch.strategies import ModelParallelStrategy
from datasets import load_dataset
from .pixtral import Pixtral12B
from .datatools import _preprocess_samples_for_pixtral, _pixtral_collate_fn
from .configs import pixtral_configs
from ..utils.prompts.sft_finetuning import prompt_template


def train_pixtral(pixtral_configs):

    # set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = pixtral_configs["gpus_devices"]

    # set the seed
    pl.seed_everything(42)

    # Load the dataset
    dataset = load_dataset(pixtral_configs["dataset_url"])

    # preprocess dataset
    print("Preprocessing dataset...")
    # create partial function for preprocess_samples_for_pixtral
    preprocess_samples_for_pixtral_partial = partial(
        _preprocess_samples_for_pixtral,
        prompt_template=prompt_template,
        for_generation=False,
        max_length=pixtral_configs["max_length"],
        padding=pixtral_configs["padding"],
        truncation=pixtral_configs["truncation"],
        image_size=pixtral_configs["image_size"],
    )

    dataset = dataset.map(
        preprocess_samples_for_pixtral_partial,
        batched=True,
        batch_size=pixtral_configs["preprocess_batch_size"],
        num_proc=pixtral_configs["preprocess_num_proc"],
        remove_columns=[
            "sentence",
            "reasoning",
            "output",
            "image_1",
            "image_2",
            "image_3",
            "image_4",
            "image_5",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "pixel_values", "labels"]
    )

    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=pixtral_configs["train_batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=pixtral_configs["num_workers"],
        collate_fn=_pixtral_collate_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=pixtral_configs["test_batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=pixtral_configs["num_workers"],
        collate_fn=_pixtral_collate_fn,
    )

    print("Initializing model...")
    # Initialize the model
    model = Pixtral12B(pixtral_configs)

    logger = (
        pl.pytorch.loggers.WandbLogger(
            project=pixtral_configs["project_name"], name=pixtral_configs["run_name"]
        )
        if pixtral_configs["use_wandb"]
        else None
    )

    if pixtral_configs["strategy"] == "model_parallel":
        strategy = ModelParallelStrategy(
            data_parallel_size=pixtral_configs["data_parallel_size"],
            tensor_parallel_size=pixtral_configs["tensor_parallel_size"],
        )
    else:
        strategy = "auto"

    # Initialize the trainer
    print("Initializing trainer...")
    trainer = pl.Trainer(
        max_epochs=pixtral_configs["max_epochs"],
        devices=pixtral_configs["num_devices"],  # number of GPUs to use
        accelerator=pixtral_configs["accelerator"],
        logger=logger if logger is not None else None,
        enable_checkpointing=True,
        default_root_dir="checkpoints/pixtral",
        # fast_dev_run=True,
        # limit_train_batches=1,
        # limit_test_batches=1,
        log_every_n_steps=1,
        strategy=strategy,
        accumulate_grad_batches=pixtral_configs["accumulate_grad_batches"],
    )

    print("Measuring accuracy on test set, before training...")
    # measure accuracy on test set, before training
    trainer.test(model, test_loader)

    print("Training the model...")
    # Train the model
    trainer.fit(model, train_loader)

    print("Measuring accuracy on test set, after training...")
    # measure accuracy on test set, after training
    trainer.test(model, test_loader)

    # save the peft model
    model.pixtral.language_model.save_pretrained(
        f"checkpoints/pixtral/peft_model/{pixtral_configs['run_name']}/"
    )

    return model
