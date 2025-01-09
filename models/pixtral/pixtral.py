# This file contains the implementation of Pixtral-12B model using https://huggingface.co/mistral-community/pixtral-12b
import os

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper  # fmt: skip
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from peft import LoraConfig, get_peft_model
from torchao.quantization.quant_api import (
    int8_dynamic_activation_int8_weight,
    quantize_,
)
from torchao.dtypes.nf4tensor import nf4_weight_only
from transformers import AutoModelForImageTextToText, AutoTokenizer

from ..utils.metrics import Perplexity, TopImageAutoRegJson


# Pixtral12B model with training utils
class Pixtral12B(pl.LightningModule):
    def __init__(self, pixtral_configs):
        super().__init__()
        self.save_hyperparameters(pixtral_configs)

        # needed to avoid running self.configure_model() more than once when calling trainer.fit() and trainer.test() in the same process.
        self.model_configured = False

        self.pixtral = AutoModelForImageTextToText.from_pretrained(
            self.hparams["model_url"],
            torch_dtype=torch.float16,
        )

        # Initialize metrics and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_url"], padding_side="left"
        )
        self.train_top_image_accuracy = TopImageAutoRegJson(
            self.tokenizer,
            reasoning_tag_name=self.hparams["reasoning_tag_name"],
            output_json_tag_name=self.hparams["output_json_tag_name"],
            csv_analysis_path=None,
            ignore_token=self.hparams["ignore_token"],
        )
        self.train_perplexity = Perplexity()
        self.csv_analysis_dir_path = self.hparams["csv_analysis_dir_path"]
        os.makedirs(self.csv_analysis_dir_path, exist_ok=True)
        self.csvs_created = 0

        # Apply quantization
        if self.hparams["quantize"]:
            quantize_(self.pixtral, nf4_weight_only())

        # Configure LoRA
        lora_config = LoraConfig(
            target_modules=self.hparams["target_modules"],
            task_type=self.hparams["task_type"],
            r=self.hparams["r"],
            lora_alpha=self.hparams["lora_alpha"],
            # init_lora_weights=True,
        )

        self.pixtral.language_model = get_peft_model(
            self.pixtral.language_model, lora_config
        )

        self.pixtral.language_model.half()

        # Load LoRA parameters if path is provided
        if "lora_path" in self.hparams and self.hparams["lora_path"] is not None:
            print(f"Loading LoRA parameters from {self.hparams['lora_path']}")
            self.pixtral.language_model.load_adapter(self.hparams["lora_path"])

    def configure_model(self):
        if self.model_configured or self.device_mesh is None:
            return

        self.model_configured = True

        # Apply FSDP wrapping with modified config
        fsdp_config = {
            "mesh": self.device_mesh["tensor_parallel"],
        }
        layers = self.pixtral.language_model.model.model.layers

        for layer_id in range(len(layers)):
            # Apply activation checkpointing
            layers[layer_id] = checkpoint_wrapper(layers[layer_id])

            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(layers) - 1
            fully_shard(
                layers[layer_id],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        # Apply FSDP to the whole model
        fully_shard(self, **fsdp_config)

        # Freeze base model parameters
        for param in self.pixtral.parameters():
            param.requires_grad = False

        # unfreeze LoRA parameters
        for name, param in self.pixtral.language_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

    def forward(self, inputs, training=False, **kwargs):
        # Ensure model is in training mode during training
        if training:
            self.pixtral.train()
            # Enable gradients for the forward pass
            with torch.set_grad_enabled(True):
                outputs = self.pixtral(**inputs, **kwargs)
                # Ensure loss has gradients
                if hasattr(outputs, "loss") and not outputs.loss.requires_grad:
                    outputs.loss = outputs.loss.requires_grad_()
                return outputs
        else:
            self.pixtral.eval()
            return self.pixtral(**inputs, **kwargs)

    def on_train_epoch_start(self):
        self.train_top_image_accuracy.reset()
        self.train_perplexity.reset()

    # compute the loss
    def training_step(self, batch, batch_idx):
        outputs = self(batch, training=True)
        loss = outputs.loss
        self.train_top_image_accuracy.update(outputs.logits.detach(), batch["labels"])
        self.train_perplexity.update(outputs.logits.detach(), batch["labels"])
        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(batch, training=False)
        loss = outputs.loss
        self.test_top_image_accuracy.update(outputs.logits.detach(), batch["labels"])
        self.test_perplexity.update(outputs.logits.detach(), batch["labels"])
        return loss

    def on_train_epoch_start(self):
        self.train_top_image_accuracy.reset()
        self.train_perplexity.reset()

    def on_test_epoch_start(self):
        self.test_top_image_accuracy = TopImageAutoRegJson(
            self.tokenizer,
            csv_analysis_path=os.path.join(
                self.csv_analysis_dir_path,
                f"{self.hparams['run_name']}_{self.csvs_created}.csv",
            ),
            reasoning_tag_name=self.hparams["reasoning_tag_name"],
            output_json_tag_name=self.hparams["output_json_tag_name"],
            ignore_token=self.hparams["ignore_token"],
        )
        self.test_perplexity = Perplexity()
        self.csvs_created += 1

    def on_train_batch_end(self):
        # TODO: set the base model gradients to None
        # for name, param in self.pixtral.named_parameters():
        #     if "lora" not in name:
        #         param.grad = None
        pass

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            self.log(
                "train_perplexity",
                self.train_perplexity.compute(),
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "train_top_image_accuracy",
                self.train_top_image_accuracy.compute(),
                rank_zero_only=True,
                sync_dist=True,
            )

    def on_test_epoch_end(self):
        if self.trainer.is_global_zero:
            self.log(
                "test_perplexity",
                self.test_perplexity.compute(),
                rank_zero_only=True,
                sync_dist=True,
            )
            self.log(
                "test_top_image_accuracy",
                self.test_top_image_accuracy.compute(),
                rank_zero_only=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        # TODO: Only optimize parameters that require gradients (LoRA)
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams["learning_rate"],
        )
        return optimizer
