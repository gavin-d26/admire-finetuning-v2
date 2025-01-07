# This file contains the implementation of CLIP model using Huggingface's CLIPTextModelWithProjection and CLIPVisionModelWithProjection

import torch
import torch.nn.functional as F
from transformers import (
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    get_scheduler,
)
import lightning as pl
from ..utils.metrics import (
    TopImageAccuracyViaSimilarity,
    SpearmanRankCorrelationViaSimilarity,
)


# CLIP model with training utils
class CLIP(pl.LightningModule):
    def __init__(self, clip_configs):
        super().__init__()
        self.clip_configs = clip_configs
        self.clip_text = CLIPTextModelWithProjection.from_pretrained(
            clip_configs["model_url"]
        )
        self.clip_vision = CLIPVisionModelWithProjection.from_pretrained(
            clip_configs["model_url"]
        )
        if "similarity_weights" in clip_configs:
            self.register_buffer(
                "similarity_weights", torch.tensor(clip_configs["similarity_weights"])
            )

        # hyperparameters
        self.save_hyperparameters(clip_configs)
        self.learning_rate = clip_configs["learning_rate"]
        self.train_top_image_accuracy = TopImageAccuracyViaSimilarity()
        self.train_spearman_rank_correlation = SpearmanRankCorrelationViaSimilarity()
        self.test_top_image_accuracy = TopImageAccuracyViaSimilarity()
        self.test_spearman_rank_correlation = SpearmanRankCorrelationViaSimilarity()

        # self.clip_text.text_model.requires_grad_(False)
        # self.clip_vision.vision_model.requires_grad_(False)

    # computes the similarity scores between the text and the images
    def compute_similarity_scores(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        # reshape pixel_values to (B, 3, 224, 224)
        B = pixel_values.shape[0]
        pixel_values = pixel_values.reshape(-1, 3, 224, 224)

        # compute text embeddings
        text_vectors = self.clip_text(
            input_ids=input_ids, attention_mask=attention_mask
        ).text_embeds

        # compute vision embeddings and reshape to (B, num_images, 512)
        vision_vectors = self.clip_vision(pixel_values=pixel_values).image_embeds.view(
            B, -1, 512
        )

        # cosine similarity
        similarity = torch.matmul(
            text_vectors.unsqueeze(1), vision_vectors.transpose(-1, -2)
        ).squeeze(1) / (
            text_vectors.norm(dim=-1).unsqueeze(-1) * vision_vectors.norm(dim=-1)
        )
        return similarity

    def on_train_epoch_start(self):
        self.train_top_image_accuracy.reset()
        self.train_spearman_rank_correlation.reset()

    # compute the loss
    def training_step(self, batch, batch_idx):
        similarity = self.compute_similarity_scores(batch)
        loss = self.pair_wise_distance_loss(similarity[:, :-1], similarity[:, 1:])
        self.train_top_image_accuracy.update(similarity.detach())
        self.train_spearman_rank_correlation.update(similarity.detach())
        self.log(
            "my_loss", loss.detach(), on_epoch=True, prog_bar=True, logger=True
        )  # on_epoch=True indicates that the loss will be logged (accumulated) at the end of the epoch
        return loss

    def on_train_epoch_end(self):
        # apparently lightning automatically determines the frequency of logging (batch/epoch) based on the method name
        self.log("train_top_image_accuracy", self.train_top_image_accuracy.compute())
        self.log(
            "train_spearman_rank_correlation",
            self.train_spearman_rank_correlation.compute(),
        )

    def on_test_epoch_start(self):
        self.test_top_image_accuracy.reset()
        self.test_spearman_rank_correlation.reset()

    def test_step(self, batch, batch_idx):
        similarity = self.compute_similarity_scores(batch)
        loss = self.pair_wise_distance_loss(similarity[:, :-1], similarity[:, 1:])
        self.test_top_image_accuracy.update(similarity.detach())
        self.test_spearman_rank_correlation.update(similarity.detach())
        return loss

    def on_test_epoch_end(self):
        # apparently lightning automatically determines the frequency of logging (batch/epoch) based on the method name
        self.log("test_top_image_accuracy", self.test_top_image_accuracy.compute())
        self.log(
            "test_spearman_rank_correlation",
            self.test_spearman_rank_correlation.compute(),
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    # compute pair-wise distance loss
    def pair_wise_distance_loss(self, score_n, score_n_minus_1):
        if hasattr(self, "similarity_weights"):
            return (
                -F.logsigmoid(score_n - score_n_minus_1) * self.similarity_weights
            ).mean()
        return -F.logsigmoid(score_n - score_n_minus_1).mean()


# if __name__=="__main__":
#     model = CLIP('openai/clip-vit-base-patch16')

#     #dummy inputs
#     pixel_values = torch.randn(2, 5, 3, 224, 224)

#     # shape = (batch_size, seq_length)
#     input_ids = torch.randint(0, 1000, (2, 64))
#     attention_mask = torch.randint(0, 2, (2, 64))

#     model.rerank({
#         'input_ids': input_ids,
#         'attention_mask': attention_mask,
#         'pixel_values': pixel_values
#     })
