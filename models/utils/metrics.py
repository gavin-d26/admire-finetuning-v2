# contains metrics functions to evaluate the model performance.
# the ADMIRE Task evaluates performance based on
# 1. Top Image Accuracy: Correct identification of the most representative image.
# 2. Rank Correlation: Spearman’s rank correlation of model rankings with ground truth.

# We need implementations of these metrics under multiple modeling scenarios/output formats.

# 1. Image ranking via similarity: The model outputs the ranking of images based on their similarity to the compound text.

# 2. Auto-regressive generation: The model generates text indicative of the image similarity. ie, given 5 images tagged with
# alphabets A, B, C, D, E, the model generates text that ranks the images in the order of their similarity to the compound text.


import json

import pandas as pd
import torch
import torch.nn.functional as F


# Top Image Accuracy via Cosine Similarity (CLIP)
class TopImageAccuracyViaSimilarity:
    def __init__(self):
        self.predictions = None
        self.labels = None

    def update(self, similarity_scores, labels=None):

        if labels is None:
            labels = torch.ones(
                similarity_scores.shape[0], device=similarity_scores.device
            )

        self.predictions = (
            similarity_scores.argmax(dim=-1)
            if self.predictions is None
            else torch.cat((self.predictions, similarity_scores.argmax(dim=-1)), dim=0)
        )
        self.labels = (
            labels if self.labels is None else torch.cat((self.labels, labels), dim=0)
        )

    def compute(self):
        accuracy = (self.predictions == self.labels).float()

        # reset batch
        self.predictions = None
        self.labels = None

        return accuracy.mean().item()

    def reset(self):
        self.predictions = None
        self.labels = None


# Spearman Rank Correlation via Cosine Similarity (CLIP)
class SpearmanRankCorrelationViaSimilarity:
    def __init__(self):
        self.predictions = None
        self.labels = None

    def update(self, similarity_scores, labels=None):
        """
        Update the predictions and labels.

        Args:
            similarity_scores (torch.Tensor): A tensor of shape (B, num_images) containing the similarity scores of each image.
            labels (torch.Tensor): A tensor of shape (B,) containing the ground truth label of the most representative image.
        """
        if labels is None:
            # assumes the images are supplied to the model in decreasing order of similarity to compound text
            labels = (
                torch.arange(similarity_scores.size(1), device=similarity_scores.device)
                .unsqueeze(0)
                .tile(similarity_scores.size(0), 1)
            )

        # Get the ranks of the similarity scores
        ranks = similarity_scores.argsort(dim=1, descending=True)

        # assumming labels are ranked as 0, 1, 2, 3, 4 -> most similar to least similar
        labels = (
            labels.argsort(dim=1, descending=False)
            if labels.dim() != 1
            else labels.unsqueeze(0)
            .argsort(dim=1, descending=False)
            .tile(similarity_scores.size(0), 1)
        )

        self.predictions = (
            ranks
            if self.predictions is None
            else torch.cat((self.predictions, ranks), dim=0)
        )
        self.labels = (
            labels if self.labels is None else torch.cat((self.labels, labels), dim=0)
        )

    def compute(self):
        """
        Compute the Spearman rank correlation.
        """

        # Compute the Spearman rank correlation
        diff_square = (self.predictions - self.labels) ** 2

        _, n = self.predictions.shape

        spearman_coefficient = 1 - ((6 * diff_square.sum(dim=-1)) / (n * (n**2 - 1)))

        # reset batch
        self.predictions = None
        self.labels = None

        return spearman_coefficient.mean().item()

    def reset(self):
        self.predictions = None
        self.labels = None


# class to compute  Perplexity
class Perplexity:
    def __init__(self, padding_value=-100):
        self.log_probs = None
        self.padding_value = padding_value

    def update(self, logits, targets):
        # logits.shape = (batch_size, block_size, num_classes)
        # targets.shape = (batch_size, block_size)

        # move logits and targets to CPU
        logits = logits.cpu()
        targets = targets.cpu()

        N, S, C = logits.shape

        # Create a mask for padding values
        mask = targets == self.padding_value

        # Create a valid targets tensor where padding values are replaced with 0
        # (0 is a safe index for gather, these values are masked later)
        valid_targets = targets.masked_fill(mask, 0)

        # Add small epsilon to prevent log(0)
        probs = F.softmax(logits, dim=-1).clamp(min=1e-10)
        probs = probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(
            -1
        )  # (N, S)

        # Apply the mask to set probabilities to 1 for padding tokens
        probs = probs.masked_fill(mask, 1)

        # Calculate log probabilities and average over sequence length (excluding padding)
        log_probs = torch.log(probs).sum(dim=-1) / torch.logical_not(mask).sum(dim=-1)

        self.log_probs = (
            torch.cat([self.log_probs, log_probs], dim=0)
            if self.log_probs is not None
            else log_probs
        )

    # used to compute Perplexity at the end of an epoch
    def compute(self):
        perplexity = torch.exp(-self.log_probs.mean())
        self.reset()
        return perplexity.item()

    def reset(self):
        self.log_probs = None


class TopImageAutoRegJson:
    def __init__(
        self,
        tokenizer,
        csv_analysis_path=None,
        reasoning_tag_name="reasoning",
        output_json_tag_name="output",
        ignore_token="</s>",
        padding_value=-100,
    ):
        """
        Initialize the TopImageAutoRegJson class.

        Args:
            tokenizer (Tokenizer): The tokenizer to use for decoding the predictions and targets.
            csv_analysis_path (str, optional): The path to save the analysis to a CSV file. Defaults to None.
            reasoning_tag_name (str, optional): The name of the tag for the reasoning section. Defaults to "reasoning".
            output_json_tag_name (str, optional): The name of the tag for the output JSON section. Defaults to "output".
            ignore_token (str, optional): The token to ignore in the predictions and targets. Defaults to "</s>".
        """

        self.reasoning_tag_name = reasoning_tag_name
        self.output_json_tag_name = output_json_tag_name
        self.tokenizer = tokenizer
        self.predictions = None
        self.labels = None
        self.csv_analysis_path = csv_analysis_path  # path to save the analysis to a CSV
        self.ignore_token = ignore_token  # usually padding token, avoid printing in the CSV analysis file
        self.padding_value = padding_value  # integer value of the padding token

    def update(self, logits, targets):
        """
        Update the predictions and labels.

        Args:
            logits (torch.Tensor): A tensor of shape Union[(B, seq_len_logits, vocab_size), (B, seq_len_logits)] containing the logits of each image.\
                If using the dataset for generation, the logits are of shape (B, seq_len_logits) using an arbitrary decoding strategy.
                
            targets (torch.Tensor): A tensor of shape (B, seq_len_targets) containing the ground truth label of the most representative image.
        """

        # move logits and targets to CPU
        logits = logits.cpu()
        targets = targets.cpu()

        # get the predicted labels
        if logits.dim() == 3:
            # used during training to measure top image accuracy.
            predictions = logits.argmax(dim=-1)
        else:
            # for generation, we need to decode the logits to get the reasoning and output before passing to this method.
            predictions = logits

        predictions = self.tokenizer.batch_decode(predictions)

        # replace padding tokens with ignore token
        targets = targets.masked_fill(
            targets == self.padding_value,
            self.tokenizer.convert_tokens_to_ids(self.ignore_token),
        )
        # get the ground truth labels
        labels = self.tokenizer.batch_decode(targets)

        self.predictions = (
            predictions if self.predictions is None else self.predictions + predictions
        )
        self.labels = labels if self.labels is None else self.labels + labels

    def compute(self):
        """
        Compute the Top Image Accuracy.

        predictions/targets format -> <output>{some JSON struct}</output>
        """

        pred_reasoning_str_list = []
        pred_output_str_list = []
        full_prediction_str_list = []
        pred_json_list = []
        pred_image_list = []

        for prediction in self.predictions:
            full_prediction_str = prediction.replace(self.ignore_token, "").strip()
            try:
                pred_reasoning_str = (
                    prediction.split(f"<{self.reasoning_tag_name}>")[1]
                    .split(f"</{self.reasoning_tag_name}>")[0]
                    .strip()
                )
            except:
                pred_reasoning_str = ""

            try:
                pred_output_str = (
                    prediction.split(f"<{self.output_json_tag_name}>")[1]
                    .split(f"</{self.output_json_tag_name}>")[0]
                    .strip()
                )
                pred_json = {}  # Initialize pred_json before try block
                try:
                    pred_json = json.loads(pred_output_str)
                    if "Image" in pred_json:
                        # Convert string to int if needed
                        pred_image = pred_json["Image"]
                        if isinstance(pred_image, str):
                            try:
                                pred_image = int(pred_image)
                            except ValueError:
                                pred_image = -1
                        pred_image_list.append(pred_image)
                    else:
                        pred_image_list.append(-1)
                except:
                    pred_image_list.append(-1)
            except:
                pred_output_str = ""
                pred_json = {}  # Initialize pred_json in the outer except block
                pred_image_list.append(-1)

            pred_reasoning_str_list.append(pred_reasoning_str)
            pred_output_str_list.append(pred_output_str)
            full_prediction_str_list.append(full_prediction_str)
            pred_json_list.append(pred_json)

        label_reasoning_str_list = []
        label_output_str_list = []
        full_label_str_list = []
        label_json_list = []
        label_image_list = []

        for label in self.labels:
            full_label_str = label.replace(self.ignore_token, "").strip()
            try:
                label_reasoning_str = (
                    label.split(f"<{self.reasoning_tag_name}>")[1]
                    .split(f"</{self.reasoning_tag_name}>")[0]
                    .strip()
                )
            except:
                label_reasoning_str = ""

            try:
                label_output_str = (
                    label.split(f"<{self.output_json_tag_name}>")[1]
                    .split(f"</{self.output_json_tag_name}>")[0]
                    .strip()
                )
                label_json = json.loads(label_output_str)
                if "Image" in label_json:
                    # Convert string to int if needed
                    label_image = label_json["Image"]
                    if isinstance(label_image, str):
                        try:
                            label_image = int(label_image)
                        except ValueError:
                            label_image = -1
                    label_image_list.append(label_image)
                else:
                    label_image_list.append(-1)
            except:
                label_output_str = ""
                label_image_list.append(-1)

            label_reasoning_str_list.append(label_reasoning_str)
            label_output_str_list.append(label_output_str)
            full_label_str_list.append(full_label_str)
            label_json_list.append(label_json)

        # compute the Top Image Accuracy
        preds = torch.tensor(pred_image_list)
        labels = torch.tensor(label_image_list)

        # Create mask for valid cases (excluding where both pred and label are -1)
        valid_mask = ~((preds == -1) & (labels == -1))

        # Calculate accuracy only on valid cases
        if valid_mask.sum().item() > 0:
            top_image_accuracy = (
                preds[valid_mask] == labels[valid_mask]
            ).sum().item() / valid_mask.sum().item()
        else:
            top_image_accuracy = 0.0  # Return 0 if no valid cases exist

        # save the analysis to a CSV file
        if self.csv_analysis_path:

            df = pd.DataFrame(
                {
                    "full_prediction": full_prediction_str_list,
                    "full_label": full_label_str_list,
                    "pred_reasoning": pred_reasoning_str_list,
                    "label_reasoning": label_reasoning_str_list,
                    "pred_output": pred_output_str_list,
                    "label_output": label_output_str_list,
                    "pred_json": pred_json_list,
                    "label_json": label_json_list,
                    "pred_image": pred_image_list,
                    "label_image": label_image_list,
                }
            )
            df.to_csv(self.csv_analysis_path)

        return top_image_accuracy

    def reset(self):
        self.predictions = None
        self.labels = None
