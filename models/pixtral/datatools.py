# This script contains tools to preprocess the UCSC Admire dataset for the Pixtral-12B model.

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from .configs import pixtral_configs
from ..utils.prompts import prompt_template

"""sample = dataset['train'][0]
    {'id': 0,
 'language': 'en',
 'compound': 'Cross that bridge when you come to it',
 'sentence_type': 'literal',
 'sentence': 'The hikers had to cross that bridge when they came to it, carefully making their way over the rickety wooden planks suspended high above the raging river.',
 'style': '<NONE>',
 'image_1_prompt': "A close-up of a person's feet standing on a wooden bridge, with the bridge's planks and ropes visible in the foreground. The background shows a fast-moving river, with water splashing against the bridge's supports. The person's feet are steady, but the bridge creaks slightly under their weight.",
 'image_1': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=1024x1024>,
 'image_2_prompt': 'A scenic landscape of a rural area, with a small bridge crossing over a tranquil stream. The bridge is old and worn, with moss growing on its stones. The surrounding trees are lush and green, and a few birds can be seen flying overhead. The atmosphere is peaceful and serene.',
 'image_2': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=1024x1024>,
 'image_3_prompt': 'A person sitting at a desk, surrounded by papers and files, with a worried expression on their face. A clock on the wall shows a deadline looming, and a calendar in the background is marked with important dates. The atmosphere is tense and anxious, with a sense of uncertainty.',
 'image_3': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=1024x1024>,
 'image_4_prompt': "A person standing at a fork in the road, looking ahead at a distant horizon. The atmosphere is calm and contemplative, with a few wispy clouds in the sky. The person's facial expression is thoughtful, as if weighing their options and considering the path ahead.",
 'image_4': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=1024x1024>,
 'image_5_prompt': 'A person riding a unicycle across a tightrope suspended high above a city street. The background shows a bustling metropolis, with people walking and cars driving by. The person on the unicycle is focused and concentrated, with a sense of determination and skill.',
 'image_5': <PIL.WebPImagePlugin.WebPImageFile image mode=RGB size=1024x1024>} 
 'reasoning': '<reasoning> Some text </reasoning>'}
 'output': '<output>{ Some JSON }</output>'}
 """

model_url = pixtral_configs["model_url"]

# padding is done on the left side
processor = AutoProcessor.from_pretrained(
    pixtral_configs["model_url"], padding_side="left"
)


# function to format the prompt template for pixtral. adds [IMG] tags into the template.
def format_prompt_template(prompt_template, sentence, reasoning, output):
    output = (
        output[len("<output>\n\n") :] if output.startswith("<output>\n\n") else output
    )
    user_content = (
        prompt_template.replace("{{SENTENCE}}", sentence)
        .replace("<image_1>{{IMAGE_1_DESCRIPTION}}</image_1>", "[IMG]")
        .replace("<image_2>{{IMAGE_2_DESCRIPTION}}</image_2>", "[IMG]")
        .replace("<image_3>{{IMAGE_3_DESCRIPTION}}</image_3>", "[IMG]")
        .replace("<image_4>{{IMAGE_4_DESCRIPTION}}</image_4>", "[IMG]")
        .replace("<image_5>{{IMAGE_5_DESCRIPTION}}</image_5>", "[IMG]")
    )
    assistant_content = reasoning + "\n\n" + output
    return user_content, assistant_content


# NOTE: HF Datasets produce a dict of lists for batched data, so we need to preprocess the samples keeping this in mind.
# NOTE: Pass this function to a Lambda function in the map method of the dataset object.
def _preprocess_samples_for_pixtral(
    example,
    prompt_template,
    for_generation=False,
    max_length=2048,
    padding="max_length",
    truncation=True,
    image_size=224,
):
    # Preprocess the text
    sentence_list = example["sentence"]
    reasoning_list = example["reasoning"]
    output_list = example["output"]
    image1_list = example["image_1"]
    image2_list = example["image_2"]
    image3_list = example["image_3"]
    image4_list = example["image_4"]
    image5_list = example["image_5"]

    batch_prompt_list = []
    batch_images_list = []
    labels_list = []

    # process each sample in the batch
    for i in range(len(sentence_list)):
        sentence = sentence_list[i]
        reasoning = reasoning_list[i]
        output = output_list[i]
        image1 = image1_list[i].resize((image_size, image_size))
        image2 = image2_list[i].resize((image_size, image_size))
        image3 = image3_list[i].resize((image_size, image_size))
        image4 = image4_list[i].resize((image_size, image_size))
        image5 = image5_list[i].resize((image_size, image_size))

        # format the prompt template
        user_content, assistant_content = format_prompt_template(
            prompt_template, sentence, reasoning, output
        )

        # create messages list
        messages = [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": ("<reasoning>" if for_generation else assistant_content),
            },
        ]

        # if for_generation, we don't need the entire input sentence, just the reasoning and output.
        if for_generation:
            labels_list.append(assistant_content)

        prompt = processor.apply_chat_template(
            messages,
            continue_final_message=for_generation,
        )

        batch_prompt_list.append(prompt)
        batch_images_list.append([image1, image2, image3, image4, image5])

    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    input_dict = processor(
        text=batch_prompt_list,
        images=batch_images_list,
        return_tensors="pt",
        add_special_tokens=False,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )

    if for_generation:
        # TODO: if using the dataset for generation, do we need to pad the labels (used for evaluation/metrics)?
        # yes, since we're using batched decoding.
        labels = processor(
            text=labels_list,
            return_tensors="pt",
            add_special_tokens=False,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )["input_ids"]
        input_dict["partial_labels"] = labels
        return input_dict

    # create labels (input_ids[1:]), replace padding tokens with -100, mask instruction tokens
    input_dict["labels"] = input_dict["input_ids"].clone()
    input_dict["labels"][input_dict["attention_mask"] == 0] = -100

    # use 1: tokens in labels and :-1 in input_ids
    input_dict["labels"] = input_dict["labels"][:, 1:]
    input_dict["input_ids"] = input_dict["input_ids"][:, :-1]
    input_dict["attention_mask"] = input_dict["attention_mask"][:, :-1]

    # SFT only on model generations.
    # find the instruction end token and set all tokens before it to -100 in labels.
    instruction_end_token_id = processor.tokenizer.convert_tokens_to_ids("[/INST]")
    instruction_end_idx = (input_dict["labels"] == instruction_end_token_id).nonzero()[:, 1]  # fmt: skip
    for i, idx in enumerate(instruction_end_idx):
        input_dict["labels"][i, : idx + 1] = -100

    return input_dict


def _pixtral_collate_fn(batch):
    # Convert pixel_values tensor to list of lists using unbind
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    pixel_values = [
        tensor for item in batch for tensor in item["pixel_values"].unbind(0)
    ]
    labels = torch.stack([item["labels"] for item in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": labels,
    }


# if __name__ == "__main__":
#     from tqdm import tqdm

#     dataset = load_dataset("UCSC-Admire/idiom-SFT-dataset-561-2024-12-06_00-40-30")

#     # preprocess each sample in the dataset and print the max length
#     max_length = 0
#     for i in tqdm(range(len(dataset["test"]))):
#         sample = dataset["test"][i : i + 1]
#         input_dict = _preprocess_samples_for_pixtral(
#             sample,
#             prompt_template,
#             for_generation=False,
#             max_length=None,
#             padding="do_not_pad",
#             truncation=False,
#             image_size=224,
#         )
#         max_length = max(max_length, input_dict["input_ids"].shape[1])
#     print(max_length)
