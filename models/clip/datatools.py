# This script contains tools to preprocess the UCSC Admire dataset for the CLIP model.

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPTokenizerFast
from .configs import clip_configs


# Load the dataset
# dataset = load_dataset("UCSC-Admire/idiom-dataset-100-2024-11-11_14-37-58")

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
 """

model_url = clip_configs["model_url"]

img_processor = CLIPImageProcessor.from_pretrained(model_url)
text_processor = CLIPTokenizerFast.from_pretrained(model_url)


# NOTE: HF Datasets produce a dict of lists for batched data, so we need to preprocess the samples keeping this in mind.
def _preprocess_samples_for_clip(example):
    # Preprocess the text
    text = example["sentence"]
    output = text_processor(
        text, return_tensors="pt", padding="max_length", max_length=64, truncation=True
    )

    pixel_values = []

    # Preprocess the images
    for i in range(1, 6):
        image = example[f"image_{i}"]
        image_input = img_processor(image, return_tensors="pt")
        pixel_values.append(image_input["pixel_values"].unsqueeze(0))

    output.update({"pixel_values": torch.cat(pixel_values, dim=0).transpose(0, 1)})

    return output


# dataset = dataset.map(_preprocess_sample, batched=True, batch_size=30, num_proc=2)


# print(dataset['train'][0])
