import anthropic
import datasets
import random
import json
import os
import pickle
from tqdm import tqdm


DATASET_URL = None  # dataset url in HF
data_split = "train"
SAVE_FREQ = 10  # save every 10 records
MODEL = None  # model name in anthropic

ONLY_FAILED = True
failed_ids_path = os.path.join("sft_data", data_split, f"failed_ids_{data_split}.json")

data_save_path = (
    os.path.join("sft_data", data_split)
    if not ONLY_FAILED
    else os.path.join("sft_data", data_split, "failed")
)


def make_sft_data_for_record(item, client):
    try:
        SENTENCE = item["sentence"]
        COMPOUND_PHRASE = item["compound"]
        COMPOUND_TYPE = item["sentence_type"]

        images = [
            (item["image_1_prompt"], item["image_1"], 1),
            (item["image_2_prompt"], item["image_2"], 2),
            (item["image_3_prompt"], item["image_3"], 3),
            (item["image_4_prompt"], item["image_4"], 4),
            (item["image_5_prompt"], item["image_5"], 5),
        ]

        random.shuffle(images)

        IMAGE_1_DESCRIPTION, IMAGE_1, IMAGE_1_INDEX = images[0]
        IMAGE_2_DESCRIPTION, IMAGE_2, IMAGE_2_INDEX = images[1]
        IMAGE_3_DESCRIPTION, IMAGE_3, IMAGE_3_INDEX = images[2]
        IMAGE_4_DESCRIPTION, IMAGE_4, IMAGE_4_INDEX = images[3]
        IMAGE_5_DESCRIPTION, IMAGE_5, IMAGE_5_INDEX = images[4]

        images_indexed = [
            IMAGE_1_INDEX,
            IMAGE_2_INDEX,
            IMAGE_3_INDEX,
            IMAGE_4_INDEX,
            IMAGE_5_INDEX,
        ]
        CORRECT_IMAGE = images_indexed.index(1) + 1

        message = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": 'You are an AI assistant tasked with generating Supervised Fine-Tuning (SFT) data for a vision-language model. The data will be used to train the model to analyze sentences containing compound phrases and match them to relevant images. Your goal is to create high-quality, diverse examples that mimic human-like reasoning and decision-making processes.\n\nHere is the sentence and descriptions of five images:\n\n<sentence>\n{{SENTENCE}}\n</sentence>\n\n<image_descriptions>\n<IMAGE_1_DESCRIPTION>{{IMAGE_1_DESCRIPTION}}</IMAGE_1_DESCRIPTION>\n<IMAGE_2_DESCRIPTION>{{IMAGE_2_DESCRIPTION}}</IMAGE_2_DESCRIPTION>\n<IMAGE_3_DESCRIPTION>{{IMAGE_3_DESCRIPTION}}</IMAGE_3_DESCRIPTION>\n<IMAGE_4_DESCRIPTION>{{IMAGE_4_DESCRIPTION}}</IMAGE_4_DESCRIPTION>\n<IMAGE_5_DESCRIPTION>{{IMAGE_5_DESCRIPTION}}</IMAGE_5_DESCRIPTION>\n</image_descriptions>\n\nFor this specific example:\n- The compound phrase used {{COMPOUND_PHRASE}}.\n- The compund type is  {{COMPOUND_TYPE}} (idiomatic or literal).\n- The image that best matches the meaning of the compound is Image {{CORRECT_IMAGE}}.\n\nYour task is to generate an analysis that demonstrates the thought process and reasoning a vision-language model should follow when interpreting the sentence and selecting the most relevant image. The output should be in the following JSON format:\n\n<output>\n{\n  "Compound Type": "[Idiomatic/Literal] - [Brief explanation of why]",\n  "Compound": "[Identified compound phrase]",\n  "Compound Meaning": "[Explanation of the compound\'s meaning in the context of the sentence]",\n  "Reasoning": "[Explanation of why the selected image represents the compound\'s meaning in the context of the sentence the closest]"\n  "Alternative Interpretations": "[If applicable, briefly mention any other possible interpretations]"\n  "Image": "[Index Number of the image that matches the compound\'s meaning]",\n  "Confidence": "[High/Medium/Low] - [Brief explanation of your confidence level]"\n}\n</output>\n\nTo generate high-quality SFT data:\n\n1. Identify the compound phrase in the sentence and explain why it\'s {{COMPOUND_TYPE}}.\n2. Provide a clear and concise explanation of the compound\'s meaning in the context of the sentence.\n3. Analyze the image descriptions and explain why Image {{CORRECT_IMAGE}} best matches the compound\'s meaning. Include specific details from the image description that support this choice.\n4. Assume that the images have the Image Index Number in the top left corner as that is what the data will look like to the VLM.\n5. If applicable, mention alternative interpretations or images that could be considered, but explain why they are less suitable than the chosen image.\n6. Assign a confidence level (High/Medium/Low) based on how well the chosen image matches the compound\'s meaning, and briefly explain this rating.\n\nBefore providing the final JSON output, use <reasoning> tags to explain your thought process for each step of the analysis. This should demonstrate the kind of reasoning the vision-language model should learn to perform.\n\nAdditional guidelines:\n- Ensure your analysis is thorough and logical, considering multiple aspects of the sentence and images.\n- Use natural language that mimics human-like reasoning and decision-making.\n- If the compound type is idiomatic, explain both the literal and figurative meanings to show understanding of the idiom.\n- If the compound type is literal, explain why it\'s not being used idiomatically in this context.\n- Consider cultural context and potential multiple meanings when analyzing the compound and images.\n- If none of the images seem to match the compound\'s meaning perfectly, explain the limitations and why the chosen image is still the best match.\n\nRemember, the goal is to create diverse, high-quality examples that will help train the vision-language model to perform this task effectively. Your generated data should demonstrate the kind of nuanced analysis and reasoning the model should learn to emulate.\n\nBegin your analysis now, starting with your reasoning process in <reasoning> tags, followed by the JSON output in <output> tags.'.replace(
                                "{{SENTENCE}}", SENTENCE
                            )
                            .replace(
                                "{{IMAGE_1_DESCRIPTION}}", str(IMAGE_1_DESCRIPTION)
                            )
                            .replace(
                                "{{IMAGE_2_DESCRIPTION}}", str(IMAGE_2_DESCRIPTION)
                            )
                            .replace(
                                "{{IMAGE_3_DESCRIPTION}}", str(IMAGE_3_DESCRIPTION)
                            )
                            .replace(
                                "{{IMAGE_4_DESCRIPTION}}", str(IMAGE_4_DESCRIPTION)
                            )
                            .replace(
                                "{{IMAGE_5_DESCRIPTION}}", str(IMAGE_5_DESCRIPTION)
                            )
                            .replace("{{COMPOUND_PHRASE}}", str(COMPOUND_PHRASE))
                            .replace("{{COMPOUND_TYPE}}", str(COMPOUND_TYPE))
                            .replace("{{CORRECT_IMAGE}}", str(CORRECT_IMAGE)),
                        }
                    ],
                }
            ],
        )

        text = message.content[0].text

        # seperate the reasoning and output but keep the <> tags in each section
        reasoning, output = text.split("</reasoning>")
        reasoning = reasoning + "</reasoning>"
        output = "<output>" + output

        # create a python dictionary from the output
        new_record = {
            "id": item["id"],
            "language": item["language"],
            "compound": COMPOUND_PHRASE,
            "sentence_type": COMPOUND_TYPE,
            "sentence": SENTENCE,
            "style": item["style"],
            "correct_image": CORRECT_IMAGE,
            "image_1_prompt": IMAGE_1_DESCRIPTION,
            "image_1": IMAGE_1,
            "image_2_prompt": IMAGE_2_DESCRIPTION,
            "image_2": IMAGE_2,
            "image_3_prompt": IMAGE_3_DESCRIPTION,
            "image_3": IMAGE_3,
            "image_4_prompt": IMAGE_4_DESCRIPTION,
            "image_4": IMAGE_4,
            "image_5_prompt": IMAGE_5_DESCRIPTION,
            "image_5": IMAGE_5,
            "reasoning": reasoning,
            "output": output,
        }
    except Exception as e:
        print(f"Error: {e}")
        new_record = int(item["id"])

    return new_record


if __name__ == "__main__":
    try:
        client = anthropic.Anthropic()
        if client is None:
            raise Exception("Client is None")

        dataset = datasets.load_dataset(DATASET_URL)
        os.makedirs(data_save_path, exist_ok=True)

        # Process examples sequentially with a for loop
        new_examples = []
        failed_ids = []
        if ONLY_FAILED:
            old_failed_ids = set(json.load(open(failed_ids_path)))

        print(f"Processing {len(dataset[data_split])} items")
        for i, example in tqdm(enumerate(dataset[data_split])):
            if ONLY_FAILED:
                if int(example["id"]) not in old_failed_ids:
                    continue
            record = make_sft_data_for_record(example, client)
            if isinstance(record, int):
                failed_ids.append(record)
            else:
                new_examples.append(record)

            # Save every 5 records and at the end
            if (i + 1) % SAVE_FREQ == 0 or i == len(dataset[data_split]) - 1:
                save_file = os.path.join(
                    data_save_path, f"sft_data_{data_split}_mid.pkl"
                )
                with open(save_file, "wb") as f:
                    pickle.dump(new_examples, f)

                if ONLY_FAILED:
                    with open(failed_ids_path, "w") as f:
                        json.dump(failed_ids, f)

            # if (i + 1) % 3 == 0:
            #     break

        # save final file
        save_file = os.path.join(data_save_path, f"sft_data_{data_split}_complete.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(new_examples, f)

        # save failed ids
        with open(failed_ids_path, "w") as f:
            json.dump(failed_ids, f)

    except Exception as e:
        print(f"Error: {e}")
