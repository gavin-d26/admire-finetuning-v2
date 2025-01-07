import os
import pickle
from datetime import datetime
from datasets import load_dataset, Dataset, Features, Value, Image, DatasetDict


train_path = None
test_path = None
push_to_hub = False
original_dataset = None

# extract NUM_COMPOUNDS from the original dataset name
compounds = original_dataset.split("-")[3]

# Load the pickle files
with open(train_path, "rb") as f:
    train = pickle.load(f)

with open(test_path, "rb") as f:
    test = pickle.load(f)


# Create the dataset
features = Features(
    {
        "id": Value("int32"),
        "language": Value("string"),
        "compound": Value("string"),
        "sentence_type": Value("string"),
        "sentence": Value("string"),
        "style": Value("string"),
        "correct_image": Value("int32"),
        "image_1_prompt": Value("string"),
        "image_2_prompt": Value("string"),
        "image_3_prompt": Value("string"),
        "image_4_prompt": Value("string"),
        "image_5_prompt": Value("string"),
        "image_1": Image(),
        "image_2": Image(),
        "image_3": Image(),
        "image_4": Image(),
        "image_5": Image(),
        "reasoning": Value("string"),
        "output": Value("string"),
    }
)


# Create a DatasetDict with train and test splits
dataset_dict = DatasetDict(
    {
        "train": Dataset.from_list(train, features=features),
        "test": Dataset.from_list(test, features=features),
    }
)


# Push to hub if requested
if push_to_hub:
    organization_name = "UCSC-Admire"
    dataset_name = (
        f"idiom-SFT-dataset-{compounds}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    dataset_dict.push_to_hub(
        f"{organization_name}/{dataset_name}",
        token=os.environ["HUGGINGFACE_TOKEN"],
    )
    print(f"Dataset pushed to HuggingFace Hub: {organization_name}/{dataset_name}")
