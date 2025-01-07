# contains various config variable needed for clip training

clip_configs = {
    "model_url": "openai/clip-vit-base-patch32",  # model url
    "dataset_url": "UCSC-Admire/idiom-dataset-561-2024-12-02_11-48-08",  # dataset url
    # num processes used to preprocess the dataset
    "num_proc": 10,
    "max_length": 64,  # max length of the "compund" text
    "truncation": True,  # truncate the "compound" text if it exceeds max_length
    "padding": "max_length",  # pad the "compound" text if it is less than max_length
    "similarity_weights": [1.5, 1.25, 1.125, 1],  # weights for the similarity scores
    # image size HxW
    "image_size": 224,
    "precision": "bf16-mixed",  # precision for training
    # number of images in each sample of the dataset (this is fixed for the UCSC Admire dataset -> 5)
    "num_images": 5,
    # training hyperparameters
    "learning_rate": 1e-6,
    "batch_size": 128,
    "max_epochs": 2,
    # training hardware
    "gpus_devices": "4",  # device number to use for training
    "accelerator": "gpu",  # device to use for training
}
