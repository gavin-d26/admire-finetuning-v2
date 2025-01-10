# contains various config variable needed for clip training

import torch

# fmt: off
pixtral_configs = {
    # model and dataset configs
    "model_url": "mistral-community/pixtral-12b",  # model url
    "dataset_url": "UCSC-Admire/idiom-SFT-dataset-561-2024-12-06_00-40-30",  # dataset url
    
    # data processing configs
    "max_length": 2144,  # max length of the "compund" text
    "truncation": True,  # truncate the "compound" text if it exceeds max_length 
    "padding": "max_length",  # pad the "compound" text if it is less than max_length
    "image_size": 224,  # image size HxW
    "num_images": 5,  # number of images in each sample (fixed for UCSC Admire dataset)
    "preprocess_batch_size": 30,  # batch size for preprocess_samples_for_pixtral
    "preprocess_num_proc": 10,  # num processes used to preprocess the dataset
    "reasoning_tag_name": "reasoning",  # tag name for reasoning
    "output_json_tag_name": "output",  # tag name for output
    "ignore_token": "</s>",  # token to ignore in the predictions and targets
    
    # training hyperparameters
    "learning_rate": 5e-7,
    "train_batch_size": 1,  # reduced to prevent OOM
    "test_batch_size": 3,  # reduced to prevent OOM
    "max_epochs": 2,
    "accumulate_grad_batches": 128,
    "limit_train_batches": None,
    "limit_test_batches": None,
    
    
    # quantization configs
    "quantize": False,  # whether to quantize the model, nf4 with FSDP2 is unstable.
    
    # lora configs
    "target_modules": "all-linear",
    "task_type": "CAUSAL_LM", 
    "r": 16,
    "lora_alpha": 8,
    "lora_path": None,  # path to load pretrained LoRA parameters from. Train a new model from scratch by leaving lora_path as None
    
    # hardware configs
    "gpus_devices": "0, 1, 3, 4, 5, 2",  # device numbers to use for training
    "accelerator": "gpu",  # device to use for training
    "num_devices": 6,  # number of accelerators to use for training
    "num_workers": 6, #4  # number of workers for the dataloader
    
    # logging configs
    "run_name": "pixtral-test-2",  # run name
    "project_name": "admire-finetuning",  # project name
    "use_wandb": True,  # use wandb for logging
    "csv_analysis_dir_path": "analysis",  # path to save the analysis csv
    
    # FSDP configs
    "strategy": "model_parallel",  # strategy to use for FSDP
    
    # data_parallel_size*tensor_parallel_size must be less than or equal to the number of GPUs available
    "data_parallel_size": 1,  # data parallel size
    "tensor_parallel_size": 6,  # tensor parallel size
}

# put a check pixtral_configs["num_devices"] == len(pixtral_configs["gpus_devices"].split(","))
if pixtral_configs["accelerator"]=="gpu" and pixtral_configs["num_devices"] != len(pixtral_configs["gpus_devices"].split(",")):
    raise ValueError(f"The number of devices specified in gpus_devices ({pixtral_configs['gpus_devices']}) does not match the number of devices specified in num_devices ({pixtral_configs['num_devices']}). Please adjust the gpus_devices.")

# fmt: on
