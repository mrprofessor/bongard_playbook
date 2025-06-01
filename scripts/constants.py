# This file contains constants used throughout the project.
# It includes paths to data files, model names, and other config settings.

TEST_DATASET = "assets/data/bongard-ow/bongard_ow_test.json"
TRANSFORMED_DATASET = "transformed_dataset.json"

DATA_DIR = "assets/data/bongard-ow"
RESULTS_DIR = "results"
CAPTIONS_DIR = "captions"

LLM_MODELS = {
    "llama4": "llama4:scout",
    "deepseekr1": "deepseek-r1:70b",
    "qwen3": "qwen3:32b",
    "gpt41": "gpt-4.1",
}

VLM_MODELS = {
    "blip": "Salesforce/blip-image-captioning-base",
    "blip2": "Salesforce/blip2-opt-6.7b-coco",                    # Doesn't work
    "instructBLIP": "Salesforce/instructblip-vicuna-7b",
}

AIO_MODELS = {
    "llama4":  "llama4:scout",
    "llava":  "llava:34b",
    "qwen25": "qwen2.5vl:72b",
    "gemma":  "gemma3:27b",
    "gpt41":  "gpt-4.1",
}
