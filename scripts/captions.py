import os
import copy
import json
import torch
import argparse
import numpy as np
from PIL import Image
from constants import VLM_MODELS
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    LlavaProcessor, LlavaForConditionalGeneration,
    AutoProcessor, AutoModelForImageTextToText
)

def main(args):
    vlm = args.vlm

    vlm_models = {
        "blip": "Salesforce/blip-image-captioning-base",
        "blip2": "Salesforce/blip2-opt-6.7b-coco", # Doesn't work
        "instructBLIP": "Salesforce/instructblip-vicuna-7b",
        "llava": "llava-hf/llava-1.5-7b-hf",
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor based on VLM choice
    if vlm == 'blip':
        model_name = vlm_models.get(vlm)
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    elif vlm == 'blip2':
        model_name = vlm_models.get(vlm)
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    elif vlm == 'instructBLIP':
        model_name = vlm_models.get(vlm)
        processor = InstructBlipProcessor.from_pretrained(model_name)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    elif vlm == 'llava':
        processor = LlavaProcessor.from_pretrained(vlm_models[vlm])
        model = LlavaForConditionalGeneration.from_pretrained(
            vlm_models[vlm],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:
        raise ValueError(f"Unsupported VLM model: {vlm}")
    
    model.to(device)
    model.eval()

    image_path = 'assets/data/bongard-ow/bongard_ow_test.json'
    caption_path = f'captions/{vlm}.json'

    captions = []
    with open(image_path, 'r') as f:
        bongard_ow_test = json.load(f)

        total_samples = len(bongard_ow_test)
        print(f"Processing {total_samples} samples...")

        for idx, sample in enumerate(bongard_ow_test):
            uid = sample['uid']

            print("="*100) 
            print(f"Processing sample {idx}/{total_samples} ({(idx+1)/total_samples:.2%})")
            print(f"UID: {uid}")

            imageFiles = [os.path.join('assets/data/bongard-ow', imageFile) for imageFile in sample['imageFiles']]

            # Load and process images
            images = [Image.open(imageFile).convert("RGB") for imageFile in imageFiles]
            
            # Generate captions for each image
            sample_captions = []
            for i, image in enumerate(images):
                with torch.no_grad():
                    
                    if vlm == 'blip':
                        inputs = processor(images=image, return_tensors="pt").to(device)
                    elif vlm == 'llava':
                        # LLaVA requires special formatting with image tokens
                        prompt = "USER: <image>\nDescribe the image in detail.\nASSISTANT:"
                        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
                    else:
                        # BLIP2 and InstructBLIP use simpler prompts
                        inputs = processor(images=image, return_tensors="pt").to(device)
                    
                    generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
                    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                    # Clean up the caption (remove the input prompt if it's included)
                    if vlm == 'llava':
                        # For LLaVA, extract only the assistant's response
                        if "ASSISTANT:" in caption:
                            caption = caption.split("ASSISTANT:")[-1].strip()

                sample_captions.append(caption)
            
            sample['captions'] = sample_captions
            print("Captions\n")
            print(sample['captions'])
            print()
            captions.append(copy.deepcopy(sample))

        with open(caption_path, "w") as file:
            json.dump(captions, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', type=str, required=True,
                        choices=VLM_MODELS,
                        help='choose a caption model')

    args = parser.parse_args()
    main(args)
