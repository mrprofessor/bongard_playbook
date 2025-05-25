import os
import copy
import json
import torch
import argparse
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, InstructBlipProcessor, InstructBlipForConditionalGeneration

def main(args):
    vlm = args.vlm
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and processor based on VLM choice
    if vlm == 'blip2':
        model_name = "Salesforce/blip2-opt-6.7b"
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    else:  # instructBLIP
        model_name = "Salesforce/instructblip-vicuna-7b"
        processor = InstructBlipProcessor.from_pretrained(model_name)
        model = InstructBlipForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    
    model.to(device)
    model.eval()

    image_path = 'assets/data/bongard-ow/bongard_ow_test.json'
    caption_path = f'{vlm}.json'

    captions = []
    with open(image_path, 'r') as f:
        bongard_ow_test = json.load(f)

        total_samples = len(bongard_ow_test)
        processed_samples = 0
        print(f"Processing {total_samples} samples...")

        for sample in bongard_ow_test:
            uid = sample['uid']
            
            processed_samples += 1
            print(f"Processing sample {processed_samples}/{total_samples} ({processed_samples/total_samples:.2%})")
            print(f"UID: {uid}")

            imageFiles = [os.path.join('assets/data/bongard-ow', imageFile) for imageFile in sample['imageFiles']]

            # Load and process images
            images = [Image.open(imageFile).convert("RGB") for imageFile in imageFiles]
            
            # Generate captions for each image
            sample_captions = []
            for i, image in enumerate(images):
                with torch.no_grad():
                    # Both models use the same interface
                    prompt = "Describe the image in detail."
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
                    generated_ids = model.generate(**inputs, max_new_tokens=500, do_sample=True, temperature=0.7)
                    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Clean up the caption (remove the input prompt if it's included)
                    if prompt in caption:
                        caption = caption.replace(prompt, "").strip()
                    
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
    parser.add_argument('--vlm', type=str, choices=['blip2', 'instructBLIP'], help='choose a caption model')

    
    args = parser.parse_args()
    main(args)
