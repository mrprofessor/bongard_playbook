import copy
import json
import argparse

import constants
import utils

def generate_captioning_prompt():
    return "USER: \nProvide a brief, factual description of what you see in this image in 2-3 sentences.\nASSISTANT:"

def parse_captioner_response(response_text: str):
    return response_text.strip()

def process_dataset(model):
    caption_path = f'{constants.CAPTIONS_DIR}/{model}_captions.json'

    with open(constants.TEST_DATASET, 'r') as f:
        data = json.load(f)

    client = utils.create_client(model)
    prompt = generate_captioning_prompt()

    results = []

    for idx, record in enumerate(data):
        print(f"Processing query {record['uid']} -- {idx}/{len(data)}")
        print("="*100)

        captions = []
        for image_path in record['imageFiles']:
            image = utils.encode_image(f"{constants.DATA_DIR}/{image_path}")
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
                        ]
                    }
                ]
                response = client.chat.completions.create(
                    model=constants.AIO_MODELS.get(model),
                    messages=messages,
                    max_tokens=512
                )
                caption = response.choices[0].message.content
                print("====")
                print(image_path)
                print(caption)
                captions.append(parse_captioner_response(caption))

            except AttributeError:
                print("Error generating caption")
                captions.append("ERROR")


        # Fixed: Store the record with its captions
        record_with_captions = copy.deepcopy(record)
        record_with_captions['captions'] = captions
        results.append(record_with_captions)
        break

    breakpoint()
    with open(caption_path, "w") as file:
        json.dump(results, file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=constants.AIO_MODELS.keys(),
                        help='choose a caption model')
    args = parser.parse_args()
    process_dataset(args.model)
