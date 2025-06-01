import json
import random
import base64
import argparse

from typing import List

import constants
import utils


def query_prompt_generator(
    positive_images: List[str],
    negative_images: List[str],
    test_image: str
) -> List:
    # Prepare the message content
    prompt = []
    prompt.append({
        "type": "text",
        "text": "You will be shown a set of positive and negative image examples."
            "Your task is to determine whether the test image more closely "
            " resembles the positive or negative examples, based on visual "
            "patterns alone."
            "\nPOSITIVE EXAMPLES:\n"
    })

    # Add positive examples
    for i, img_path in enumerate(positive_images):
        prompt.extend([
            {"type": "text", "text": f"\nPositive Example {i+1}:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{utils.encode_image(img_path)}"
                }
            }
        ])

    # Add negative examples
    prompt.append({
        "type": "text",
        "text": "\nNEGATIVE EXAMPLES:\n"
    })
    for i, img_path in enumerate(negative_images):
        prompt.extend([
            {"type": "text", "text": f"\nNegative Example {i+1}:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{utils.encode_image(img_path)}"
                }
            }
        ])

    # Add the test image
    prompt.extend([
        {
            "type": "text",
            "text": "\nQUERY IMAGE (classify this one):"
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{utils.encode_image(test_image)}"
            }
        },
        {
            "type": "text",
            "text": """
Based on the visual differences between the positive and negative examples:
1. Your classification: (positive or negative)
2. What is the visual pattern shared among positive examples that is not present in the negative examples?
3. Does the test image contain this pattern?

**Your response must be in the following exact JSON format:**

`json
{
    "classification": "[positive or negative]",
    "pattern": "[shared visual feature]",
    "reasoning": "[one concise sentence explaining why the test image matches or does not match that feature]"
}
`
"""
        }
    ])
    return prompt


def process_dataset(model: str):
    # Load dataset
    with open(constants.TRANSFORMED_DATASET, 'r') as f:
        data = json.load(f)

    # Shuffle the dataset
    random.shuffle(data)

    # Initialize results list
    results = []

    for idx, record in enumerate(data):
        print(f"Processing query {record['uid']} -- {idx}/{len(data)}")
        print("="*100)

        # Generate query prompt
        content = query_prompt_generator(
            record["positive_set"],
            record["negative_set"],
            record["query_image"]
        )

        # Query VLLM model
        try:
            client = utils.create_client(args.model)
            response = client.chat.completions.create(
                model= constants.AIO_MODELS[args.model],
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=4092
            )
        except Exception as e:
            raise e

        model_response = response.choices[0].message.__getattribute__("content")

        # Parse model response
        answer, common_pattern, reasoning = utils.parse_model_response(
            model_response
        )


        print(f"Answer: {answer}")
        print(f"Common Pattern: {common_pattern}")
        print(f"Reasoning: {reasoning}\n")

        # Build output
        output = {
            "uid": record["uid"],
            "commonSense": record["commonSense"],
            "concept": record["concept"],
            "caption": record["caption"],
            "query_image": record["query_image"],
            "answer": answer,
            "common_pattern": common_pattern,
            "reasoning": reasoning,
            "raw_response": model_response  # <-- use model response here
        }
        results.append(output)

    # Save results
    results_file_path = f"{constants.RESULTS_DIR}/{args.model}.json"
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=constants.AIO_MODELS,
        help='Model name (e.g., llava, gemma, gpt41)'
    )
    args = parser.parse_args()
    process_dataset(args.model)
