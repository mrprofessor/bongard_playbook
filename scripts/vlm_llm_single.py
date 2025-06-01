import os
import re
import sys
import json
import copy
import random
import argparse

from utils import create_client
from constants import VLM_MODELS, LLM_MODELS


def generate_prompt(positive_set, negative_set, query_desc):
    return f"""
        Positive: {positive_set}

        Negative: {negative_set}

        Query: {query_desc}

        Based on the visual patterns:
            1. Identify what distinguishes positive from negative examples
            2. Determine if the query image has this feature and classify whether it's positive or negative.

        Which set does the query image belong to? Positive or Negative?

        **Your response must be in the following exact JSON format:**

        ```json
        {{
            "classification": "[positive OR negative]",
            "pattern": "[one concise sentence describing the common pattern identified in positive sentences]",
            "reasoning": "[one concise sentence explaining why the query matches or doesn't match the common pattern]"
        }}
        ```
    """


def extract_answer(text):
    """
    Extract classification (positive/negative) and reason from LLM response.
    If the reason is not found, returns the entire text as the sentence.
    Args:
        text (str): Raw response text from LLM
    Returns:
        tuple: (classification, common_pattern, reasoning) as strings
    """
    classification_match = re.search(r'"classification":\s*"(positive|negative)"', text, re.IGNORECASE | re.DOTALL)
    common_pattern_match = re.search(r'"pattern":\s*"(.*?)"', text, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r'"reason":\s*"(.*?)"', text, re.IGNORECASE | re.DOTALL)

    answer = classification_match.group(1).lower() if classification_match else "unknown"
    common_pattern = common_pattern_match.group(1).strip() if common_pattern_match else "Error"
    reasoning = reason_match.group(1).strip() if reason_match else text.strip()

    return answer, common_pattern, reasoning


def main(args):
    vlm = args.vlm
    llm = args.llm

    caption_path = f'metadata/{vlm}.json'
    save_path = f'results/{vlm}_{llm}.json'

    llm_model = LLM_MODELS.get(llm)
    client = create_client(llm)

    query_list = []
    with open(caption_path, 'r') as f:
        bongard_ow = json.load(f)
        for sample in bongard_ow:
            uid = sample['uid']
            commonSense = sample['commonSense']
            concept = sample['concept']
            caption = sample['caption']
            imageFiles = sample['imageFiles']

            captions = sample['captions']
            positive = captions[:6]
            query_a = captions[6]
            negative = captions[7:13]
            query_b = captions[13]

            query = {}
            query['commonSense'] = commonSense
            query['concept'] = concept
            query['caption'] = caption
            query['positive'] = positive
            query['negative'] = negative

            query['uid'] = uid + '_A'
            query['query'] = query_a
            query_list.append(copy.deepcopy(query))

            query['uid'] = uid + '_B'
            query['query'] = query_b
            query_list.append(copy.deepcopy(query))

        random.shuffle(query_list)

        summary = []
        error_list = []
        total_queries = len(query_list)

        for idx, query in enumerate(query_list):
            print("="*100)
            print(f"Processing query {idx+1}/{total_queries} ({(idx+1)/total_queries:.2%})")
            print(f"UID: {query['uid']}")

            prompt = generate_prompt(query['positive'], query['negative'], query['query'])

            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=4096,
                    temperature=0.7,
                    n=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                text = response.choices[0].message.content
                query['answer'], query['common_pattern'], query['sentence'] = extract_answer(text)

                print(text)
                print(f"Answer: {query['answer']}")
                print(f"Common Pattern: {query['common_pattern']}")
                print(f"Sentence: {query['sentence']}")
                print()

                summary.append(copy.deepcopy(query))
            except Exception as e:
                error = {
                    'uid': query['uid'],
                    'query': query['query'],
                    'error': str(e)
                }
                error_list.append(error)
                print(f"Error processing query {query['uid']}: {e}")

        with open(save_path, "w") as file:
            json.dump(summary, file, indent=4)

        print(error_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', type=str, required=True,
                        choices=VLM_MODELS.keys(),
                        help='choose a caption model')
    parser.add_argument('--llm', type=str, required=True,
                        choices=LLM_MODELS.keys(),
                        help='choose a LLM model')

    args = parser.parse_args()
    main(args)
