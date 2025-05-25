import os
import re
import json
import copy
import random
from openai import OpenAI
import argparse

# Load your API key from an environment variable or secret management service
# Initialize OpenAI client for Ollama
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama'  # Required but can be anything for Ollama
)

prompt_base = '''TASK: Pattern Recognition and Classification

GIVEN DATA:
- POSITIVE examples (6 sentences): These share a common pattern
- NEGATIVE examples (6 sentences): These do NOT share that pattern
- QUERY (1 sentence): Classify this as positive or negative

INSTRUCTIONS:
1. Find the pattern that ALL positive sentences share
2. Check if query matches this pattern
3. Respond in the EXACT format below

REQUIRED OUTPUT FORMAT:
Classification: [positive OR negative]
Common sentence: [one sentence describing the shared pattern of positive examples]

EXAMPLE:
Classification: positive
Common sentence: All positive sentences describe round objects.

DO NOT include explanations, reasoning, or extra text. Follow the format exactly.

DATA TO ANALYZE:
'''

def extract_answer(text):
    """
    Extract classification (positive/negative) and common sentence from LLM response.
    If the common sentence is not found, returns the entire text as the sentence.
    Args:
        text (str): Raw response text from LLM
    Returns:
        tuple: (classification, sentence) both as strings
    """
    classification_match = re.search(r'Classification:\s*(positive|negative)', text, re.IGNORECASE)
    sentence_match = re.search(r'Common sentence:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)

    answer = classification_match.group(1).lower() if classification_match else "unknown"
    sentence = sentence_match.group(1).strip() if sentence_match else text.strip()

    return answer, sentence


def main(args):
    vlm = args.vlm
    llm = args.llm

    caption_path = f'{vlm}.json'
    save_path = f'{vlm}_{llm}.json'

    llm_models = {
        "llama": "llama4:scout",
        "deepseek": "deepseek-r1:70b",
    }
    llm_model = llm_models.get(llm)

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
            print(f"Processing query {idx+1}/{total_queries} ({(idx+1)/total_queries:.2%})")
            print(f"UID: {query['uid']}")

            prompt = copy.deepcopy(prompt_base)
            prompt += 'positive: ' + str(query['positive']) + '\n'
            prompt += 'negative: ' + str(query['negative']) + '\n'
            prompt += 'query: ' + str(query['query']) + '\n\n'

            prompt += 'Answer:\npositive or negative:'

            try:
                response = client.chat.completions.create(
                    model=llm_model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=1024,
                    temperature=1.0,
                    n=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                text = response.choices[0].message.content + '\n'
                query['answer'], query['sentence'] = extract_answer(text)
                
                print(response)
                print(query['answer'])
                print(query['sentence'])
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
                        choices=['blip2', 'instructBLIP', 'chatcap'],
                        help='choose a caption model')
    parser.add_argument('--llm', type=str, required=True,
                        choices=['llama', 'deepseek'],
                        help='choose a LLM model')

    args = parser.parse_args()
    main(args)

