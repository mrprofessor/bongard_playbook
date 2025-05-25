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

# client = OpenAI(
#     api_key=''  # GPT API_KEY
# )

prompt_base = '''TASK: Identify the common pattern in positive examples and classify a query.

You will be given:
- 6 positive sentences: These share a common underlying characteristic or pattern.
- 6 negative sentences: These explicitly *do not* share the pattern of the positive sentences.
- 1 query sentence: This sentence needs to be classified.

Your goal is to:
1. **Classify** the query sentence as "positive" or "negative" based on whether it exhibits this identified pattern.
2. **Identify** the core concept or pattern common to *all* positive sentences.

**Your response must be in the following exact JSON format:**

```json
{
  "classification": "[positive OR negative]",
  "pattern": "[one concise sentence describing the common pattern identified in positive sentences]",
  "reason": "[one concise sentence explaining why the query matches or doesn't match the common pattern]"
}
```

DO NOT include explanations, reasoning, or extra text. Follow the format exactly.

DATA TO ANALYZE:
'''


def extract_answer(text):
    """
    Extract classification (positive/negative) and reason from LLM response.
    If the reason is not found, returns the entire text as the sentence.
    Args:
        text (str): Raw response text from LLM
    Returns:
        tuple: (classification, common_pattern, sentence) as strings
    """
    # classification_match = re.search(r'Classification:\s*(positive|negative)', text, re.IGNORECASE)
    # sentence_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', text, re.IGNORECASE | re.DOTALL)

    classification_match = re.search(r'"classification":\s*"(positive|negative)"', text, re.IGNORECASE | re.DOTALL)
    common_pattern_match = re.search(r'"pattern":\s*"(.*?)"', text, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r'"reason":\s*"(.*?)"', text, re.IGNORECASE | re.DOTALL)

    answer = classification_match.group(1).lower() if classification_match else "unknown"
    common_pattern = common_pattern_match.group(1).strip() if common_pattern_match else "Error"
    sentence = reason_match.group(1).strip() if reason_match else text.strip()

    return answer, common_pattern, sentence


def main(args):
    vlm = args.vlm
    llm = args.llm

    caption_path = f'{vlm}.json'
    save_path = f'{vlm}_{llm}.json'

    llm_models = {
        "llama": "llama4:scout",
        "deepseek": "deepseek-r1:70b",
        "gpt41": "gpt-4.1",
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
            print("="*100)
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
                        choices=['blip2', 'instructBLIP', 'chatcap'],
                        help='choose a caption model')
    parser.add_argument('--llm', type=str, required=True,
                        choices=['llama', 'deepseek', 'gpt41'],
                        help='choose a LLM model')

    args = parser.parse_args()
    main(args)
