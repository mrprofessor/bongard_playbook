import json
import random
import base64
import argparse

from typing import List

import constants
import utils


def visual_concept_test_prompt(m, n):
    """
    Generates a visual analysis prompt for DVRL paradigm.
    Args:
        m (int): Number of positive samples.
        n (int): Number of negative samples.
    Returns:
        str: The formatted prompt string.
    """
    return f"""
You are provided with {m + n + 1} images: the first {m} samples are 'cat_1', the next {n} samples are 'cat_2', and the last image is the 'query image'.

Analyze the common characteristics or patterns found in the 'cat_1' samples (positive samples: following 1 common rule) that distinctly separate them from the 'cat_2' samples (negative samples: it might not follow any possible rule).

Your task is to:
1. Determine the rule or criterion that distinguishes the 'cat_1' samples from the 'cat_2' ones.
2. Analyse the 'query image' (last image).
3. Provide your conclusion for the 'query image' if it can be categorized as either 'cat_1' or 'cat_2' based on the analysis and the rule.

Ensure that the output is clear, well-formatted, and free of unnecessary explanations.
Omit the ''' tags at the beginning and end of the page. The format of your output should be as follows:
- **Analysis**: (Your analysis here)
- **Rule**: (The distinguishing rule here)
- **Query Image**: (Query image details)
- **Conclusion**: (cat_1 or cat_2)
"""


def dvrl_prompt_generator(
    positive_images: List[str],
    negative_images: List[str],
    test_image: str
) -> List:
    """
    Generate DVRL prompt following the paper's exact methodology.
    Images are presented in sequence: positives first, then negatives, then query.
    """
    m = len(positive_images)
    n = len(negative_images)

    # Prepare the message content
    prompt = []

    # Add the task instruction
    prompt.append({
        "type": "text",
        "text": visual_concept_test_prompt(m, n)
    })

    # Add positive examples (cat_1) first
    for i, img_path in enumerate(positive_images):
        prompt.extend([
            {"type": "text", "text": f"\ncat_1 Example {i+1}:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{utils.encode_image(img_path)}"
                }
            }
        ])

    # Add negative examples (cat_2) second
    for i, img_path in enumerate(negative_images):
        prompt.extend([
            {"type": "text", "text": f"\ncat_2 Example {i+1}:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{utils.encode_image(img_path)}"
                }
            }
        ])

    # Add the query/test image last
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
        }
    ])

    return prompt


def parse_dvrl_response(response_text: str):
    """
    Parse the DVRL response according to the expected format.
    Returns analysis, rule, query_details, and conclusion.
    """
    try:
        lines = response_text.strip().split('\n')
        analysis = ""
        rule = ""
        query_details = ""
        conclusion = ""

        current_section = None

        for line in lines:
            line = line.strip()
            if line.startswith('- **Analysis**:') or line.startswith('**Analysis**:'):
                current_section = 'analysis'
                analysis = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line.startswith('- **Rule**:') or line.startswith('**Rule**:'):
                current_section = 'rule'
                rule = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line.startswith('- **Query Image**:') or line.startswith('**Query Image**:'):
                current_section = 'query'
                query_details = line.split(':', 1)[1].strip() if ':' in line else ""
            elif line.startswith('- **Conclusion**:') or line.startswith('**Conclusion**:'):
                current_section = 'conclusion'
                conclusion = line.split(':', 1)[1].strip() if ':' in line else ""
            elif current_section and line:
                # Continue adding to current section if it's a continuation
                if current_section == 'analysis':
                    analysis += " " + line
                elif current_section == 'rule':
                    rule += " " + line
                elif current_section == 'query':
                    query_details += " " + line
                elif current_section == 'conclusion':
                    conclusion += " " + line

        # Extract final classification (cat_1 or cat_2)
        final_classification = conclusion.lower()
        if 'cat_1' in final_classification:
            classification = 'positive'
        elif 'cat_2' in final_classification:
            classification = 'negative'
        else:
            # Fallback parsing
            classification = 'positive' if 'positive' in final_classification else 'negative'

        return classification, rule.strip(), analysis.strip(), query_details.strip()

    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response text: {response_text}")
        return "unknown", "", "", ""


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

        # Generate DVRL prompt (following paper's methodology)
        content = dvrl_prompt_generator(
            record["positive_set"],
            record["negative_set"],
            record["query_image"]
        )

        # Query VLLM model
        try:
            client = utils.create_client(args.model)
            response = client.chat.completions.create(
                model=constants.AIO_MODELS[args.model],
                messages=[{
                    "role": "user",
                    "content": content
                }],
                max_tokens=4096,
                temperature=0  # Paper uses temperature 0
            )
        except Exception as e:
            print(f"Error querying model: {e}")
            continue

        model_response = response.choices[0].message.content

        print()
        print(model_response)
        print()

        # Parse model response according to DVRL format
        answer, rule, analysis, query_details = parse_dvrl_response(model_response)

        print(f"Answer: {answer}")
        print(f"Rule: {rule}")
        print(f"Analysis: {analysis}")
        print(f"Query Details: {query_details}\n")

        # Build output
        output = {
            "uid": record["uid"],
            "commonSense": record["commonSense"],
            "concept": record["concept"],
            "caption": record["caption"],
            "query_image": record["query_image"],
            "answer": answer,
            "rule": rule,
            "analysis": analysis,
            "query_details": query_details,
            "raw_response": model_response
        }
        results.append(output)

    # Save results
    results_file_path = f"{constants.RESULTS_DIR}/{args.model}_dvrl.json"
    with open(results_file_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to: {results_file_path}")


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
