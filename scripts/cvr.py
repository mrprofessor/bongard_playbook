#!/usr/bin/env python3
import re
import json
import random
import argparse
import textwrap
import logging
from typing import List, Tuple

import constants
import utils


class BongardCVR:
    """Contrastive Visual Reasoning"""

    def __init__(
        self,
        model_name: str,
        temperature: int,
        max_tokens: float,
        dataset_path: str,
        output_path: str,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.model_id = constants.AIO_MODELS[model_name]
        self.client = utils.create_client(model_name)
        self.results = []

    def load_dataset(self) -> List[dict]:
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        random.shuffle(data)
        return data

    @staticmethod
    def base_prompt() -> str:
        return textwrap.dedent(
            """\
            You are provided with 13 images: the first 6 images are `group_a`, the next 6 images are `group_b`, and the last image is the `query_image`.

            Your task is to:
            1. Identify the visual feature or pattern shared among `group_a` images that clearly distinguishes them from `group_b`.
            2. Analyze the `query_image` for the presence or absence of this feature.
            3. Classify the `query_image` as belonging to either `group_a` or `group_b`.

            IMPORTANT:
            - The distinguishing feature should be something consistent across all `group_a` images, even if it appears occasionally in `group_b`.
            - Return ONLY a valid JSON block, formatted exactly like this:

            ```
            {
                "analysis": "Brief analysis comparing group_a and group_b examples",
                "distinguishing_feature": "Distinguishing feature between group_a and group_b.",
                "query_image": "What you observe in the query image",
                "classification": "group_a" or "group_b"
            }
            ```
            """
        )

    def generate_prompt(
        self, positive_images: List[str], negative_images: List[str], test_image: str
    ) -> List[dict]:
        prompt = []
        prompt.append({"type": "text", "text": self.base_prompt()})

        for i, img_path in enumerate(positive_images):
            prompt.extend(
                [
                    {"type": "text", "text": f"\ngroup_a Example {i+1}:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{utils.encode_image(img_path)}"
                        },
                    },
                ]
            )

        for i, img_path in enumerate(negative_images):
            prompt.extend(
                [
                    {"type": "text", "text": f"\ngroup_b Example {i+1}:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{utils.encode_image(img_path)}"
                        },
                    },
                ]
            )

        prompt.extend(
            [
                {"type": "text", "text": "\nQUERY IMAGE (classify this one):"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{utils.encode_image(test_image)}"
                    },
                },
            ]
        )

        return prompt

    def query_model(self, content: List[dict]) -> str:
        try:
            completion_params = utils.get_completion_params(
                self.model_name, self.max_tokens, self.temperature
            )
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": content}],
                **completion_params,
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error querying model: {e}")
            return ""

    @staticmethod
    def parse_response(response_text: str) -> Tuple[str, str, str, str]:
        try:
            analysis = ""
            distinguishing_feature = ""
            query_details = ""
            classification = ""

            # Clean response of any backticks or markdown wrappers
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            # Extract fields using regex
            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            analysis = extract_field("analysis")
            distinguishing_feature = extract_field("distinguishing_feature")
            query_details = extract_field("query_image")
            conclusion = extract_field("classification")

            # Normalize classification
            final_classification = conclusion.lower()
            if "group_a" in final_classification:
                classification = "positive"
            elif "group_b" in final_classification:
                classification = "negative"
            else:
                classification = "unknown"

            return classification, distinguishing_feature, analysis, query_details

        except Exception as e:
            logging.error(f"Error extracting fields: {e}")
            return "unknown", "", "", ""

    def evaluate(self):
        data = self.load_dataset()
        for idx, record in enumerate(data):
            logging.info(f"Processing query {record['uid']} -- {idx + 1}/{len(data)}")
            logging.info("=" * 100)

            prompt = self.generate_prompt(
                record["positive_set"], record["negative_set"], record["query_image"]
            )
            model_response = self.query_model(prompt)

            answer, distinguishing_feature, analysis, query_details = (
                self.parse_response(model_response)
            )

            logging.info(f"\n{model_response} \n")
            logging.info(f"Answer: {answer}")
            logging.info(f"Distinguishing Feature: {distinguishing_feature}")
            logging.info(f"Analysis: {analysis}")
            logging.info(f"Query Details: {query_details}\n")

            self.results.append(
                {
                    "uid": record["uid"],
                    "commonSense": record["commonSense"],
                    "concept": record["concept"],
                    "caption": record["caption"],
                    "query_image": record["query_image"],
                    "answer": answer,
                    "distinguishing_feature": distinguishing_feature,
                    "analysis": analysis,
                    "query_details": query_details,
                    "raw_response": model_response,
                }
            )

    def save_results(self):
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Results saved to: {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=constants.AIO_MODELS,
        help="Model name (e.g., llava16, gemma3, gpt41)",
    )
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the evaluator
    model_name = args.model
    dataset_path = constants.TRANSFORMED_DATASET
    output_path = f"{constants.RESULTS_DIR}/cvr_{args.model}.json"
    max_tokens = 4092
    temperature = 0
    evaluator = BongardCVR(
        model_name, temperature, max_tokens, dataset_path, output_path
    )

    evaluator.evaluate()
    evaluator.save_results()
