#!/usr/bin/env python3
import re
import json
import copy
import random
import logging
import textwrap
import argparse
from typing import List, Tuple


import utils
import constants


class BongardCMR:
    """Caption Mediated Reasoning"""

    def __init__(
        self,
        vlm: str,
        llm: str,
        max_tokens: int,
        temperature: float,
        captions_path: str,
        output_path: str,
    ):
        self.vlm = vlm
        self.llm = llm
        self.client = utils.create_client(llm)

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.caption_path = captions_path
        self.output_path = output_path
        self.model_id = constants.LLM_MODELS[llm]

        self.query_list = []
        self.results = []
        self.errors = []

    @staticmethod
    def generate_prompt(
        positive_set: List[str], negative_set: List[str], query_desc: str
    ) -> str:
        return textwrap.dedent(
            f"""\
            Compare these two groups of image descriptions and classify the query.
            The descriptions are provided in a JSON format.

            GROUP A:
            {chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(positive_set))}

            GROUP B:
            {chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(negative_set))}

            QUERY:
            {query_desc}

            Your task is to:
            1. Identify the visual feature or pattern shared among `group_a` descriptions that clearly distinguishes them from `group_b`.
            2. Analyze the `query_description` for the presence or absence of this feature.
            3. Classify the `query_description` as belonging to either `group_a` or `group_b`.

            IMPORTANT:
            - The distinguishing feature should be something consistent across all `group_a` images, even if it appears occasionally in `group_b`.
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "analysis": "Brief analysis comparing group_a and group_b examples",
                "distinguishing_feature": "Distinguishing feature between group_a and group_b.",
                "query_image": "What you observe in the query image",
                "classification": "group_a" or "group_b"
            }}
            ```
            """
        )
    

    # group_a = positive. group_b = negative
    # 1. Identify the visual feature or pattern shared among `group_a` descriptions that clearly distinguishes them from `group_b`.
    # Note: The distinguishing feature should be something consistent across all `group_a` images, even if it appears occasionally in `group_b`.


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

    def load_queries(self):
        with open(self.caption_path, "r") as f:
            data = json.load(f)

        for sample in data:
            uid = sample["uid"]
            commonSense = sample["commonSense"]
            concept = sample["concept"]
            caption = sample["caption"]
            captions = sample["captions"]
            query_a = captions[6]
            query_b = captions[13]

            base_query = {
                "uid": uid,
                "commonSense": commonSense,
                "concept": concept,
                "caption": caption,
                "positive": captions[:6],
                "negative": captions[7:13],
            }

            self.query_list.append({**base_query, "uid": uid + "_A", "query": query_a})
            self.query_list.append({**base_query, "uid": uid + "_B", "query": query_b})

        random.shuffle(self.query_list)

    def query_model(self, content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                # think = False,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logging.error(f"Error querying model: {e}")
            return ""

    def evaluate(self):
        total = len(self.query_list)
        for idx, query in enumerate(self.query_list):
            logging.info("=" * 100)
            logging.info(
                f"Processing query {idx + 1}/{total} ({(idx + 1) / total:.2%})"
            )
            logging.info(f"UID: {query['uid']}")

            prompt = self.generate_prompt(
                query["positive"], query["negative"], query["query"]
            )

            try:
                model_response = self.query_model(prompt)
                (
                    query["answer"],
                    query["distinguishing_feature"],
                    query["analysis"],
                    query["query_details"],
                ) = self.parse_response(model_response)

                logging.info(f"\n{model_response} \n")
                logging.info(f'Answer: {query["answer"]}')
                logging.info(
                    f'Distinguishing Feature: {query["distinguishing_feature"]}'
                )
                logging.info(f'Analysis: {query["analysis"]}')
                logging.info(f'Query Details: {query["query_details"],}\n')

                self.results.append(copy.deepcopy(query))
            except Exception as e:
                self.errors.append(
                    {"uid": query["uid"], "query": query["query"], "error": str(e)}
                )
                logging.error(f"Error processing query {query['uid']}: {e}")

    def save_results(self):
        with open(self.output_path, "w") as file:
            json.dump(self.results, file, indent=4)

        logging.info(f"\nResults saved to: {self.output_path}")
        if self.errors:
            logging.error("\nErrors:")
            for err in self.errors:
                logging.error(err)

    def run(self):
        self.load_queries()
        self.evaluate()
        self.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vlm",
        type=str,
        required=True,
        choices=[*constants.VLM_MODELS.keys(), *constants.AIO_MODELS.keys()],
        help="choose a caption model",
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        choices=constants.LLM_MODELS.keys(),
        help="choose a LLM model",
    )
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\tCaption Mediated Reasoning:\tVLM: {args.vlm} \t Model: {args.llm}")

    # Initialize the evaluator
    max_tokens = 4092
    temperature = 0
    caption_path = f"{constants.CAPTIONS_DIR}/{args.vlm}_captions.json"
    output_path = f"{constants.RESULTS_DIR}/cmr_{args.vlm}_{args.llm}.json"
    logging.info(f"Caption path: {caption_path}")
    logging.info(f"Output path: {output_path}")

    evaluator = BongardCMR(
        args.vlm, args.llm, max_tokens, temperature, caption_path, output_path
    )
    evaluator.run()
