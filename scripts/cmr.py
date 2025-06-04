#!/usr/bin/env python3
import os
import re
import json
import copy
import random
import argparse
from typing import List, Tuple

import utils
import constants


class BongardCMR:
    def __init__(self, vlm: str, llm: str, captions_path: str, output_path: str):
        self.vlm = vlm
        self.llm = llm
        self.llm_model = constants.LLM_MODELS[llm]
        self.client = utils.create_client(llm)

        self.caption_path = captions_path
        self.output_path = output_path

        self.query_list = []
        self.results = []
        self.errors = []

    @staticmethod
    def generate_prompt(positive_set: List[str], negative_set: List[str], query_desc: str) -> str:
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

    @staticmethod
    def extract_answer(text: str) -> Tuple[str, str, str]:
        classification_match = re.search(
            r'"classification":\s*"(positive|negative)"', text, re.IGNORECASE | re.DOTALL
        )
        common_pattern_match = re.search(
            r'"pattern":\s*"(.*?)"', text, re.IGNORECASE | re.DOTALL
        )
        reason_match = re.search(r'"reason":\s*"(.*?)"', text, re.IGNORECASE | re.DOTALL)

        answer = classification_match.group(1).lower() if classification_match else "unknown"
        common_pattern = common_pattern_match.group(1).strip() if common_pattern_match else "Error"
        reasoning = reason_match.group(1).strip() if reason_match else text.strip()

        return answer, common_pattern, reasoning

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

    def evaluate(self):
        total = len(self.query_list)
        for idx, query in enumerate(self.query_list):
            print("=" * 100)
            print(f"Processing query {idx + 1}/{total} ({(idx + 1) / total:.2%})")
            print(f"UID: {query['uid']}")

            prompt = self.generate_prompt(query["positive"], query["negative"], query["query"])

            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4096,
                    temperature=0.7,
                    n=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                text = response.choices[0].message.content
                query["answer"], query["common_pattern"], query["sentence"] = self.extract_answer(text)

                print(text)
                print(f"Answer: {query['answer']}")
                print(f"Common Pattern: {query['common_pattern']}")
                print(f"Sentence: {query['sentence']}\n")

                self.results.append(copy.deepcopy(query))
            except Exception as e:
                self.errors.append({"uid": query["uid"], "query": query["query"], "error": str(e)})
                print(f"Error processing query {query['uid']}: {e}")

    def save_results(self):
        with open(self.output_path, "w") as file:
            json.dump(self.results, file, indent=4)

        print(f"\nResults saved to: {self.output_path}")
        if self.errors:
            print("\nErrors:")
            for err in self.errors:
                print(err)

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
        choices=constants.VLM_MODELS.keys(),
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

    caption_path = f"{constants.CAPTIONS_DIR}/{args.vlm}_captions.json"
    output_path = f"{constants.RESULTS_DIR}/{args.vlm}_{args.llm}.json"

    evaluator = BongardCMR(args.vlm, args.llm, caption_path, output_path)
    evaluator.run()
