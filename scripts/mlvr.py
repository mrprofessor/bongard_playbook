#!/usr/bin/env python3
import re
import json
import random
import argparse
import textwrap
import logging
from typing import List, Tuple, Dict, Any

import constants
import utils


class BongardHierarchical:
    """Hierarchical Visual Reasoning for Bongard Problems"""

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
    def hierarchical_prompt() -> str:
        return textwrap.dedent(
            """\
            You are provided with 13 images: the first 6 images are `group_a`, the next 6 images are `group_b`, and the last image is the `query_image`.

            Analyze each image using this hierarchical structure and find the distinguishing pattern:

            FOR EACH IMAGE, EXTRACT:

            GLOBAL LEVEL:
            - scene_type: [indoor/outdoor/abstract/sports/nature/etc]
            - composition: [centered/distributed/clustered/linear/etc]
            - dominant_elements: [what stands out most]
            - overall_complexity: [simple/moderate/complex]

            OBJECT LEVEL:
            - primary_objects: [list main objects/entities with their properties]
            - object_count: [total number of distinct objects]
            - object_types: [categories of objects present]
            - object_properties: [colors, sizes, materials, states]

            RELATIONSHIP LEVEL:
            - spatial_relations: [positioning between objects]
            - interactions: [how objects/people interact]
            - groupings: [how objects cluster together]
            - alignments: [any systematic arrangements]

            PATTERN LEVEL:
            - repetitions: [what repeats and how]
            - symmetries: [types of symmetry present]
            - variations: [systematic changes across elements]
            - regularities: [consistent rules or structures]

            SEMANTIC LEVEL:
            - activity: [what action/event is happening]
            - context: [setting, purpose, meaning]
            - category: [high-level classification]
            - function: [purpose or use of scene/objects]

            TASK:
            1. Extract hierarchical features for all images
            2. Compare group_a vs group_b at each level systematically
            3. Identify which level contains the most consistent distinguishing pattern
            4. Verify this pattern holds across ALL group_a images
            5. Confirm this pattern reliably separates group_a from group_b
            6. Apply the discovered pattern to classify the query_image

            Return EXACTLY in the following format:

            ```json
            {
                "hierarchical_analysis": {
                    "global_differences": "What differs at global level",
                    "object_differences": "What differs at object level",
                    "relationship_differences": "What differs at relationship level",
                    "pattern_differences": "What differs at pattern level",
                    "semantic_differences": "What differs at semantic level"
                },
                "distinguishing_level": "Which level (global/object/relationship/pattern/semantic) contains the key difference",
                "distinguishing_feature": "The specific feature that separates the groups",
                "group_a_pattern": "How group_a consistently exhibits this feature",
                "group_b_pattern": "How group_b consistently differs from this feature",
                "query_analysis": "Hierarchical analysis of the query image",
                "query_feature_match": "Does query exhibit the distinguishing feature?",
                "classification": "group_a" or "group_b",
                "confidence": "high/medium/low"
            }
            ```
            """
        )

    @staticmethod
    def progressive_prompt() -> str:
        return textwrap.dedent(
            """\
            You are provided with 13 images: the first 6 images are `group_a`, the next 6 images are `group_b`, and the last image is the `query_image`.

            Use this PROGRESSIVE ANALYSIS approach:

            STEP 1 - SURFACE INSPECTION:
            Look at immediate visual elements: colors, shapes, objects, people
            Question: Is there an obvious visual difference?

            STEP 2 - ACTION ANALYSIS:
            Identify activities, behaviors, interactions happening in each image
            Question: Do the groups show different types of actions?

            STEP 3 - COMPOSITIONAL ANALYSIS:
            Examine layout, arrangement, spatial relationships, groupings
            Question: Are objects arranged differently between groups?

            STEP 4 - CATEGORICAL ANALYSIS:
            Consider high-level categories: sports types, settings, object classes
            Question: Do the groups represent different categories?

            STEP 5 - FUNCTIONAL ANALYSIS:
            Think about purposes, capabilities, affordances, uses
            Question: Do objects/scenes serve different functions?

            STEP 6 - CONTEXTUAL ANALYSIS:
            Consider setting, environment, cultural context, meaning
            Question: Do the groups occur in different contexts?

            STEP 7 - PATTERN VERIFICATION:
            Test your hypothesis against ALL examples in both groups
            Question: Does your proposed pattern consistently hold?

            For each step, explicitly state what you observe before moving to the next level.
            Stop when you find a pattern that reliably distinguishes the groups.

            Return ONLY a valid JSON block:

            ```
            {
                "step_by_step_analysis": {
                    "step_1_surface": "Surface-level observations",
                    "step_2_actions": "Action-level observations",
                    "step_3_composition": "Compositional observations",
                    "step_4_categories": "Categorical observations",
                    "step_5_functions": "Functional observations",
                    "step_6_context": "Contextual observations",
                    "step_7_verification": "Pattern verification results"
                },
                "stopping_point": "Which step revealed the distinguishing pattern",
                "distinguishing_feature": "The feature that separates the groups",
                "verification": "Confirmation this pattern holds for all examples",
                "query_analysis": "Step-by-step analysis of query image",
                "classification": "group_a" or "group_b",
                "reasoning": "Why the query belongs to this group"
            }
            ```
            """
        )

    def generate_prompt(
        self, positive_images: List[str], negative_images: List[str], test_image: str, method: str = "hierarchical"
    ) -> List[dict]:
        prompt = []

        if method == "hierarchical":
            prompt.append({"type": "text", "text": self.hierarchical_prompt()})
        else:
            prompt.append({"type": "text", "text": self.progressive_prompt()})

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
    def parse_hierarchical_response(response_text: str) -> Tuple[str, str, str, Dict[str, Any]]:
        try:
            # Clean response
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            # Try to parse as JSON first
            try:
                data = json.loads(cleaned)
                classification = data.get("classification", "").lower()
                distinguishing_feature = data.get("distinguishing_feature", "")

                # Extract analysis components
                hierarchical_analysis = data.get("hierarchical_analysis", {})
                analysis_summary = f"Level: {data.get('distinguishing_level', '')}, Feature: {distinguishing_feature}"

                # Normalize classification
                if "group_a" in classification:
                    final_classification = "positive"
                elif "group_b" in classification:
                    final_classification = "negative"
                else:
                    final_classification = "unknown"

                return final_classification, distinguishing_feature, analysis_summary, data

            except json.JSONDecodeError:
                # Fallback to regex extraction
                def extract_field(field_name: str) -> str:
                    pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                    match = re.search(pattern, cleaned, re.IGNORECASE)
                    return match.group(1).strip() if match else ""

                distinguishing_feature = extract_field("distinguishing_feature")
                classification = extract_field("classification").lower()
                analysis_summary = extract_field("group_a_pattern") + " vs " + extract_field("group_b_pattern")

                if "group_a" in classification:
                    final_classification = "positive"
                elif "group_b" in classification:
                    final_classification = "negative"
                else:
                    final_classification = "unknown"

                return final_classification, distinguishing_feature, analysis_summary, {}

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return "unknown", "", "", {}

    @staticmethod
    def parse_progressive_response(response_text: str) -> Tuple[str, str, str, Dict[str, Any]]:
        try:
            # Clean response
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            # Try to parse as JSON first
            try:
                data = json.loads(cleaned)
                classification = data.get("classification", "").lower()
                distinguishing_feature = data.get("distinguishing_feature", "")

                # Extract step analysis
                step_analysis = data.get("step_by_step_analysis", {})
                stopping_point = data.get("stopping_point", "")
                analysis_summary = f"Stopped at: {stopping_point}, Feature: {distinguishing_feature}"

                # Normalize classification
                if "group_a" in classification:
                    final_classification = "positive"
                elif "group_b" in classification:
                    final_classification = "negative"
                else:
                    final_classification = "unknown"

                return final_classification, distinguishing_feature, analysis_summary, data

            except json.JSONDecodeError:
                # Fallback to regex extraction similar to hierarchical
                def extract_field(field_name: str) -> str:
                    pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                    match = re.search(pattern, cleaned, re.IGNORECASE)
                    return match.group(1).strip() if match else ""

                distinguishing_feature = extract_field("distinguishing_feature")
                classification = extract_field("classification").lower()
                reasoning = extract_field("reasoning")

                if "group_a" in classification:
                    final_classification = "positive"
                elif "group_b" in classification:
                    final_classification = "negative"
                else:
                    final_classification = "unknown"

                return final_classification, distinguishing_feature, reasoning, {}

        except Exception as e:
            logging.error(f"Error parsing progressive response: {e}")
            return "unknown", "", "", {}

    def evaluate(self, method: str = "hierarchical"):
        data = self.load_dataset()
        for idx, record in enumerate(data):
            logging.info(f"Processing query {record['uid']} -- {idx + 1}/{len(data)} (method: {method})")
            logging.info("=" * 100)

            prompt = self.generate_prompt(
                record["positive_set"], record["negative_set"], record["query_image"], method
            )
            model_response = self.query_model(prompt)

            if method == "hierarchical":
                answer, distinguishing_feature, analysis, raw_data = self.parse_hierarchical_response(model_response)
            else:
                answer, distinguishing_feature, analysis, raw_data = self.parse_progressive_response(model_response)

            logging.info(f"\n{model_response} \n")
            logging.info(f"Answer: {answer}")
            logging.info(f"Method: {method}")
            logging.info(f"Distinguishing Feature: {distinguishing_feature}")
            logging.info(f"Analysis: {analysis}\n")

            self.results.append(
                {
                    "uid": record["uid"],
                    "commonSense": record["commonSense"],
                    "concept": record["concept"],
                    "caption": record["caption"],
                    "query_image": record["query_image"],
                    "method": method,
                    "answer": answer,
                    "distinguishing_feature": distinguishing_feature,
                    "analysis": analysis,
                    "raw_response": model_response,
                    "parsed_data": raw_data,
                }
            )

    def save_results(self, method: str):
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
    parser.add_argument(
        "--method",
        type=str,
        default="hierarchical",
        choices=["hierarchical", "progressive"],
        help="Analysis method: hierarchical or progressive",
    )
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Initialize the evaluator
    model_name = args.model
    method = args.method
    dataset_path = constants.TRANSFORMED_DATASET
    output_path = f"{constants.RESULTS_DIR}/mlvr_{method}_{args.model}.json"
    max_tokens = 4092
    temperature = 0

    evaluator = BongardHierarchical(
        model_name, temperature, max_tokens, dataset_path, output_path
    )

    evaluator.evaluate(method)
    evaluator.save_results(method)
