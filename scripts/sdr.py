#!/usr/bin/env python3
import re
import json
import copy
import random
import logging
import textwrap
import argparse
from typing import List, Tuple, Dict, Any


import utils
import constants


class BongardSDR:
    """Structured Description Reasoning - Two Stage Approach with Context"""

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
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    @staticmethod
    def generate_stage1_prompt(positive_set: List[str], negative_set: List[str]) -> str:
        """Stage 1: Identify distinguishing feature between groups"""
        return textwrap.dedent(
            f"""\
            Analyze these two groups of image descriptions to identify the distinguishing feature.
            The descriptions are provided below.

            group_a:
            {chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(positive_set))}

            group_b:
            {chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(negative_set))}

            Your task is to:
            1. Carefully compare the detailed JSON descriptions in group_a and group_b.
            2. Identify the visual feature or pattern that is consistently present in group_a descriptions and distinguishes them from group_b.
            3. The distinguishing feature should be something that appears in all or most group_a images, even if it occasionally appears in group_b.

            IMPORTANT:
            - Focus on identifying the key distinguishing characteristic
            - Be specific and clear about what makes group_a different from group_b
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "analysis": "Brief analysis comparing group_a and group_b examples",
                "distinguishing_feature": "The key visual feature that distinguishes group_a from group_b"
            }}
            ```
            """
        )

    @staticmethod
    def generate_stage2_prompt(query_desc: str) -> str:
        """Stage 2: Classify query based on previously identified distinguishing feature"""
        return textwrap.dedent(
            f"""\
            Now, based on the distinguishing feature you identified in your previous analysis, classify the following query image description.

            QUERY IMAGE DESCRIPTION:

            {query_desc}

            Your task is to:
            1. Analyze the query image description for the presence or absence of the distinguishing feature you identified earlier.
            2. Classify the query as belonging to either group_a or group_b based on this feature.
            3. Provide clear reasoning for your classification, referencing your previous analysis.

            IMPORTANT:
            - The distinguishing feature should be something consistent across all `group_a` images, even if it appears occasionally in `group_b`.
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "query_analysis": "Analysis of what you observe in the query image",
                "reasoning": "Clear reasoning for the classification based on the distinguishing feature you identified earlier",
                "classification": "group_a" or "group_b"
            }}
            ```
            """
        )

    @staticmethod
    def parse_stage1_response(response_text: str) -> Tuple[str, str]:
        """Parse Stage 1 response to extract distinguishing feature"""
        try:
            # Clean response of any backticks or markdown wrappers
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            analysis = extract_field("analysis")
            distinguishing_feature = extract_field("distinguishing_feature")

            return distinguishing_feature, analysis

        except Exception as e:
            logging.error(f"Error parsing stage 1 response: {e}")
            return "", ""

    @staticmethod
    def parse_stage2_response(response_text: str) -> Tuple[str, str, str]:
        """Parse Stage 2 response to extract classification"""
        try:
            # Clean response of any backticks or markdown wrappers
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            query_analysis = extract_field("query_analysis")
            reasoning = extract_field("reasoning")
            conclusion = extract_field("classification")

            # Normalize classification
            classification = conclusion.lower()
            if "group_a" in classification:
                classification = "positive"
            elif "group_b" in classification:
                classification = "negative"
            else:
                classification = "unknown"

            return classification, query_analysis, reasoning

        except Exception as e:
            logging.error(f"Error parsing stage 2 response: {e}")
            return "unknown", "", ""

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

    def query_model_with_context(self, content: str, conversation_id: str = None) -> str:
        """Query model with optional conversation context"""
        if conversation_id and conversation_id in self.conversation_history:
            messages = self.conversation_history[conversation_id].copy()
        else:
            messages = []
        
        messages.append({"role": "user", "content": content})
        
        try:
            completion_params = utils.get_completion_params(
                self.llm, self.max_tokens, self.temperature
            )
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                **completion_params,
            )
            
            assistant_response = response.choices[0].message.content or ""
            
            # Store the updated conversation
            if conversation_id:
                messages.append({"role": "assistant", "content": assistant_response})
                self.conversation_history[conversation_id] = messages
            
            return assistant_response
            
        except Exception as e:
            logging.error(f"Error querying model: {e}")
            return ""

    def query_model(self, content: str) -> str:
        """Legacy method for backward compatibility"""
        return self.query_model_with_context(content)

    def evaluate(self):
        total = len(self.query_list)
        for idx, query in enumerate(self.query_list):
            logging.info("=" * 100)
            logging.info(
                f"Processing query {idx + 1}/{total} ({(idx + 1) / total:.2%})"
            )
            logging.info(f"UID: {query['uid']}")

            try:
                conversation_id = query['uid']
                
                # Stage 1: Identify distinguishing feature
                stage1_prompt = self.generate_stage1_prompt(
                    query["positive"], query["negative"]
                )
                
                stage1_response = self.query_model_with_context(stage1_prompt, conversation_id)
                distinguishing_feature, analysis = self.parse_stage1_response(stage1_response)

                # Stage 2: Classify query based on distinguishing feature (with context)
                stage2_prompt = self.generate_stage2_prompt(query["query"])
                
                stage2_response = self.query_model_with_context(stage2_prompt, conversation_id)
                classification, query_analysis, reasoning = self.parse_stage2_response(stage2_response)

                logging.info(f"\n{stage2_response} \n")
                logging.info(f'Answer: {classification}')
                logging.info(f'Distinguishing Feature: {distinguishing_feature}')
                logging.info(f'Analysis: {analysis}')
                logging.info(f'Query Details: {query_analysis}\n')

                # Store results
                query["distinguishing_feature"] = distinguishing_feature
                query["analysis"] = analysis
                query["answer"] = classification
                query["query_details"] = query_analysis
                query["reasoning"] = reasoning

                self.results.append(copy.deepcopy(query))

                # Clean up conversation history to save memory
                if conversation_id in self.conversation_history:
                    del self.conversation_history[conversation_id]

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
    logging.info(f"\tTwo-Stage Caption Mediated Reasoning:\tVLM: {args.vlm} \t Model: {args.llm}")

    # Initialize the evaluator
    max_tokens = 4092
    temperature = 0
    caption_path = f"{constants.CAPTIONS_DIR}/{args.vlm}_scaptions.json"
    output_path = f"{constants.RESULTS_DIR}/sdr_{args.vlm}_{args.llm}.json"
    logging.info(f"Caption path: {caption_path}")
    logging.info(f"Output path: {output_path}")

    evaluator = BongardSDR(
        args.vlm, args.llm, max_tokens, temperature, caption_path, output_path
    )
    evaluator.run()
