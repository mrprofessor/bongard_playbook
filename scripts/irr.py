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


class BongardIRR:
    """Iterative Refinement Reasoning - Multi-Stage Approach with Feedback Loops"""

    def __init__(
        self,
        vlm: str,
        llm: str,
        max_tokens: int,
        temperature: float,
        captions_path: str,
        output_path: str,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
    ):
        self.vlm = vlm
        self.llm = llm
        self.client = utils.create_client(llm)

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.caption_path = captions_path
        self.output_path = output_path
        self.model_id = constants.LLM_MODELS[llm]
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        self.query_list = []
        self.results = []
        self.errors = []
        self.conversation_history: Dict[str, List[Dict[str, str]]] = {}

    @staticmethod
    def generate_stage1_prompt(positive_set: List[str], negative_set: List[str]) -> str:
        """Stage 1: Initial distinguishing feature identification"""
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
            - Rate your confidence in this feature identification (0.0 to 1.0)
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "analysis": "Brief analysis comparing group_a and group_b examples",
                "distinguishing_feature": "The key visual feature that distinguishes group_a from group_b",
                "confidence": 0.85,
                "reasoning": "Explanation of why you are confident/uncertain about this feature"
            }}
            ```
            """
        )

    @staticmethod
    def generate_validation_prompt(
        positive_set: List[str], negative_set: List[str], feature: str, confidence: float
    ) -> str:
        """Stage 2: Validate and potentially refine the distinguishing feature"""
        return textwrap.dedent(
            f"""\
            You previously identified this distinguishing feature: "{feature}"
            Your confidence was: {confidence}

            Now, let's validate this feature more carefully against ALL examples:

            group_a:
            {chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(positive_set))}

            group_b:
            {chr(10).join(f"{i+1}. {caption}" for i, caption in enumerate(negative_set))}

            Your task is to:
            1. Check if the feature "{feature}" is truly present in ALL or MOST group_a examples
            2. Check if this feature is absent or rare in group_b examples
            3. If the feature doesn't work well, propose a better alternative
            4. Rate your confidence in the final feature

            IMPORTANT:
            - Be critical of your previous analysis
            - Look for counterexamples that might disprove the feature
            - If you find a better feature, use it instead
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "validation_analysis": "Critical analysis of the previous feature",
                "feature_works": true/false,
                "counterexamples": "Any examples that don't fit the feature",
                "refined_feature": "The refined or new distinguishing feature (same as original if no change)",
                "confidence": 0.90,
                "reasoning": "Explanation of your validation and refinement process"
            }}
            ```
            """
        )

    @staticmethod
    def generate_classification_prompt(query_desc: str, feature: str) -> str:
        """Stage 3: Initial classification based on validated distinguishing feature"""
        return textwrap.dedent(
            f"""\
            Based on our thorough analysis, we identified this distinguishing feature:
            "{feature}"

            Now, classify the following query image description:

            QUERY IMAGE DESCRIPTION:
            {query_desc}

            Your task is to:
            1. Analyze the query image description for the presence or absence of the distinguishing feature
            2. Classify the query as belonging to either group_a or group_b based on this feature
            3. Provide clear reasoning for your classification
            4. Rate your confidence in this classification (0.0 to 1.0)

            IMPORTANT:
            - The distinguishing feature should be something consistent across all `group_a` images
            - Be precise about what constitutes the feature vs similar but different behaviors
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "query_analysis": "Analysis of what you observe in the query image",
                "feature_present": true/false,
                "reasoning": "Clear reasoning for the classification based on the distinguishing feature",
                "classification": "group_a" or "group_b",
                "confidence": 0.85
            }}
            ```
            """
        )

    @staticmethod
    def generate_classification_validation_prompt(query_desc: str, feature: str, 
                                                initial_classification: str, 
                                                initial_reasoning: str, 
                                                confidence: float) -> str:
        """Stage 4: Validate and potentially refine the classification"""
        return textwrap.dedent(
            f"""\
            You previously classified this query as: {initial_classification}
            Your reasoning was: "{initial_reasoning}"
            Your confidence was: {confidence}

            Let's double-check this classification more carefully:

            DISTINGUISHING FEATURE: "{feature}"
            QUERY IMAGE DESCRIPTION: {query_desc}

            Your task is to:
            1. Re-examine the query for the EXACT distinguishing feature (not just similar behaviors)
            2. Look for subtle differences that might change the classification
            3. Consider edge cases or alternative interpretations
            4. Verify if your initial classification was correct

            CRITICAL QUESTIONS TO ASK YOURSELF:
            - Does the query show the EXACT same type of behavior as group_a examples?
            - Could this be a similar but fundamentally different activity?
            - Are you being too broad or too narrow in applying the feature?

            IMPORTANT:
            - Be highly critical of your previous analysis
            - If you find any doubt, refine your classification
            - Return ONLY a valid JSON block, formatted exactly like this: (Do not produce any other text)

            ```
            {{
                "validation_analysis": "Critical re-examination of the query against the exact feature",
                "classification_correct": true/false,
                "refined_reasoning": "Updated reasoning based on careful re-analysis",
                "refined_classification": "group_a" or "group_b",
                "confidence": 0.90,
                "key_distinctions": "Important distinctions that influenced the final decision"
            }}
            ```
            """
        )

    @staticmethod
    def parse_stage1_response(response_text: str) -> Tuple[str, str, float, str]:
        """Parse Stage 1 response to extract initial feature and confidence"""
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            def extract_float(field_name: str) -> float:
                pattern = rf'"?{field_name}"?\s*:\s*([0-9]*\.?[0-9]+)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return float(match.group(1)) if match else 0.0

            analysis = extract_field("analysis")
            distinguishing_feature = extract_field("distinguishing_feature")
            confidence = extract_float("confidence")
            reasoning = extract_field("reasoning")

            return distinguishing_feature, analysis, confidence, reasoning

        except Exception as e:
            logging.error(f"Error parsing stage 1 response: {e}")
            return "", "", 0.0, ""

    @staticmethod
    def parse_validation_response(response_text: str) -> Tuple[str, bool, str, str, float, str]:
        """Parse validation response"""
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            def extract_bool(field_name: str) -> bool:
                pattern = rf'"?{field_name}"?\s*:\s*(true|false)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).lower() == "true" if match else False

            def extract_float(field_name: str) -> float:
                pattern = rf'"?{field_name}"?\s*:\s*([0-9]*\.?[0-9]+)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return float(match.group(1)) if match else 0.0

            validation_analysis = extract_field("validation_analysis")
            feature_works = extract_bool("feature_works")
            counterexamples = extract_field("counterexamples")
            refined_feature = extract_field("refined_feature")
            confidence = extract_float("confidence")
            reasoning = extract_field("reasoning")

            return validation_analysis, feature_works, counterexamples, refined_feature, confidence, reasoning

        except Exception as e:
            logging.error(f"Error parsing validation response: {e}")
            return "", False, "", "", 0.0, ""

    @staticmethod
    def parse_classification_response(response_text: str) -> Tuple[str, bool, str, str, float]:
        """Parse initial classification response"""
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            def extract_bool(field_name: str) -> bool:
                pattern = rf'"?{field_name}"?\s*:\s*(true|false)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).lower() == "true" if match else False

            def extract_float(field_name: str) -> float:
                pattern = rf'"?{field_name}"?\s*:\s*([0-9]*\.?[0-9]+)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return float(match.group(1)) if match else 0.0

            query_analysis = extract_field("query_analysis")
            feature_present = extract_bool("feature_present")
            reasoning = extract_field("reasoning")
            conclusion = extract_field("classification")
            confidence = extract_float("confidence")

            # Normalize classification
            classification = conclusion.lower()
            if "group_a" in classification:
                classification = "positive"
            elif "group_b" in classification:
                classification = "negative"
            else:
                classification = "unknown"

            return query_analysis, feature_present, reasoning, classification, confidence

        except Exception as e:
            logging.error(f"Error parsing classification response: {e}")
            return "", False, "", "unknown", 0.0

    @staticmethod
    def parse_classification_validation_response(response_text: str) -> Tuple[str, bool, str, str, float, str]:
        """Parse classification validation response"""
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()

            def extract_field(field_name: str) -> str:
                pattern = rf'"?{field_name}"?\s*:\s*"([^"]+)"'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).strip() if match else ""

            def extract_bool(field_name: str) -> bool:
                pattern = rf'"?{field_name}"?\s*:\s*(true|false)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return match.group(1).lower() == "true" if match else False

            def extract_float(field_name: str) -> float:
                pattern = rf'"?{field_name}"?\s*:\s*([0-9]*\.?[0-9]+)'
                match = re.search(pattern, cleaned, re.IGNORECASE)
                return float(match.group(1)) if match else 0.0

            validation_analysis = extract_field("validation_analysis")
            classification_correct = extract_bool("classification_correct")
            refined_reasoning = extract_field("refined_reasoning")
            refined_conclusion = extract_field("refined_classification")
            confidence = extract_float("confidence")
            key_distinctions = extract_field("key_distinctions")

            # Normalize classification
            classification = refined_conclusion.lower()
            if "group_a" in classification:
                classification = "positive"
            elif "group_b" in classification:
                classification = "negative"
            else:
                classification = "unknown"

            return validation_analysis, classification_correct, refined_reasoning, classification, confidence, key_distinctions

        except Exception as e:
            logging.error(f"Error parsing classification validation response: {e}")
            return "", False, "", "unknown", 0.0, ""

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

    def iterative_feature_refinement(self, query: dict) -> Tuple[str, float, str, List[dict]]:
        """Iteratively refine the distinguishing feature until confident or max iterations"""
        conversation_id = query['uid']
        
        # Stage 1: Initial feature identification
        stage1_prompt = self.generate_stage1_prompt(query["positive"], query["negative"])
        stage1_response = self.query_model_with_context(stage1_prompt, conversation_id)
        
        current_feature, analysis, confidence, reasoning = self.parse_stage1_response(stage1_response)
        
        logging.info(f"Initial feature: {current_feature} (confidence: {confidence})")
        
        iteration = 1
        validation_history = []
        
        # Iterative refinement loop
        while iteration <= self.max_iterations and confidence < self.confidence_threshold:
            logging.info(f"Refinement iteration {iteration}/{self.max_iterations}")
            
            # Stage 2: Validation and refinement
            validation_prompt = self.generate_validation_prompt(
                query["positive"], query["negative"], current_feature, confidence
            )
            validation_response = self.query_model_with_context(validation_prompt, conversation_id)
            
            (validation_analysis, feature_works, counterexamples, 
             refined_feature, new_confidence, validation_reasoning) = self.parse_validation_response(validation_response)
            
            # Store validation history
            validation_history.append({
                "iteration": iteration,
                "original_feature": current_feature,
                "refined_feature": refined_feature,
                "confidence_change": new_confidence - confidence,
                "feature_works": feature_works,
                "counterexamples": counterexamples
            })
            
            # Update current feature and confidence
            current_feature = refined_feature if refined_feature else current_feature
            confidence = new_confidence
            
            logging.info(f"Iteration {iteration}: {current_feature} (confidence: {confidence})")
            
            iteration += 1
        
        final_analysis = f"Converged after {iteration-1} iterations. Final confidence: {confidence}"
        
        return current_feature, confidence, final_analysis, validation_history

    def iterative_query_classification(self, query: dict, feature: str, conversation_id: str) -> Tuple[str, str, bool, str, float, List[dict]]:
        """Iteratively classify the query with validation and refinement"""
        
        # Initial classification
        classification_prompt = self.generate_classification_prompt(query["query"], feature)
        classification_response = self.query_model_with_context(classification_prompt, conversation_id)
        
        query_analysis, feature_present, reasoning, classification, confidence = (
            self.parse_classification_response(classification_response)
        )
        
        logging.info(f"Initial classification: {classification} (confidence: {confidence})")
        
        classification_history = [{
            "iteration": 0,
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning
        }]
        
        iteration = 1
        
        # Iterative validation loop
        while iteration <= self.max_iterations and confidence < self.confidence_threshold:
            logging.info(f"Classification validation iteration {iteration}/{self.max_iterations}")
            
            # Validation and refinement
            validation_prompt = self.generate_classification_validation_prompt(
                query["query"], feature, classification, reasoning, confidence
            )
            validation_response = self.query_model_with_context(validation_prompt, conversation_id)
            
            (validation_analysis, classification_correct, refined_reasoning, 
             refined_classification, new_confidence, key_distinctions) = (
                self.parse_classification_validation_response(validation_response)
            )
            
            # Store validation history
            classification_history.append({
                "iteration": iteration,
                "validation_analysis": validation_analysis,
                "classification_correct": classification_correct,
                "original_classification": classification,
                "refined_classification": refined_classification,
                "confidence_change": new_confidence - confidence,
                "key_distinctions": key_distinctions
            })
            
            # Update current classification and confidence
            classification = refined_classification
            reasoning = refined_reasoning
            confidence = new_confidence
            
            logging.info(f"Classification iteration {iteration}: {classification} (confidence: {confidence})")
            
            iteration += 1
        
        return query_analysis, reasoning, feature_present, classification, confidence, classification_history

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
                
                # Iterative feature refinement
                final_feature, final_confidence, refinement_analysis, validation_history = (
                    self.iterative_feature_refinement(query)
                )
                
                # Iterative query classification using refined feature
                query_analysis, classification_reasoning, feature_present, classification, classification_confidence, classification_history = (
                    self.iterative_query_classification(query, final_feature, conversation_id)
                )

                logging.info(f'Answer: {classification}')
                logging.info(f'Final Feature: {final_feature}')
                logging.info(f'Feature Confidence: {final_confidence}')
                logging.info(f'Classification Confidence: {classification_confidence}')
                logging.info(f'Feature Present: {feature_present}')
                logging.info(f'Query Analysis: {query_analysis}\n')

                # Store comprehensive results
                query["distinguishing_feature"] = final_feature
                query["feature_confidence"] = final_confidence
                query["classification_confidence"] = classification_confidence
                query["refinement_analysis"] = refinement_analysis
                query["feature_validation_history"] = validation_history
                query["classification_history"] = classification_history
                query["feature_iterations_used"] = len(validation_history) + 1
                query["classification_iterations_used"] = len(classification_history)
                query["answer"] = classification
                query["query_details"] = query_analysis
                query["feature_present"] = feature_present
                query["classification_reasoning"] = classification_reasoning

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
        
        # Log iteration statistics
        if self.results:
            feature_iterations = [r.get("feature_iterations_used", 1) for r in self.results]
            classification_iterations = [r.get("classification_iterations_used", 1) for r in self.results]
            
            avg_feature_iterations = sum(feature_iterations) / len(feature_iterations)
            avg_classification_iterations = sum(classification_iterations) / len(classification_iterations)
            max_feature_iterations = max(feature_iterations)
            max_classification_iterations = max(classification_iterations)
            
            logging.info(f"Average feature iterations: {avg_feature_iterations:.2f}")
            logging.info(f"Average classification iterations: {avg_classification_iterations:.2f}")
            logging.info(f"Maximum feature iterations: {max_feature_iterations}")
            logging.info(f"Maximum classification iterations: {max_classification_iterations}")
            
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
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="maximum number of refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="confidence threshold to stop iteration (default: 0.8)",
    )
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\tIterative Refinement Reasoning:\tVLM: {args.vlm} \t Model: {args.llm}")
    logging.info(f"\tMax Iterations: {args.max_iterations}, Confidence Threshold: {args.confidence_threshold}")

    # Initialize the evaluator
    max_tokens = 4092
    temperature = 0
    caption_path = f"{constants.CAPTIONS_DIR}/{args.vlm}_scaptions.json"
    output_path = f"{constants.RESULTS_DIR}/irr_{args.vlm}_{args.llm}.json"
    logging.info(f"Caption path: {caption_path}")
    logging.info(f"Output path: {output_path}")

    evaluator = BongardIRR(
        args.vlm, args.llm, max_tokens, temperature, caption_path, output_path,
        args.max_iterations, args.confidence_threshold
    )
    evaluator.run()