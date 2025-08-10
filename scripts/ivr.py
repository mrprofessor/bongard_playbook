#!/usr/bin/env python3
import re
import json
import copy
import random
import argparse
import textwrap
import logging
from typing import List, Tuple, Dict, Any

import constants
import utils


class BongardIVR:
    """Iterative Visual Reasoning - Single Loop Visual Analysis with Refinement"""

    def __init__(
        self,
        model_name: str,
        temperature: int,
        max_tokens: float,
        dataset_path: str,
        output_path: str,
        max_iterations: int = 3,
        confidence_threshold: float = 0.8,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.model_id = constants.AIO_MODELS[model_name]
        self.client = utils.create_client(model_name)
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

        self.results = []
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}

    def load_dataset(self) -> List[dict]:
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        random.shuffle(data)
        return data

    @staticmethod
    def generate_initial_prompt() -> str:
        """Stage 1: Initial visual analysis with confidence rating"""
        return textwrap.dedent(
            """\
            You are provided with 13 images: the first 6 images are `group_a`, the next 6 images are `group_b`, and the last image is the `query_image`.

            Your task is to:
            1. Identify the visual feature or pattern shared among `group_a` images that clearly distinguishes them from `group_b`.
            2. Analyze the `query_image` for the presence or absence of this feature.
            3. Classify the `query_image` as belonging to either `group_a` or `group_b`.
            4. Rate your overall confidence in this analysis (0.0 to 1.0).

            IMPORTANT:
            - The distinguishing feature should be something consistent across all `group_a` images, even if it appears occasionally in `group_b`.
            - Look carefully at ALL images before making your decision.
            - Be honest about your confidence level.
            - Return ONLY a valid JSON block, formatted exactly like this:

            ```
            {
                "analysis": "Brief analysis comparing group_a and group_b examples",
                "distinguishing_feature": "The key visual feature that distinguishes group_a from group_b",
                "query_analysis": "What you observe in the query image",
                "classification": "group_a" or "group_b",
                "confidence": 0.85,
                "reasoning": "Clear reasoning for your classification and confidence level"
            }
            ```
            """
        )

    @staticmethod
    def generate_refinement_prompt(
        previous_analysis: str,
        previous_feature: str,
        previous_classification: str,
        previous_confidence: float,
        iteration: int
    ) -> str:
        """Stage 2-N: Iterative refinement prompt"""
        return textwrap.dedent(
            f"""\
            This is iteration {iteration} of your visual analysis. You previously concluded:

            PREVIOUS ANALYSIS: {previous_analysis}
            DISTINGUISHING FEATURE: {previous_feature}
            CLASSIFICATION: {previous_classification}
            CONFIDENCE: {previous_confidence}

            Now, let's take a more critical look at ALL the images again:

            Your task is to:
            1. Re-examine ALL 13 images with fresh eyes
            2. Challenge your previous analysis - look for counterexamples or better patterns
            3. Pay special attention to images that might not fit your previous pattern
            4. Re-assess the query image classification
            5. Update your analysis if needed

            CRITICAL QUESTIONS TO ASK YOURSELF:
            - Looking at ALL group_a images, is this pattern truly consistent?
            - Are there group_a images that don't fit my identified pattern?
            - Do any group_b images accidentally match my pattern?
            - Does the query image show the EXACT same visual pattern as group_a?
            - Could this be visually similar but fundamentally different?
            - Is there a more fundamental visual difference I'm missing?
            - What would change my mind about this classification?

            IMPORTANT:
            - Be highly critical of your previous analysis
            - If you find issues, refine your understanding
            - If you're more confident, explain why
            - Look for subtle visual distinctions you might have missed
            - Return ONLY a valid JSON block, formatted exactly like this:

            ```
            {{
                "analysis": "Updated analysis after critical re-examination",
                "distinguishing_feature": "Refined or confirmed distinguishing feature",
                "query_analysis": "Updated analysis of what you observe in the query image",
                "classification": "group_a" or "group_b",
                "confidence": 0.90,
                "reasoning": "Explanation of what changed (or didn't) and why",
                "iteration_notes": "What this iteration revealed or confirmed"
            }}
            ```
            """
        )

    def generate_visual_prompt(
        self, positive_images: List[str], negative_images: List[str], test_image: str, text_prompt: str
    ) -> List[dict]:
        """Generate visual prompt with all images"""
        prompt = []
        prompt.append({"type": "text", "text": text_prompt})

        # Add group_a images
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

        # Add group_b images
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

        # Add query image
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

    def query_model_with_context(self, content: List[dict], conversation_id: str = None) -> str:
        """Query model with optional conversation context"""
        if conversation_id and conversation_id in self.conversation_history:
            messages = self.conversation_history[conversation_id].copy()
        else:
            messages = []
        
        messages.append({"role": "user", "content": content})
        
        try:
            completion_params = utils.get_completion_params(
                self.model_name, self.max_tokens, self.temperature
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

    @staticmethod
    def parse_response(response_text: str) -> Tuple[str, str, str, str, float, str, str]:
        """Parse response to extract all fields"""
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
            query_analysis = extract_field("query_analysis")
            conclusion = extract_field("classification")
            confidence = extract_float("confidence")
            reasoning = extract_field("reasoning")
            iteration_notes = extract_field("iteration_notes")

            # Normalize classification
            classification = conclusion.lower()
            if "group_a" in classification:
                classification = "positive"
            elif "group_b" in classification:
                classification = "negative"
            else:
                classification = "unknown"

            return analysis, distinguishing_feature, query_analysis, classification, confidence, reasoning, iteration_notes

        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return "", "", "", "unknown", 0.0, "", ""

    def iterative_visual_analysis(self, record: dict) -> Tuple[str, str, str, str, float, str, List[dict]]:
        """Perform iterative visual analysis with single integrated loop"""
        conversation_id = record['uid']
        
        # Initial analysis
        initial_prompt_text = self.generate_initial_prompt()
        initial_prompt = self.generate_visual_prompt(
            record["positive_set"], record["negative_set"], record["query_image"], initial_prompt_text
        )
        
        initial_response = self.query_model_with_context(initial_prompt, conversation_id)
        
        analysis, distinguishing_feature, query_analysis, classification, confidence, reasoning, _ = (
            self.parse_response(initial_response)
        )
        
        logging.info(f"Initial analysis: {classification} (confidence: {confidence})")
        logging.info(f"Initial feature: {distinguishing_feature}")
        
        iteration_history = [{
            "iteration": 0,
            "analysis": analysis,
            "distinguishing_feature": distinguishing_feature,
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning
        }]
        
        iteration = 1
        
        # Iterative refinement loop
        while iteration <= self.max_iterations and confidence < self.confidence_threshold:
            logging.info(f"Visual analysis iteration {iteration}/{self.max_iterations}")
            
            # Generate refinement prompt (text only - images stay in conversation context)
            refinement_prompt_text = self.generate_refinement_prompt(
                analysis, distinguishing_feature, classification, confidence, iteration
            )
            
            # Send only text prompt (images are already in conversation context)
            refinement_response = self.query_model_with_context(
                [{"type": "text", "text": refinement_prompt_text}], conversation_id
            )
            
            # Parse updated response
            new_analysis, new_feature, new_query_analysis, new_classification, new_confidence, new_reasoning, iteration_notes = (
                self.parse_response(refinement_response)
            )
            
            # Store iteration history
            iteration_history.append({
                "iteration": iteration,
                "analysis": new_analysis,
                "distinguishing_feature": new_feature,
                "classification": new_classification,
                "confidence": new_confidence,
                "confidence_change": new_confidence - confidence,
                "reasoning": new_reasoning,
                "iteration_notes": iteration_notes
            })
            
            # Update current state
            analysis = new_analysis
            distinguishing_feature = new_feature
            query_analysis = new_query_analysis
            classification = new_classification
            confidence = new_confidence
            reasoning = new_reasoning
            
            logging.info(f"Iteration {iteration}: {classification} (confidence: {confidence})")
            
            iteration += 1
        
        final_reasoning = f"Converged after {iteration} iterations. Final confidence: {confidence}"
        
        return analysis, distinguishing_feature, query_analysis, classification, confidence, final_reasoning, iteration_history

    def evaluate(self):
        data = self.load_dataset()
        for idx, record in enumerate(data):
            logging.info(f"Processing query {record['uid']} -- {idx + 1}/{len(data)}")
            logging.info("=" * 100)

            try:
                # Iterative visual analysis
                analysis, distinguishing_feature, query_analysis, classification, final_confidence, final_reasoning, iteration_history = (
                    self.iterative_visual_analysis(record)
                )

                logging.info(f'Final Answer: {classification}')
                logging.info(f'Distinguishing Feature: {distinguishing_feature}')
                logging.info(f'Final Confidence: {final_confidence}')
                logging.info(f'Query Analysis: {query_analysis}\n')

                # Store comprehensive results
                self.results.append(
                    {
                        "uid": record["uid"],
                        "commonSense": record["commonSense"],
                        "concept": record["concept"],
                        "caption": record["caption"],
                        "query_image": record["query_image"],
                        "answer": classification,
                        "distinguishing_feature": distinguishing_feature,
                        "analysis": analysis,
                        "query_details": query_analysis,
                        "final_confidence": final_confidence,
                        "final_reasoning": final_reasoning,
                        "iteration_history": iteration_history,
                        "iterations_used": len(iteration_history),
                        "raw_response": f"Final confidence: {final_confidence}"
                    }
                )

                # Clean up conversation history to save memory
                conversation_id = record['uid']
                if conversation_id in self.conversation_history:
                    del self.conversation_history[conversation_id]

            except Exception as e:
                logging.error(f"Error processing query {record['uid']}: {e}")

    def save_results(self):
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Results saved to: {self.output_path}")
        
        # Log iteration statistics
        if self.results:
            iterations_used = [r.get("iterations_used", 1) for r in self.results]
            avg_iterations = sum(iterations_used) / len(iterations_used)
            max_iterations_used = max(iterations_used)
            
            logging.info(f"Average iterations used: {avg_iterations:.2f}")
            logging.info(f"Maximum iterations used: {max_iterations_used}")

    def run(self):
        self.evaluate()
        self.save_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=constants.AIO_MODELS,
        help="Model name (e.g., gpt41, gemini20, etc.)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of refinement iterations (default: 3)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.8,
        help="Confidence threshold to stop iteration (default: 0.8)",
    )
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\tIterative Visual Reasoning:\tModel: {args.model}")
    logging.info(f"\tMax Iterations: {args.max_iterations}, Confidence Threshold: {args.confidence_threshold}")

    # Initialize the evaluator
    model_name = args.model
    dataset_path = constants.TRANSFORMED_DATASET
    output_path = f"{constants.RESULTS_DIR}/ivr_{args.model}.json"
    max_tokens = 4092
    temperature = 0
    
    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Output path: {output_path}")

    evaluator = BongardIVR(
        model_name, temperature, max_tokens, dataset_path, output_path,
        args.max_iterations, args.confidence_threshold
    )
    evaluator.run()