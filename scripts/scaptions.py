import os
import json
import argparse
import logging
import re
import constants
import utils

class BongardStructuredCaptioner:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        temperature: float,
        dataset_path: str,
        output_path: str,
    ):
        self.model = model
        self.client = utils.create_client(model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.output_path = output_path
        self.dataset_path = dataset_path

    def captioning_prompt(self):
        return """Analyze this image and describe everything visible in the following JSON format only.

        Describe only what is directly visible in the image with no inferences, assumptions, or speculation.

        {
            "scene": "Overall setting and environment (urban/natural/forestside/riverbanks etc.)",
            "objects": {
                "living": "People, animals, plants with their key characteristics",
                "non_living": "Inanimate objects, structures, vehicles, tools, etc."
            },
            "activities": "Observable actions, interactions, and behaviors, such as playing a sport, painting, fishing in a river, dancing on the stage etc.",
            "perspective_viewpoint": "Camera angle and viewpoint (aerial, ground level, close-up, wide shot, eye level, bird's eye, etc.)",
            "spatial_layout": "Positioning and relationships - what's in foreground/background, left/right, center, relative sizes and positioning",
            "quantities_and_scale": {
                "object_counts": "Specific counts where relevant and countable (3 people, dozens of cars, few trees, many birds, etc.)",
                "relative_sizes": "Size relationships between objects and their prominence in the image",
                "crowd_density": "For gatherings - sparse, moderate, crowded, packed, or individual presence"
            },
            "motion_and_dynamics": {
                "motion_evidence": "Signs of movement - motion blur, frozen action, static positioning, dynamic poses",
                "energy_level": "Overall sense of activity - calm, active, energetic, chaotic, peaceful, intense"
            },
            "textual_information": "Any visible text, signs, labels, writing with style/formatting and location",
            "visual_patterns": "Colors, shapes, textures, patterns, materials, brand logos, unique garments, or any distinctive features that stand out",
            "contextual_factors": "Weather conditions, season indicators, environmental clues, lighting conditions, time of day, shadows, reflections",
            "emotional_undertones": "Facial expressions, body language, overall mood",
            "summary": "Single descriptive sentence capturing the complete essence of the image"
        }

        Requirements:
        - Describe only what is directly visible
        - Use clear, specific language
        - Avoid duplicating details across JSON fields
        - Ensure valid JSON formatting
        - Be comprehensive but concise in each field"""
    
    def parse_captioner_response(self, response_text: str) -> dict:
        """Parse the response, first trying JSON extraction, then falling back to regex parsing."""
        response_text = response_text.strip()

        # Try to extract and parse JSON from ```json blocks first
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            print("JSON MATCHES")
            try:
                return json.loads(json_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Fallback: regex parsing
        nested_fields = {"objects", "quantities_and_scale", "motion_and_dynamics"}
        all_fields = ["scene", "objects", "activities", "perspective_viewpoint", "spatial_layout",
                    "quantities_and_scale", "motion_and_dynamics", "textual_information", 
                    "visual_patterns", "contextual_factors", "emotional_undertones", "summary"]
        
        def clean_value(value):
            """Clean extracted value by removing newlines and extra whitespace."""
            return re.sub(r'\s+', ' ', re.sub(r'\\n', ' ', value.strip()))
        
        def try_parse_json(value, field):
            """Try to parse value as JSON if it's a nested field."""
            if field in nested_fields:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    pass
            return value

        result = {}
        for field in all_fields:
            # Try main pattern, then fallback pattern
            patterns = [
                rf'"{field}":\s*(\{{[^}}]*\}})' if field in nested_fields else rf'"{field}":\s*"([^"]*)"',
                rf'"{field}":\s*([^,}}]*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = clean_value(re.sub(r'^["\s]*|[",\s]*$', '', match.group(1)))
                    result[field] = try_parse_json(value, field)
                    break
            else:
                result[field] = ""

        return result if any(result.values()) else {field: "Parse failed" for field in all_fields}


    def process(self):
        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        results = []
        for idx, record in enumerate(data):
            logging.info(f"Processing query {idx + 1}/{len(data)} ({(idx + 1) / len(data):.2%})")
            logging.info("=" * 100)

            captions = []
            for image_path in record["imageFiles"]:
                image_file_path = os.path.join(constants.DATA_DIR, image_path)
                image = utils.encode_image(image_file_path)

                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": self.captioning_prompt()},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image}"
                                    },
                                },
                            ],
                        }
                    ]

                    completion_params = utils.get_completion_params(
                        self.model, self.max_tokens, self.temperature
                    )
                    response = self.client.chat.completions.create(
                        model=constants.AIO_MODELS.get(self.model),
                        messages=messages,
                        **completion_params,
                    )

                    caption_text = response.choices[0].message.content
                    logging.info("====")
                    logging.info(image_path)
                    logging.info(caption_text)

                    # Parse the hierarchical response
                    parsed_caption = self.parse_captioner_response(caption_text)
                    captions.append(parsed_caption)

                except Exception as e:
                    logging.error(f"Error generating caption: {e}")
                    # Create error response with all fields
                    error_response = {field: "ERROR" for field in [
                        "scene", "objects", "activities", "perspective_viewpoint", "spatial_layout",
                        "quantities_and_scale", "motion_and_dynamics", "textual_information",
                        "visual_patterns", "contextual_factors", "emotional_undertones", "summary"
                    ]}
                    error_response["summary"] = str(e)
                    captions.append(error_response)

            record["captions"] = captions
            results.append(record)

        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=constants.AIO_MODELS.keys(),
        help="Choose a caption model",
    )
    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Initialize parameters
    max_tokens = 4096
    temperature = 0
    dataset_path = constants.TEST_DATASET
    output_path = os.path.join(constants.CAPTIONS_DIR, f"{args.model}_scaptions.json")

    # Initialize captioner
    captioner = BongardStructuredCaptioner(
        args.model, max_tokens, temperature, dataset_path, output_path
    )
    captioner.process()

if __name__ == "__main__":
    main()
