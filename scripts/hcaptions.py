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
        return """Describe this image in three levels and format your response as valid JSON:
            Describe only what is directly visible in the image with no inferences, assumptions, or speculation.
            {
                "scene": "What type of environment/setting is this?",
                "objects": "What are the main objects, people, or elements present?",
                "action": "What actions or interactions are happening?",
                "summary": "One sentence summary highlighting the most distinctive aspect"
            }
        """

    def parse_captioner_response(self, response_text: str) -> dict:
        """Parse the response using string matching instead of JSON parsing."""
        response_text = response_text.strip()

        # Initialize result dictionary
        result = {
            "scene": "",
            "objects": "",
            "action": "",
            "summary": ""
        }

        # Define patterns to extract each field
        patterns = {
            "scene": r'"scene":\s*"([^"]*)"',
            "objects": r'"objects":\s*"([^"]*)"',
            "action": r'"action":\s*"([^"]*)"',
            "summary": r'"summary":\s*"([^"]*)"'
        }

        # Extract each field using regex
        for field, pattern in patterns.items():
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                result[field] = match.group(1).strip()
            else:
                # Fallback: try to find the field without quotes around the value
                fallback_pattern = f'"{field}":\\s*([^,}}]*)'
                fallback_match = re.search(fallback_pattern, response_text, re.IGNORECASE | re.DOTALL)
                if fallback_match:
                    # Clean up the extracted value
                    value = fallback_match.group(1).strip()
                    # Remove trailing quotes, commas, and whitespace
                    value = re.sub(r'[",\s]*$', '', value)
                    # Remove leading quotes
                    value = re.sub(r'^["]*', '', value)
                    result[field] = value

        # If parsing fails completely, try to extract any meaningful content
        if not any(result.values()):
            # As a last resort, return the raw response
            result = {
                "scene": "Parse failed",
                "objects": "Parse failed",
                "action": "Parse failed",
                "summary": response_text[:200] + "..." if len(response_text) > 200 else response_text
            }

        return result

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
                    captions.append({
                        "scene": "ERROR",
                        "objects": "ERROR",
                        "action": "ERROR",
                        "summary": str(e)
                    })


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
    max_tokens = 512
    temperature = 0
    dataset_path = constants.TEST_DATASET
    output_path = os.path.join(constants.CAPTIONS_DIR, f"{args.model}_structured.json")

    # Initialize captioner
    captioner = BongardStructuredCaptioner(
        args.model, max_tokens, temperature, dataset_path, output_path
    )
    captioner.process()

if __name__ == "__main__":
    main()
