import os
import json
import argparse
import logging

import constants
import utils


class BongardCaptioner:
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
        return "Write a brief, factual description of this image in 2-3 sentences. Do not include any introductory phrases."

    def parse_captioner_response(self, response_text: str) -> str:
        return response_text.strip()

    def process(self):
        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        results = []

        for idx, record in enumerate(data):
            logging.info(f"Processing query {record['uid']} -- {idx + 1}/{len(data)}")
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
                    caption = response.choices[0].message.content
                    logging.info("====")
                    logging.info(image_path)
                    logging.info(caption)
                    captions.append(self.parse_captioner_response(caption))

                except Exception as e:
                    logging.error(f"Error generating caption: {e}")
                    captions.append("ERROR")

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
    output_path = os.path.join(constants.CAPTIONS_DIR, f"{args.model}_captions.json")

    # Initialize captioner
    captioner = BongardCaptioner(
        args.model, max_tokens, temperature, dataset_path, output_path
    )
    captioner.process()


if __name__ == "__main__":
    main()
