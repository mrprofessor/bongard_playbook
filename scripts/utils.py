import re
import os
import io
import sys
import base64
from PIL import Image
from openai import OpenAI

def create_client(model):
    base_url = 'http://localhost:11434/v1'
    api_key = 'ollama'

    if model == "gpt41":
        base_url = 'https://api.openai.com/v1'
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is not set")
            sys.exit(1)

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client

def encode_image(image_path: str) -> str:
    """Safely convert image to base64-encoded JPEG for model input"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure RGB format
            img.thumbnail((512, 512))  # Optional: resize to reduce payload
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to process image '{image_path}': {e}")

def parse_model_response(response: str) -> tuple:
    """
    Extract classification (positive/negative) and reason from LLM response.
    If the reason is not found, returns the entire text as the sentence.
    Args:
        response (str): Raw response text from LLM
    Returns:
        tuple: (classification, common_pattern, reasoning) as strings
    """
    classification_match = re.search(r'"classification":\s*"(positive|negative)"', response, re.IGNORECASE | re.DOTALL)
    common_pattern_match = re.search(r'"pattern":\s*"(.*?)"', response, re.IGNORECASE | re.DOTALL)
    reason_match = re.search(r'"reasoning":\s*"(.*?)"', response, re.IGNORECASE | re.DOTALL)

    answer = classification_match.group(1).lower() if classification_match else "unknown"
    common_pattern = common_pattern_match.group(1).strip() if common_pattern_match else "Error"
    reasoning = reason_match.group(1).strip() if reason_match else response.strip()

    return answer, common_pattern, reasoning
