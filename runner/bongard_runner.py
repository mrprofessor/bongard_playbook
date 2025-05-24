import os
import json
import base64
import argparse
from PIL import Image
import requests
from io import BytesIO
import csv
from datetime import datetime
from bongard_canvas import BongardCanvas
from bongard_accuracy import BongardAccuracy


# Configuration
OLLAMA_API = "http://127.0.0.1:8080/api/generate"  # Update to your port
DEFAULT_MODEL = "gemma3:latest"  # Default set to Gemma 3
SAVE_DEBUG_IMAGES = True
DEBUG_DIR = "debug_images"
TEST_FILE = 'assets/data/bongard-ow/bongard_ow_test.json'
DATA_DIR = 'assets/data/bongard-ow'


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Bongard problem evaluation with configurable model')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL,
                        help=f'Ollama model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--no-debug', action='store_true',
                        help='Disable saving debug images')

    # If no args provided, use defaults
    import sys
    if len(sys.argv) == 1:
        print("No command line arguments provided. Using defaults:")
        print(f"  Model: {DEFAULT_MODEL}")
        print("  Debug images: Enabled")
        print("-" * 50)

    return parser.parse_args()


def prompt_generator() -> str:
    """Generate appropriate prompt based on layout type"""
    return """This is a Bongard problem presented as a single image with three rows:
- TOP ROW: 6 positive examples (512x512 each) that share a common concept
- MIDDLE ROW: 6 negative examples (512x512 each) that don't have this concept
- BOTTOM ROW: One query image

You are looking at ONE SINGLE IMAGE that contains all these elements arranged in rows.
The bottom image is the query image, please analyze it.

Based on the visual patterns:
1. Identify what distinguishes positive from negative examples
2. Determine if the query image has this feature

Answer in this format:
Category: [positive/negative]
Common concept: [explanation]"""


def resize_image(image_path: str, max_size: int = 512) -> Image.Image:
    """Resize image to prevent OOM errors"""
    img = Image.open(image_path)
    img.thumbnail((max_size, max_size))
    return img


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def main():
    # Parse command line arguments
    args = parse_arguments()
    model = args.model if hasattr(args, 'model') else DEFAULT_MODEL
    save_debug_images = not args.no_debug if hasattr(args, 'no_debug') else True

    print(f"Using model: {model}")
    print(f"Saving debug images: {'Yes' if save_debug_images else 'No'}")

    # Create results file with model name in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = model.replace(":", "_").replace("/", "_")
    results_file = f"bongard_results_{model_name_safe}_{timestamp}.csv"

    # Create debug directory if needed
    if save_debug_images:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # Load the Bongard test data
    with open(TEST_FILE, 'r') as f:
        bongard_ow_test = json.load(f)

    # Initialize the canvas creator
    canvas_creator = BongardCanvas(
        grid_size=(512, 512),  # You can adjust these parameters
        padding=20,
        divider_height=3,
        divider_gap=30
    )

    # Open CSV file for writing
    with open(results_file, 'w', newline='') as csvfile:
        fieldnames = ['uid', 'query_id', 'predicted_answer', 'concept',
                     'common_sense', 'caption', 'model',
                     'model_full_response', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process all samples
        for idx, sample in enumerate(bongard_ow_test):
            uid = sample['uid']
            commonSense = sample['commonSense']
            concept = sample['concept']
            caption = sample['caption']

            print(f"\nProcessing Bongard problem {idx+1}/{len(bongard_ow_test)}: {uid}")
            print(f"Common sense: {commonSense}")
            print(f"Concept: {concept}")
            print(f"Caption: {caption}")
            print("-" * 50)

            # Get image file paths
            imageFiles = [os.path.join(DATA_DIR, imageFile)
                         for imageFile in sample['imageFiles']]

            # Process each query - moved the try-catch inside this loop
            for query_id, query_idx in [('A', 6), ('B', 13)]:
                try:
                    row_data = {
                        'uid': uid,
                        'query_id': query_id,
                        'concept': concept,
                        'common_sense': commonSense,
                        'caption': caption,
                        'error': None
                    }

                    # Create single query canvas
                    canvas = canvas_creator.create_single_query_layout(imageFiles, query_idx)
                    prompt = prompt_generator()  # Get the prompt

                    # Save debug image if enabled
                    if save_debug_images:
                        debug_path = os.path.join(DEBUG_DIR, f"canvas_{uid}_query_{query_id}.png")
                        canvas.save(debug_path)
                        print(f"Debug: Saved canvas to {debug_path}")

                    # Convert to base64
                    img_base64 = image_to_base64(canvas)

                    print(f"Processing query {query_id}...")

                    # Make request to Ollama
                    response = requests.post(OLLAMA_API,
                        json={
                            "model": model,
                            "prompt": prompt,
                            "images": [img_base64],
                            "stream": False
                        },
                        timeout=300  # 5 minute timeout
                    )

                    if response.status_code == 200:
                        result = response.json()['response']
                        print(f"\n{model}'s response for query {query_id}:")
                        print(result[:200] + "..." if len(result) > 200 else result)

                        # Parse the response
                        answer = "unknown"
                        response_lower = result.lower()
                        if "category: positive" in response_lower:
                            answer = "positive"
                        elif "category: negative" in response_lower:
                            answer = "negative"

                        row_data['predicted_answer'] = answer
                        row_data['model_full_response'] = result

                        print(f"Extracted answer: {answer}")
                    else:
                        error_msg = f"Error: {response.status_code} - {response.text}"
                        print(error_msg)
                        row_data['error'] = error_msg
                        row_data['predicted_answer'] = "error"
                        row_data['model_full_response'] = ""

                    # Write row to CSV
                    writer.writerow(row_data)
                    csvfile.flush()  # Ensure data is written immediately

                except Exception as e:
                    error_msg = f"Error processing sample {uid}, query {query_id}: {e}"
                    print(error_msg)
                    # Write error for this specific query
                    writer.writerow({
                        'uid': uid,
                        'query_id': query_id,
                        'concept': concept,
                        'common_sense': commonSense,
                        'caption': caption,
                        'predicted_answer': 'error',
                        'model_full_response': '',
                        'error': str(e)
                    })
                    csvfile.flush()  # Make sure error is written immediately

            print("-" * 50)

    print(f"\nResults saved to: {results_file}")

    try:
        evaluator = BongardAccuracy(results_file)
        evaluator.calculate_accuracy()
    except Exception as e:
        print(f"Error calculating accuracy: {e}")

if __name__ == '__main__':
    main()
