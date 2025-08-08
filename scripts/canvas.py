import os
import json
import argparse
import logging
from PIL import Image
import constants

class BongardCanvas:
    """Class for creating Bongard problem canvas layouts"""

    def __init__(self, grid_size=(220, 220), padding=25):
        self.grid_width, self.grid_height = grid_size
        self.padding = padding

    def resize_image_fixed(self, image_path, target_size=None):
        """Resize image to exact target size for grid images"""
        if target_size is None:
            target_size = (self.grid_width, self.grid_height)

        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:  # Already a PIL Image
            img = image_path

        img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img

    def create_horizontal_layout(self, image_files, query_idx):
        """Create horizontal layout: Positive Set | Negative Set | Query"""

        # Section dimensions
        section_width = 3 * self.grid_width + 4 * self.padding  # 3 images + padding
        section_height = 2 * self.grid_height + 3 * self.padding  # 2 rows + padding
        query_section_width = self.grid_width + 2 * self.padding  # Just enough for one image

        # Total canvas dimensions
        canvas_width = section_width + section_width + query_section_width + 4 * self.padding
        canvas_height = section_height + 2 * self.padding

        # Create canvas with transparent background
        canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))

        # POSITIVE SET (left section) - images 0-5
        pos_start_x = self.padding
        pos_start_y = self.padding

        for i in range(6):
            row = i // 3
            col = i % 3
            x = pos_start_x + col * (self.grid_width + self.padding)
            y = pos_start_y + row * (self.grid_height + self.padding)

            img = self.resize_image_fixed(image_files[i])
            # Convert to RGBA if needed to maintain transparency
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            canvas.paste(img, (x, y), img)  # Use img as mask to preserve transparency

        # NEGATIVE SET (middle section) - images 7-12
        neg_start_x = pos_start_x + section_width + self.padding
        neg_start_y = self.padding

        for i in range(6):
            row = i // 3
            col = i % 3
            x = neg_start_x + col * (self.grid_width + self.padding)
            y = neg_start_y + row * (self.grid_height + self.padding)

            img_idx = i + 7  # Negative set starts at index 7
            if img_idx < len(image_files):
                img = self.resize_image_fixed(image_files[img_idx])
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                canvas.paste(img, (x, y), img)

        # QUERY (right section) - same size as all other images
        query_start_x = neg_start_x + section_width + self.padding
        query_image = self.resize_image_fixed(image_files[query_idx])  # Same size as others!

        # Center the query image vertically in its section
        query_x = query_start_x + self.padding
        query_y = self.padding + (section_height - self.grid_height) // 2

        if query_image.mode != 'RGBA':
            query_image = query_image.convert('RGBA')
        canvas.paste(query_image, (query_x, query_y), query_image)

        return canvas


class BongardVisualizer:
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.canvas = BongardCanvas()

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def create_visualizations(self):
        """Create Problem A and B visualizations for all records"""

        with open(self.dataset_path, "r") as f:
            data = json.load(f)

        for idx, record in enumerate(data):
            # if record['uid'] !=  '0348':
            #     continue
            logging.info(f"Processing {record['uid']} -- {idx + 1}/{len(data)}")

            # Get image file paths
            image_files = []
            for image_path in record["imageFiles"]:
                image_file_path = os.path.join(constants.DATA_DIR, image_path)
                image_files.append(image_file_path)

            try:
                # Create Problem A visualization (positive query - index 6)
                canvas_a = self.canvas.create_horizontal_layout(image_files, query_idx=6)
                problem_a_path = os.path.join(self.output_dir, f"{record['uid']}_problem_A.png")
                canvas_a.save(problem_a_path)
                logging.info(f"Saved: {problem_a_path}")

                # Create Problem B visualization (negative query - index 13)
                canvas_b = self.canvas.create_horizontal_layout(image_files, query_idx=13)
                problem_b_path = os.path.join(self.output_dir, f"{record['uid']}_problem_B.png")
                canvas_b.save(problem_b_path)
                logging.info(f"Saved: {problem_b_path}")

            except Exception as e:
                logging.error(f"Error creating visualizations for {record['uid']}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate Bongard problem visualizations")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=constants.TEST_DATASET,
        help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./bongard_visualizations",
        help="Output directory for visualizations"
    )

    args = parser.parse_args()

    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create visualizer and run
    visualizer = BongardVisualizer(args.dataset_path, args.output_dir)
    visualizer.create_visualizations()

    logging.info(f"All visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
