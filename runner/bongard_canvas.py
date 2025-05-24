from PIL import Image, ImageDraw


class BongardCanvas:
    """Class for creating Bongard problem canvas layouts"""

    def __init__(self, grid_size=(512, 512), padding=20, divider_height=3, divider_gap=30):
        self.grid_width, self.grid_height = grid_size
        self.padding = padding
        self.divider_height = divider_height
        self.divider_gap = divider_gap

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

    def resize_image_aspect(self, image_path, max_width=None, max_height=None):
        """Resize image preserving aspect ratio"""
        if max_width is None:
            max_width = self.grid_width
        if max_height is None:
            max_height = self.grid_height

        if isinstance(image_path, str):
            img = Image.open(image_path)
        else:  # Already a PIL Image
            img = image_path

        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return img

    def _create_base_canvas(self, image_files):
        """Create the base canvas with positive and negative examples"""
        # Resize grid images to fixed size
        grid_images = []
        for i in range(14):
            if i == 6 or i == 13:  # Skip query images
                continue
            img = self.resize_image_fixed(image_files[i])
            grid_images.append(img)

        # Canvas dimensions
        canvas_width = 6 * self.grid_width + 7 * self.padding
        canvas_height = 3 * self.grid_height + 4 * self.padding + 2 * (self.divider_height + self.divider_gap)

        canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
        draw = ImageDraw.Draw(canvas)

        # Row 1: Set 1 (images 0-5)
        row1_y = self.padding
        for i in range(6):
            x = self.padding + i * (self.grid_width + self.padding)
            canvas.paste(grid_images[i], (x, row1_y))

        # Horizontal divider after Set 1
        divider1_y = row1_y + self.grid_height + self.padding
        draw.line([(self.padding, divider1_y), (canvas_width - self.padding, divider1_y)],
                  fill='gray', width=self.divider_height)

        # Row 2: Set 2 (images 7-12, offset by 6 in grid_images)
        row2_y = divider1_y + self.divider_height + self.divider_gap
        for i in range(6):
            x = self.padding + i * (self.grid_width + self.padding)
            canvas.paste(grid_images[i + 6], (x, row2_y))

        # Horizontal divider after Set 2
        divider2_y = row2_y + self.grid_height + self.padding
        draw.line([(self.padding, divider2_y), (canvas_width - self.padding, divider2_y)],
                  fill='gray', width=self.divider_height)

        # Row 3 starting position
        row3_y = divider2_y + self.divider_height + self.divider_gap

        return canvas, draw, canvas_width, row3_y

    def create_layout(self, image_files):
        """Create a horizontal layout: Set 1 (top), Set 2 (middle), Both Queries (bottom)"""
        canvas, draw, canvas_width, row3_y = self._create_base_canvas(image_files)

        # Resize query images preserving aspect ratio
        query_a = self.resize_image_aspect(image_files[6])
        query_b = self.resize_image_aspect(image_files[13])

        # Calculate positions to center the two query images
        query_spacing = self.padding * 4  # Extra spacing between queries
        total_width = query_a.width + query_b.width + query_spacing
        start_x = (canvas_width - total_width) // 2

        # Query A - center vertically in the row
        query_a_x = start_x
        query_a_y = row3_y + (self.grid_height - query_a.height) // 2
        canvas.paste(query_a, (query_a_x, query_a_y))

        # Query B - center vertically in the row
        query_b_x = start_x + query_a.width + query_spacing
        query_b_y = row3_y + (self.grid_height - query_b.height) // 2
        canvas.paste(query_b, (query_b_x, query_b_y))

        return canvas

    def create_single_query_layout(self, image_files, query_idx):
        """Create a nice grid layout with only one query image"""
        canvas, draw, canvas_width, row3_y = self._create_base_canvas(image_files)

        # Resize the query image preserving aspect ratio
        query_image = self.resize_image_aspect(image_files[query_idx])

        # Center the query image
        query_x = (canvas_width - query_image.width) // 2
        query_y = row3_y + (self.grid_height - query_image.height) // 2
        canvas.paste(query_image, (query_x, query_y))

        return canvas
