"""
Image processing and G-code generation module for MuralBot.
"""

import os
import cv2
import numpy as np
from PIL import Image
import math
import time
from collections import defaultdict

class ImageProcessor:
    """Process images and generate G-code for the mural painting robot."""
    
    def __init__(self):
        # Default configuration
        self.config = {
            "wall_width": 2.0,         # meters
            "wall_height": 2.0,         # meters
            "total_colors": 8,          # number of colors
            "colors_per_batch": 4,      # colors per batch
            "dot_resolution": 20.0,     # mm
            "speed_xy": 1000,           # mm/min
            "speed_z": 500,             # mm/min
            "spray_delay": 0.2,         # seconds
            "dithering_method": "Floyd-Steinberg",
            "optimize_path": True,
            "border_margin": 50.0,      # mm
            "home_position": "Bottom Left"
        }
    
    def set_config(self, config):
        """Update the configuration."""
        self.config.update(config)
    
    def process_image(self, image_path, preview_only=False):
        """
        Process the image for mural painting.
        
        Args:
            image_path: Path to the input image
            preview_only: If True, only process for preview, not for G-code generation
            
        Returns:
            processed_image: The quantized image
            palette: List of RGB colors in the palette
            batches: List of batch numbers for each color
            color_counts: Count of each color in the image
        """
        # Load the image
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Calculate dimensions based on wall size and resolution
        wall_width_mm = self.config["wall_width"] * 1000  # convert to mm
        wall_height_mm = self.config["wall_height"] * 1000  # convert to mm
        dot_size_mm = self.config["dot_resolution"]
        
        # Calculate number of dots in each dimension
        dots_width = int(wall_width_mm / dot_size_mm)
        dots_height = int(wall_height_mm / dot_size_mm)
        
        # Resize image to match the number of dots
        resized_image = cv2.resize(original_image, (dots_width, dots_height), interpolation=cv2.INTER_AREA)
        
        # Generate color palette (quantize colors)
        num_colors = self.config["total_colors"]
        palette_image = resized_image.reshape(-1, 3)
        
        # Use k-means clustering to find the most representative colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, labels, palette = cv2.kmeans(
            np.float32(palette_image), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Convert palette to uint8
        palette = np.uint8(palette)
        
        # Apply dithering if selected
        if self.config["dithering_method"] == "Floyd-Steinberg":
            quantized_image = self._apply_floyd_steinberg_dithering(resized_image, palette)
        elif self.config["dithering_method"] == "Ordered":
            quantized_image = self._apply_ordered_dithering(resized_image, palette)
        else:
            # No dithering - just use nearest color
            quantized_image = palette[labels.flatten()].reshape(resized_image.shape)
        
        # Count color occurrences
        color_counts = defaultdict(int)
        for i in range(len(labels)):
            color_idx = labels[i]
            color_counts[int(color_idx)] += 1
        
        # Sort palette by usage frequency
        palette_with_counts = [(palette[i], color_counts[i]) for i in range(num_colors)]
        palette_with_counts.sort(key=lambda x: x[1], reverse=True)
        
        sorted_palette = [color for color, _ in palette_with_counts]
        sorted_counts = [count for _, count in palette_with_counts]
        
        # Assign colors to batches
        colors_per_batch = self.config["colors_per_batch"]
        batches = []
        for i in range(len(sorted_palette)):
            batch_num = i // colors_per_batch + 1
            batches.append(batch_num)
        
        if preview_only:
            return quantized_image, sorted_palette, batches, sorted_counts
        
        # For actual G-code generation, we need to do more processing
        # (implemented in generate_gcode method)
        return quantized_image, sorted_palette, batches, sorted_counts
    
    def generate_gcode(self, image_path, progress_callback=None):
        """
        Generate G-code for the mural painting robot.
        
        Args:
            image_path: Path to the input image
            progress_callback: Function to call with progress updates (0-100)
            
        Returns:
            gcode: String containing the G-code
            output_info: Dictionary containing information about the output
        """
        # Process the image
        quantized_image, palette, batches, color_counts = self.process_image(image_path)
        
        # Report progress
        if progress_callback:
            progress_callback(10)
        
        # Extract configuration values
        wall_width_mm = self.config["wall_width"] * 1000  # convert to mm
        wall_height_mm = self.config["wall_height"] * 1000  # convert to mm
        dot_size_mm = self.config["dot_resolution"]
        speed_xy = self.config["speed_xy"]
        speed_z = self.config["speed_z"]
        spray_delay = self.config["spray_delay"]
        border_margin = self.config["border_margin"]
        home_position = self.config["home_position"]
        
        # Calculate home coordinates based on home position setting
        if home_position == "Bottom Left":
            home_x, home_y = border_margin, border_margin
        elif home_position == "Bottom Right":
            home_x, home_y = wall_width_mm - border_margin, border_margin
        elif home_position == "Top Left":
            home_x, home_y = border_margin, wall_height_mm - border_margin
        elif home_position == "Top Right":
            home_x, home_y = wall_width_mm - border_margin, wall_height_mm - border_margin
        
        # Initialize G-code
        gcode = []
        gcode.append("; MuralBot G-code")
        gcode.append(f"; Generated from: {os.path.basename(image_path)}")
        gcode.append(f"; Wall dimensions: {self.config['wall_width']}m x {self.config['wall_height']}m")
        gcode.append(f"; Dot resolution: {dot_size_mm} mm")
        gcode.append(f"; Total colors: {len(palette)}")
        gcode.append(f"; Colors per batch: {self.config['colors_per_batch']}")
        gcode.append(f"; Palette: {[(r, g, b) for r, g, b in palette]}")
        gcode.append("; Color batches:")
        for i, (color, batch) in enumerate(zip(palette, batches)):
            gcode.append(f";   Color {i+1}: RGB{tuple(color)} - Batch {batch}")
        gcode.append("")
        
        # G-code preamble
        gcode.append("G21 ; Set units to millimeters")
        gcode.append("G90 ; Absolute positioning")
        gcode.append("G28 ; Home all axes")
        gcode.append(f"G0 F{speed_xy} ; Set travel speed")
        gcode.append("")
        
        # Generate commands for each batch
        total_color_batches = max(batches)
        total_dots = 0
        
        # For progress tracking
        progress_base = 10
        progress_per_batch = 90 / total_color_batches
        
        for batch_num in range(1, total_color_batches + 1):
            # Get colors in this batch
            batch_color_indices = [i for i, b in enumerate(batches) if b == batch_num]
            
            if not batch_color_indices:
                continue
            
            gcode.append(f"; === Batch {batch_num} ===")
            gcode.append(f"G0 X{home_x} Y{home_y} ; Move to home for color change")
            gcode.append("M0 ; Pause for color change")
            gcode.append("")
            
            # Process each color in the batch
            for color_idx in batch_color_indices:
                color = palette[color_idx]
                gcode.append(f"; Color: RGB{tuple(color)}")
                
                # Set the active tool (spray can)
                spray_idx = batch_color_indices.index(color_idx)
                gcode.append(f"T{spray_idx} ; Select spray can {spray_idx + 1}")
                
                # Find all dots of this color
                height, width, _ = quantized_image.shape
                dots = []
                
                for y in range(height):
                    for x in range(width):
                        pixel = quantized_image[y, x]
                        # Check if this pixel matches current color
                        if np.array_equal(pixel, color):
                            # Calculate real-world coordinates
                            real_x = x * dot_size_mm + dot_size_mm / 2 + border_margin
                            # Flip Y axis (image coordinates are top-left, G-code is bottom-left)
                            real_y = (height - y - 1) * dot_size_mm + dot_size_mm / 2 + border_margin
                            dots.append((real_x, real_y))
                
                # Optimize path if enabled
                if self.config["optimize_path"] and dots:
                    dots = self._optimize_path(dots)
                
                # Generate movement commands for this color
                current_pos = (home_x, home_y)
                for i, (x, y) in enumerate(dots):
                    # Rapid move to position
                    gcode.append(f"G0 X{x:.2f} Y{y:.2f} Z5.0 ; Move to dot position")
                    # Lower to spray position
                    gcode.append(f"G1 Z0.0 F{speed_z} ; Lower to spray position")
                    # Spray
                    gcode.append(f"G4 P{int(spray_delay * 1000)} ; Spray delay")
                    # Raise
                    gcode.append(f"G1 Z5.0 F{speed_z} ; Raise after spraying")
                    
                    # Update current position
                    current_pos = (x, y)
                    total_dots += 1
                
                gcode.append("")
            
            # Update progress
            if progress_callback:
                progress = progress_base + batch_num * progress_per_batch
                progress_callback(min(99, progress))
        
        # G-code epilogue
        gcode.append("; === Finished ===")
        gcode.append(f"G0 X{home_x} Y{home_y} ; Return to home position")
        gcode.append("M84 ; Disable motors")
        
        # Output information
        # Check if palette is a numpy array before calling tolist()
        palette_list = palette.tolist() if hasattr(palette, 'tolist') else palette
        
        output_info = {
            "total_dots": total_dots,
            "total_batches": total_color_batches,
            "palette": palette_list,  # Use the converted list
            "batches": batches,
            "color_counts": color_counts
        }
        
        # Final progress update
        if progress_callback:
            progress_callback(100)
        
        return "\n".join(gcode), output_info
    
    def _apply_floyd_steinberg_dithering(self, image, palette):
        """Apply Floyd-Steinberg dithering to an image."""
        # Convert to PIL Image for processing
        pil_image = Image.fromarray(image)
        
        # Convert palette to PIL palette format
        pil_palette = []
        for color in palette:
            pil_palette.extend(color)
        
        # Ensure palette is exactly 768 bytes (256 colors * 3 RGB values)
        if len(pil_palette) > 768:
            pil_palette = pil_palette[:768]  # Truncate if too long
        else:
            # Pad with zeros to reach 768 bytes
            pil_palette.extend([0] * (768 - len(pil_palette)))
        
        # Create a palette image and quantize
        pal_image = Image.new('P', (1, 1))
        pal_image.putpalette(pil_palette)
        
        try:
            dithered_image = pil_image.quantize(palette=pal_image, dither=Image.FLOYDSTEINBERG)
        except Exception as e:
            # Fallback to non-dithered quantization if error occurs
            print(f"Dithering error: {e}. Falling back to standard quantization.")
            dithered_image = pil_image.quantize(colors=len(palette), dither=Image.FLOYDSTEINBERG)
          # Convert back to RGB and numpy array
        dithered_image = dithered_image.convert('RGB')
        result = np.array(dithered_image)
        
        return result
    
    def _apply_ordered_dithering(self, image, palette):
        """Apply ordered dithering to an image."""
        # Create a PIL image
        pil_image = Image.fromarray(image)
        
        # Convert palette to PIL palette format
        pil_palette = []
        for color in palette:
            pil_palette.extend(color)
        
        # Ensure palette is exactly 768 bytes (256 colors * 3 RGB values)
        if len(pil_palette) > 768:
            pil_palette = pil_palette[:768]  # Truncate if too long
        else:
            # Pad with zeros to reach 768 bytes
            pil_palette.extend([0] * (768 - len(pil_palette)))
        
        # Create a palette image and quantize with ordered dithering
        pal_image = Image.new('P', (1, 1))
        pal_image.putpalette(pil_palette)
        
        try:
            dithered_image = pil_image.quantize(palette=pal_image, dither=Image.ORDERED)
        except Exception as e:
            # Fallback to non-dithered quantization if error occurs
            print(f"Dithering error: {e}. Falling back to standard quantization.")
            dithered_image = pil_image.quantize(colors=len(palette), dither=Image.ORDERED)
        
        # Convert back to RGB and numpy array
        dithered_image = dithered_image.convert('RGB')
        result = np.array(dithered_image)
        
        return result
    
    def _optimize_path(self, dots):
        """Optimize the path to minimize travel distance using nearest neighbor."""
        if not dots:
            return []
        
        # Start with the first dot
        path = [dots[0]]
        remaining = dots[1:]
        
        while remaining:
            current = path[-1]
            # Find nearest dot
            nearest_idx = 0
            nearest_dist = float('inf')
            
            for i, dot in enumerate(remaining):
                dist = math.sqrt((current[0] - dot[0])**2 + (current[1] - dot[1])**2)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = i
            
            # Add the nearest dot to the path
            path.append(remaining[nearest_idx])
            # Remove it from the remaining dots
            remaining.pop(nearest_idx)
        
        return path
