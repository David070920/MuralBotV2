"""
Advanced dithering algorithms for MuralBot image processing.
"""

import numpy as np
from PIL import Image, ImageDraw
import math

def apply_jarvis_dithering(image, palette):
    """
    Apply Jarvis-Judice-Ninke dithering algorithm for better quality dithering.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    # Create a copy of the image as float for processing
    h, w, c = image.shape
    img_float = image.astype(np.float32)
    result = np.zeros_like(image)
    
    # Find the closest color for each pixel and propagate the error
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x].copy()
            
            # Find the closest color in the palette
            distances = np.sum((palette - old_pixel) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            new_pixel = palette[closest_idx]
            
            # Store the new pixel in the result
            result[y, x] = new_pixel
            
            # Calculate the error
            error = old_pixel - new_pixel
            
            # Distribute the error (Jarvis-Judice-Ninke pattern)
            # Current row
            if x + 1 < w:
                img_float[y, x + 1] += error * 7/48
            if x + 2 < w:
                img_float[y, x + 2] += error * 5/48
            
            # Next row
            if y + 1 < h:
                if x - 2 >= 0:
                    img_float[y + 1, x - 2] += error * 3/48
                if x - 1 >= 0:
                    img_float[y + 1, x - 1] += error * 5/48
                img_float[y + 1, x] += error * 7/48
                if x + 1 < w:
                    img_float[y + 1, x + 1] += error * 5/48
                if x + 2 < w:
                    img_float[y + 1, x + 2] += error * 3/48
            
            # Two rows down
            if y + 2 < h:
                if x - 2 >= 0:
                    img_float[y + 2, x - 2] += error * 1/48
                if x - 1 >= 0:
                    img_float[y + 2, x - 1] += error * 3/48
                img_float[y + 2, x] += error * 5/48
                if x + 1 < w:
                    img_float[y + 2, x + 1] += error * 3/48
                if x + 2 < w:
                    img_float[y + 2, x + 2] += error * 1/48
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_stucki_dithering(image, palette):
    """
    Apply Stucki dithering algorithm.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    # Create a copy of the image as float for processing
    h, w, c = image.shape
    img_float = image.astype(np.float32)
    result = np.zeros_like(image)
    
    # Coefficients for Stucki dithering
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x].copy()
            
            # Find the closest color in the palette
            distances = np.sum((palette - old_pixel) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            new_pixel = palette[closest_idx]
            
            # Store the new pixel in the result
            result[y, x] = new_pixel
            
            # Calculate the error
            error = old_pixel - new_pixel
            
            # Distribute the error (Stucki pattern)
            # Current row
            if x + 1 < w:
                img_float[y, x + 1] += error * 8/42
            if x + 2 < w:
                img_float[y, x + 2] += error * 4/42
            
            # Next row
            if y + 1 < h:
                if x - 2 >= 0:
                    img_float[y + 1, x - 2] += error * 2/42
                if x - 1 >= 0:
                    img_float[y + 1, x - 1] += error * 4/42
                img_float[y + 1, x] += error * 8/42
                if x + 1 < w:
                    img_float[y + 1, x + 1] += error * 4/42
                if x + 2 < w:
                    img_float[y + 1, x + 2] += error * 2/42
            
            # Two rows down
            if y + 2 < h:
                if x - 2 >= 0:
                    img_float[y + 2, x - 2] += error * 1/42
                if x - 1 >= 0:
                    img_float[y + 2, x - 1] += error * 2/42
                img_float[y + 2, x] += error * 4/42
                if x + 1 < w:
                    img_float[y + 2, x + 1] += error * 2/42
                if x + 2 < w:
                    img_float[y + 2, x + 2] += error * 1/42
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_atkinson_dithering(image, palette):
    """
    Apply Atkinson dithering algorithm - good for preserving detail.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    # Create a copy of the image as float for processing
    h, w, c = image.shape
    img_float = image.astype(np.float32)
    result = np.zeros_like(image)
    
    # Find the closest color for each pixel and propagate the error
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x].copy()
            
            # Find the closest color in the palette
            distances = np.sum((palette - old_pixel) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            new_pixel = palette[closest_idx]
            
            # Store the new pixel in the result
            result[y, x] = new_pixel
            
            # Calculate the error
            error = old_pixel - new_pixel
            
            # Distribute the error (Atkinson pattern) - uses 1/8 of error
            error = error / 8  # Pre-divide error for Atkinson
            
            # Current row
            if x + 1 < w:
                img_float[y, x + 1] += error
            if x + 2 < w:
                img_float[y, x + 2] += error
            
            # Next row
            if y + 1 < h:
                if x - 1 >= 0:
                    img_float[y + 1, x - 1] += error
                img_float[y + 1, x] += error
                if x + 1 < w:
                    img_float[y + 1, x + 1] += error
            
            # Two rows down
            if y + 2 < h:
                img_float[y + 2, x] += error
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_sierra_dithering(image, palette):
    """
    Apply Sierra dithering algorithm.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    # Create a copy of the image as float for processing
    h, w, c = image.shape
    img_float = image.astype(np.float32)
    result = np.zeros_like(image)
    
    # Find the closest color for each pixel and propagate the error
    for y in range(h):
        for x in range(w):
            old_pixel = img_float[y, x].copy()
            
            # Find the closest color in the palette
            distances = np.sum((palette - old_pixel) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            new_pixel = palette[closest_idx]
            
            # Store the new pixel in the result
            result[y, x] = new_pixel
            
            # Calculate the error
            error = old_pixel - new_pixel
            
            # Distribute the error (Sierra pattern)
            # Current row
            if x + 1 < w:
                img_float[y, x + 1] += error * 5/32
            if x + 2 < w:
                img_float[y, x + 2] += error * 3/32
            
            # Next row
            if y + 1 < h:
                if x - 2 >= 0:
                    img_float[y + 1, x - 2] += error * 2/32
                if x - 1 >= 0:
                    img_float[y + 1, x - 1] += error * 4/32
                img_float[y + 1, x] += error * 5/32
                if x + 1 < w:
                    img_float[y + 1, x + 1] += error * 4/32
                if x + 2 < w:
                    img_float[y + 1, x + 2] += error * 2/32
            
            # Two rows down
            if y + 2 < h:
                if x - 1 >= 0:
                    img_float[y + 2, x - 1] += error * 2/32
                img_float[y + 2, x] += error * 3/32
                if x + 1 < w:
                    img_float[y + 2, x + 1] += error * 2/32
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_enhanced_floyd_steinberg(image, palette):
    """
    Apply Enhanced Floyd-Steinberg dithering with error limiting and better color matching.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    h, w, c = image.shape
    img_float = image.astype(np.float32)
    result = np.zeros_like(image)
    
    # Pre-compute LAB colors for better perceptual matching
    try:
        from skimage import color
        has_skimage = True
        # Convert palette to LAB color space
        lab_palette = color.rgb2lab(palette.reshape(1, -1, 3) / 255.0)
        lab_palette = lab_palette.reshape(-1, 3)
        
        # Convert image to LAB for better perceptual processing
        lab_image = color.rgb2lab(image / 255.0)
        
    except ImportError:
        has_skimage = False
    
    # Add a small color randomization to avoid banding
    noise = np.random.uniform(-5, 5, img_float.shape).astype(np.float32)
    img_float = np.clip(img_float + noise, 0, 255)
    
    # Define error diffusion coefficients
    # These are slightly adjusted from standard Floyd-Steinberg for better results
    diffusion_coefficients = [
        (1, 0, 7/16),   # right
        (-1, 1, 3/16),  # left bottom
        (0, 1, 5/16),   # bottom
        (1, 1, 1/16)    # right bottom
    ]
    
    for y in range(h):
        # Process every other row in reverse for serpentine scanning
        # This helps avoid directional artifacts
        reverse = (y % 2 == 1)
        x_range = range(w-1, -1, -1) if reverse else range(w)
        
        for x in x_range:
            old_pixel = img_float[y, x].copy()
            
            # Find the closest color with perceptual matching if skimage is available
            if has_skimage:
                old_pixel_lab = lab_image[y, x].copy()
                distances = np.sum((lab_palette - old_pixel_lab) ** 2, axis=1)
                closest_idx = np.argmin(distances)
            else:
                # Fall back to RGB distance
                distances = np.sum((palette - old_pixel) ** 2, axis=1)
                closest_idx = np.argmin(distances)
            
            new_pixel = palette[closest_idx]
            
            # Store the new pixel in the result
            result[y, x] = new_pixel
            
            # Calculate error in RGB space
            error = old_pixel - new_pixel
            
            # Limit error to prevent artifacts (important for mural painting)
            error = np.clip(error, -32, 32)
            
            # Distribute error according to serpentine pattern
            for dx, dy, coeff in diffusion_coefficients:
                if reverse:
                    dx = -dx  # Flip diffusion pattern for serpentine scanning
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    img_float[ny, nx] += error * coeff
    
    return np.clip(result, 0, 255).astype(np.uint8)

def apply_blue_noise_dithering(image, palette):
    """
    Apply blue noise dithering for optimal visual quality.
    Blue noise has a more random, yet uniform distribution compared to other patterns.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    h, w, c = image.shape
    result = np.zeros_like(image)
    
    # Create a blue noise pattern
    # For simplicity, we're approximating blue noise with a high-quality pseudorandom
    # pattern with threshold modulation
    threshold_map = np.zeros((h, w))
    
    # Generate blue-noise-like pattern
    # This is a simplified version; a true blue noise pattern is more complex
    noise_scale = 0.4  # Controls the noise intensity
    for y in range(h):
        for x in range(w):
            # Combine different frequency noise for better distribution
            n1 = math.sin(x * 0.1 + y * 0.1) * 0.5 + 0.5
            n2 = math.sin(x * 0.2 - y * 0.15) * 0.5 + 0.5
            n3 = math.sin(x * 0.05 + y * 0.2) * 0.5 + 0.5
            
            # Combine noise values with different weights
            noise = (n1 * 0.5 + n2 * 0.3 + n3 * 0.2) * noise_scale
            
            threshold_map[y, x] = noise
    
    # Apply the threshold map to determine colors
    for y in range(h):
        for x in range(w):
            pixel = image[y, x].astype(float)
            threshold = threshold_map[y, x] * 255
            
            # Adjust pixel values with threshold
            adjusted_pixel = np.clip(pixel + threshold - 128, 0, 255)
            
            # Find the closest color in the palette
            distances = np.sum((palette - adjusted_pixel) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            
            result[y, x] = palette[closest_idx]
    
    return result

def apply_pattern_dithering(image, palette):
    """
    Apply a geometric pattern dithering - useful for artistic effects.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        
    Returns:
        A dithered image as numpy array
    """
    h, w, c = image.shape
    result = np.zeros_like(image)
    
    # Define Bayer matrix for ordered dithering
    bayer_matrix_4x4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) / 16.0
    
    # Apply the Bayer matrix
    for y in range(h):
        for x in range(w):
            pixel = image[y, x].astype(float)
            
            # Apply threshold from Bayer matrix
            bayer_value = bayer_matrix_4x4[y % 4, x % 4] * 255 - 128
            
            # Adjust pixel values with threshold
            adjusted_pixel = np.clip(pixel + bayer_value, 0, 255)
            
            # Find the closest color in the palette
            distances = np.sum((palette - adjusted_pixel) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            
            result[y, x] = palette[closest_idx]
    
    return result

def apply_halftone_dithering(image, palette, dot_size=4):
    """
    Apply circular halftone dithering simulating traditional printing.
    
    Args:
        image: A numpy array containing the RGB image
        palette: The color palette (numpy array)
        dot_size: Size of halftone dots
        
    Returns:
        A dithered image as numpy array
    """
    h, w, c = image.shape
    result = np.zeros_like(image)
    
    # Convert to grayscale for intensity
    if c == 3:
        grayscale = np.dot(image[...,:3], [0.299, 0.587, 0.114])
    else:
        grayscale = image.copy()
    
    # Create PIL image for drawing
    pil_image = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(pil_image)
    
    # Palette for drawing (convert to tuples for PIL)
    pil_palette = [(int(r), int(g), int(b)) for r, g, b in palette]
    dark_color = min(pil_palette, key=lambda x: sum(x))
    light_color = max(pil_palette, key=lambda x: sum(x))
    
    # Draw halftone dots
    for y in range(0, h, dot_size):
        for x in range(0, w, dot_size):
            # Get average intensity for this cell
            cell_x_end = min(x + dot_size, w)
            cell_y_end = min(y + dot_size, h)
            cell = grayscale[y:cell_y_end, x:cell_x_end]
            if cell.size == 0:
                continue
                
            avg_intensity = np.mean(cell)
            
            # Calculate dot radius based on intensity (255=white, 0=black)
            # For lighter areas, smaller dots; for darker areas, larger dots
            normalized_intensity = avg_intensity / 255.0
            radius = (1.0 - normalized_intensity) * dot_size * 0.5
            
            # Draw the dot
            if radius > 0.1:  # Only draw visible dots
                draw.ellipse(
                    [x, y, x + dot_size - 1, y + dot_size - 1],
                    fill=dark_color
                )
    
    # Convert back to numpy array
    result = np.array(pil_image)
    
    return result
