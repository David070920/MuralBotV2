"""
G-code visualization tab for MuralBot application.
"""

import os
import re
import numpy as np
import cv2
import tempfile
import shutil
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QGroupBox, QFileDialog, QCheckBox, QSpinBox, QFormLayout,
                           QProgressBar, QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal


class GcodeTab(QWidget):
    """Tab for G-code visualization."""
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.simulator = GcodeSimulator()
        self.gcode_file_path = None
        self.animation_thread = None
        self.wall_width = 2.0  # Default wall width in meters
        self.wall_height = 2.0  # Default wall height in meters
        self.dot_resolution = 20.0  # Default spray dot diameter in mm
        
        self.init_ui()
    
    def update_config(self, config):
        """Update configuration from the config tab."""
        if "wall_width" in config:
            self.wall_width = config["wall_width"] 
        if "wall_height" in config:
            self.wall_height = config["wall_height"]
        if "dot_resolution" in config:
            self.dot_resolution = config["dot_resolution"]  # in mm
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Main control panel
        control_group = QGroupBox("G-code Animation Controls")
        control_layout = QVBoxLayout()
        
        # Load G-code button
        load_layout = QHBoxLayout()
        load_btn = QPushButton("Load G-code File")
        load_btn.clicked.connect(self.load_gcode_file)
        load_layout.addWidget(load_btn)
        
        # Display the file path
        self.file_label = QLabel("No G-code file loaded")
        load_layout.addWidget(self.file_label, 1)
        control_layout.addLayout(load_layout)
        
        # Animation settings
        settings_layout = QFormLayout()
        
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(10, 60)
        self.fps_spinbox.setValue(30)
        settings_layout.addRow("FPS:", self.fps_spinbox)
        
        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setRange(5, 300)
        self.duration_spinbox.setValue(30)
        settings_layout.addRow("Duration (seconds):", self.duration_spinbox)
        
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["Low (Fastest)", "Medium", "High (Slowest)"])
        self.quality_combo.setCurrentIndex(0)  # Default to low quality for speed
        settings_layout.addRow("Quality:", self.quality_combo)
        
        control_layout.addLayout(settings_layout)
        
        # Progress bar
        self.animation_progress = QProgressBar()
        self.animation_progress.setRange(0, 100)
        control_layout.addWidget(self.animation_progress)
        
        # Generate button
        self.generate_btn = QPushButton("Generate Animation")
        self.generate_btn.clicked.connect(self.generate_animation)
        self.generate_btn.setEnabled(False)
        control_layout.addWidget(self.generate_btn)
        
        control_group.setLayout(control_layout)
        
        # Add info text
        info_label = QLabel(
            "Load a G-code file and generate an MP4 animation of the robot's painting path.\n"
            "Low quality generates faster but with fewer frames. High quality takes longer but is smoother."
        )
        info_label.setWordWrap(True)
        info_label.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(control_group)
        main_layout.addWidget(info_label)
        main_layout.addStretch(1)
        
        self.setLayout(main_layout)
    
    def load_gcode(self, gcode_text, file_path=None):
        """Load G-code text."""
        # Save file path if provided
        if file_path:
            self.gcode_file_path = file_path
            self.file_label.setText(f"Loaded: {os.path.basename(file_path)}")
        
        # Parse the G-code without displaying the text
        total_frames = self.simulator.load_gcode(gcode_text)
        
        # Enable animation button if we have G-code loaded
        self.generate_btn.setEnabled(total_frames > 0)
        
        # Reset progress bar
        self.animation_progress.setValue(0)
    
    def load_gcode_file(self):
        """Load G-code from a file."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open G-code File", "", "G-code Files (*.gcode);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, "r") as f:
                    gcode_text = f.read()
                self.load_gcode(gcode_text, file_path)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Error loading G-code file: {str(e)}")
    
    def generate_animation(self):
        """Generate an MP4 animation of the G-code execution."""
        try:
            # Ask user where to save the animation
            default_filename = "gcode_animation.mp4"
            if self.gcode_file_path:
                base_name = os.path.splitext(os.path.basename(self.gcode_file_path))[0]
                default_filename = f"{base_name}_animation.mp4"
            
            output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Animation", default_filename, "MP4 Files (*.mp4);;All Files (*)"
            )
            
            if not output_path:
                return
            
            # Disable button during generation
            self.generate_btn.setEnabled(False)
            
            # Get quality setting
            quality_text = self.quality_combo.currentText()
            if "Low" in quality_text:
                quality = "low"
            elif "High" in quality_text:
                quality = "high"
            else:
                quality = "medium"
            
            # Start the optimized animation generation thread
            self.animation_thread = OptimizedAnimationThread(
                self.simulator,
                output_path,
                fps=self.fps_spinbox.value(),
                duration=self.duration_spinbox.value(),
                quality=quality,
                wall_width=self.wall_width,
                wall_height=self.wall_height,
                dot_resolution=getattr(self, "dot_resolution", 20.0)
            )
            
            # Connect signals
            self.animation_thread.progress_updated.connect(self.animation_progress.setValue)
            self.animation_thread.finished.connect(self.animation_generation_finished)
            self.animation_thread.error.connect(self.animation_generation_error)
            
            # Reset progress bar
            self.animation_progress.setValue(0)
            
            # Start the thread
            self.animation_thread.start()
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generating animation: {str(e)}")
            self.generate_btn.setEnabled(True)
    
    def animation_generation_finished(self, output_path):
        """Handle completion of animation generation."""
        self.animation_progress.setValue(100)
        self.generate_btn.setEnabled(True)
        
        # Ask if the user wants to open the video
        reply = QMessageBox.question(
            self, "Animation Complete", 
            f"Animation saved to {output_path}. Do you want to open it now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )
        
        if reply == QMessageBox.Yes:
            # Open the video with default application
            import subprocess
            import platform
            
            if platform.system() == "Windows":
                os.startfile(output_path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.call(("open", output_path))
            else:  # Linux
                subprocess.call(("xdg-open", output_path))
    
    def animation_generation_error(self, error_message):
        """Handle errors in animation generation."""
        QMessageBox.warning(self, "Error", f"Error generating animation: {error_message}")
        self.generate_btn.setEnabled(True)


class GcodeSimulator:
    """Class to parse and simulate G-code execution."""
    
    def __init__(self):
        self.gcode_lines = []
        self.positions = []
        self.colors = []
        self.current_color = (0, 0, 255)  # Default blue
        self.dots = []
        self.dot_colors = []
        self.bounds = [0, 0, 100, 100]  # [min_x, min_y, max_x, max_y]
    
    def load_gcode(self, gcode_text):
        """Load G-code from text."""
        self.gcode_lines = gcode_text.split("\n")
        self.parse_gcode()
        return len(self.positions)
    
    def parse_gcode(self):
        """Parse the G-code and extract positions and colors."""
        self.positions = []
        self.colors = []
        self.dots = []
        self.dot_colors = []
        
        current_pos = [0, 0, 0]  # [x, y, z]
        is_spraying = False
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")
          # Various regex patterns for color detection
        # Standard RGB format: ; Color: RGB(r, g, b)
        std_color_re = re.compile(r"; Color: RGB\((\d+),\s*(\d+),\s*(\d+)\)")
        
        # Alternative format without RGB prefix: ; Color: (r, g, b)
        alt_color_re = re.compile(r"; Color:\s*\((\d+),\s*(\d+),\s*(\d+)\)")
        
        # Tuple format from the palette comment: ([r, g, b])
        tuple_color_re = re.compile(r";\s*Color \d+: RGB\((\d+),\s*(\d+),\s*(\d+)\)")
        
        # Format with "np.uint8" prefix
        numpy_color_re = re.compile(r"; Color: RGB\(np\.uint8\((\d+)\), np\.uint8\((\d+)\), np\.uint8\((\d+)\)\)")
        
        for line in self.gcode_lines:
            line = line.strip()
            
            # Try all regex patterns
            for pattern in [std_color_re, alt_color_re, tuple_color_re, numpy_color_re]:
                match = pattern.search(line)
                if match:
                    r, g, b = map(int, match.groups())
                    self.current_color = (r, g, b)
                    break
            
            # Also check for tool change commands which often indicate color changes
            if line.startswith("T") and len(line) >= 2 and line[1].isdigit():
                # This is a tool change command - we need to look for the preceding color comment
                tool_num = int(line[1:].split()[0])
                # We'll keep the current color, as the tool change usually follows a color comment
            
            # Skip empty lines and pure comments
            if not line or line.startswith(";"):
                continue
            
            # Remove inline comments
            if ";" in line:
                line = line[:line.index(";")].strip()
            
            # Parse G-code commands
            parts = line.split()
            if not parts:
                continue
            
            command = parts[0]
            
            # Handle G0/G1 (movement)
            if command in ["G0", "G1"]:
                for param in parts[1:]:
                    if param.startswith("X"):
                        current_pos[0] = float(param[1:])
                    elif param.startswith("Y"):
                        current_pos[1] = float(param[1:])
                    elif param.startswith("Z"):
                        current_pos[2] = float(param[1:])
                        # Check if this is spraying position (Z close to 0)
                        is_spraying = current_pos[2] < 1.0
                
                # Record position
                self.positions.append(current_pos.copy())
                self.colors.append(self.current_color)
                
                # Update bounds
                min_x = min(min_x, current_pos[0])
                min_y = min(min_y, current_pos[1])
                max_x = max(max_x, current_pos[0])
                max_y = max(max_y, current_pos[1])
                
                # Record spray dot if in spray position
                if is_spraying:
                    self.dots.append(current_pos[:2])
                    self.dot_colors.append(self.current_color)
        
        # Update bounds
        if min_x != float("inf"):
            self.bounds = [min_x, min_y, max_x, max_y]
            
        # Print debug statistics
        print(f"Total positions: {len(self.positions)}")
        print(f"Total dots: {len(self.dots)}")
        unique_colors = set(tuple(c) for c in self.colors)
        print(f"Unique colors detected: {len(unique_colors)}")
        for i, color in enumerate(sorted(unique_colors)):
            if i < 20:  # Just show first 20 colors to keep output manageable
                print(f"  RGB{color}")
    
    def get_frame(self, frame_index):
        """Get the position and state at a specific frame."""
        if 0 <= frame_index < len(self.positions):
            return self.positions[frame_index], self.colors[frame_index]
        return None, None


class OptimizedAnimationThread(QThread):
    """Optimized thread for generating animation frames."""
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, simulator, output_path, fps=30, duration=30, quality="medium", wall_width=2.0, wall_height=2.0, dot_resolution=20.0):
        super().__init__()
        self.simulator = simulator
        self.output_path = output_path
        self.fps = fps
        self.duration = duration  # seconds
        self.quality = quality    # low, medium, high
        self.temp_dir = None
        self.wall_width = wall_width  # Wall width in meters
        self.wall_height = wall_height  # Wall height in meters
        self.dot_resolution = dot_resolution  # Spray dot diameter in mm
    
    def run(self):
        try:
            # Create temporary directory for frames if needed
            self.temp_dir = tempfile.mkdtemp()
            
            # Get basic information
            num_positions = len(self.simulator.positions)
            if num_positions == 0:
                self.error.emit("No G-code positions found")
                return
            
            # Calculate total frames based on duration and fps
            total_frames = self.fps * self.duration
            
            # Determine frame sampling based on quality setting
            if self.quality == "low":
                # Super low quality - generate only about 60 frames regardless of duration
                frame_count = min(60, total_frames)
                frame_step = max(1, total_frames // frame_count)
            elif self.quality == "medium":
                # Medium quality - generate about 150 frames regardless of duration
                frame_count = min(150, total_frames)
                frame_step = max(1, total_frames // frame_count)
            else:  # high
                # High quality - generate about 300 frames regardless of duration
                frame_count = min(300, total_frames)
                frame_step = max(1, total_frames // frame_count)
            
            # Calculate frame indices to generate
            frame_indices = list(range(0, total_frames, frame_step))
            if not frame_indices or frame_indices[-1] != total_frames - 1:
                frame_indices.append(total_frames - 1)  # Always include the last frame
            actual_frames_to_generate = len(frame_indices)
            
            # Print info for debugging
            print(f"Generating {actual_frames_to_generate} frames out of {total_frames} total frames")
            print(f"Positions: {num_positions}, Colors: {len(self.simulator.colors)}")
            
            # Show the unique colors we will be using
            unique_colors = set(tuple(c) for c in self.simulator.colors)
            print(f"Unique colors in G-code: {unique_colors}")
            
            # Get the bounds for the plot
            min_x, min_y, max_x, max_y = self.simulator.bounds
            margin = (max(max_x - min_x, max_y - min_y) * 0.05) or 10
            x_range = (min_x - margin, max_x + margin)
            y_range = (min_y - margin, max_y + margin)
            
            # Calculate dimensions maintaining proper aspect ratio based on wall dimensions
            base_width, base_height = 800, 600
            
            # Check if we need to use actual wall dimensions for better scaling
            if self.wall_width > 0 and self.wall_height > 0:
                # Use wall dimensions to calculate aspect ratio
                wall_aspect_ratio = self.wall_width / self.wall_height
                
                # Calculate dimensions that maintain wall aspect ratio
                if wall_aspect_ratio > (base_width / base_height):
                    # Width constrained
                    width = base_width
                    height = int(width / wall_aspect_ratio)
                else:
                    # Height constrained
                    height = base_height
                    width = int(height * wall_aspect_ratio)
            else:
                # Fallback to default dimensions if wall dimensions are invalid
                width, height = base_width, base_height
            
            # Calculate scaling factors to convert from G-code coordinates to image pixels
            x_scale = width / (x_range[1] - x_range[0])
            y_scale = height / (y_range[1] - y_range[0])
            
            # Function to convert G-code coordinates to image pixels
            def to_pixel(x, y):
                px = int((x - x_range[0]) * x_scale)
                py = height - int((y - y_range[0]) * y_scale)  # Flip Y-axis
                return px, py
            
            # Create video writer directly
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            
            # Generate and write frames directly to video
            for frame_idx, i in enumerate(frame_indices):
                # Create blank image with white background
                image = np.ones((height, width, 3), dtype=np.uint8) * 255
                
                # Calculate the corresponding position index
                position_idx = min(int(i * num_positions / total_frames), num_positions - 1)
                
                # Draw grid lines (light gray)
                grid_color = (240, 240, 240)
                grid_step = 50  # pixels
                for x in range(0, width, grid_step):
                    cv2.line(image, (x, 0), (x, height), grid_color, 1)
                for y in range(0, height, grid_step):
                    cv2.line(image, (0, y), (width, y), grid_color, 1)
                
                # Draw axes (light gray)
                cv2.line(image, (0, height//2), (width, height//2), (200, 200, 200), 1)
                cv2.line(image, (width//2, 0), (width//2, height), (200, 200, 200), 1)
                
                # Draw origin point if in view
                origin_pixel = to_pixel(0, 0)
                if 0 <= origin_pixel[0] < width and 0 <= origin_pixel[1] < height:
                    cv2.drawMarker(image, origin_pixel, (0, 0, 0), cv2.MARKER_CROSS, 10, 2)
                
                # Draw path up to current position (use color if available)
                positions = self.simulator.positions[:position_idx+1]
                colors = self.simulator.colors[:position_idx+1]
                
                if len(positions) > 1:
                    for j in range(1, len(positions)):
                        # Get positions and colors
                        start_pos = positions[j-1]
                        end_pos = positions[j]
                        color = colors[j]
                        
                        # Convert to BGR format for OpenCV (swap R and B)
                        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                        
                        # Draw line segment
                        start_pixel = to_pixel(start_pos[0], start_pos[1])
                        end_pixel = to_pixel(end_pos[0], end_pos[1])
                        cv2.line(image, start_pixel, end_pixel, bgr_color, 1)
                
                # Count visible dots
                visible_dot_count = 0
                for j, pos in enumerate(self.simulator.positions[:position_idx+1]):
                    if pos[2] < 1.0:  # Check if it's a spray position
                        visible_dot_count += 1
                
                # Show visible dots
                dots = self.simulator.dots[:visible_dot_count]
                dot_colors = self.simulator.dot_colors[:visible_dot_count]
                
                if dots:
                    for j, (dot, color) in enumerate(zip(dots, dot_colors)):
                        dot_pixel = to_pixel(dot[0], dot[1])
                        # Ensure pixel is within bounds
                        if 0 <= dot_pixel[0] < width and 0 <= dot_pixel[1] < height:
                            # Convert to BGR format for OpenCV
                            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                              # Calculate dot size based on dot_resolution
                            # The dot_resolution is in mm, convert to pixels based on wall dimensions
                            dot_size_mm = self.dot_resolution
                            
                            # Calculate the fraction of the wall width/height this represents
                            dot_fraction_x = dot_size_mm / (self.wall_width * 1000)
                            dot_fraction_y = dot_size_mm / (self.wall_height * 1000)
                            
                            # Convert to pixels
                            dot_size_x = int(dot_fraction_x * width)
                            dot_size_y = int(dot_fraction_y * height)
                            
                            # Use the average as the dot diameter
                            dot_radius = max(3, (dot_size_x + dot_size_y) // 4)  # Ensure at least 3px radius
                            
                            # Draw filled circle with slight transparency for overlapping dots
                            overlay = image.copy()
                            cv2.circle(overlay, dot_pixel, dot_radius, bgr_color, -1)  # Filled circle
                            
                            # Apply the overlay with transparency
                            alpha = 0.8  # Transparency factor
                            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                
                # Draw current position
                if position_idx < num_positions:
                    current_pos = positions[position_idx]
                    current_color = colors[position_idx]
                    current_pixel = to_pixel(current_pos[0], current_pos[1])
                    # Ensure pixel is within bounds
                    if 0 <= current_pixel[0] < width and 0 <= current_pixel[1] < height:
                        # Draw highlighted position
                        cv2.circle(image, current_pixel, 7, (0, 0, 0), 1)  # Black outline
                        # Convert color to BGR
                        bgr_color = (int(current_color[2]), int(current_color[1]), int(current_color[0]))
                        cv2.circle(image, current_pixel, 6, bgr_color, -1)  # Filled circle
                
                # Add title and information
                cv2.putText(image, "G-code Visualization", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(image, f"Frame: {i}/{total_frames}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add wall dimensions information
                wall_info = f"Wall Size: {self.wall_width:.2f}m × {self.wall_height:.2f}m"
                cv2.putText(image, wall_info, (width - 250, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add dot resolution information
                dot_info = f"Spray Dot Size: {self.dot_resolution:.1f}mm"
                cv2.putText(image, dot_info, (width - 250, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Calculate and display resolution information
                # Resolution in G-code units per pixel
                x_resolution = (x_range[1] - x_range[0]) / width
                y_resolution = (y_range[1] - y_range[0]) / height
                avg_resolution = (x_resolution + y_resolution) / 2
                resolution_text = f"Resolution: {avg_resolution:.2f} units/pixel"
                cv2.putText(image, resolution_text, (width - 250, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
                # Add color legend (show up to 8 unique colors)
                if dot_colors:
                    # Use dictionary to keep track of unique colors
                    unique_color_dict = {}
                    for color in dot_colors:
                        color_key = tuple(color)
                        if color_key not in unique_color_dict and len(unique_color_dict) < 8:
                            unique_color_dict[color_key] = True
                    
                    # Draw color legend
                    legend_y = 90
                    for color in unique_color_dict:
                        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                        cv2.circle(image, (20, legend_y), 10, bgr_color, -1)
                        # Draw color value text
                        rgb_text = f"RGB({color[0]}, {color[1]}, {color[2]})"
                        cv2.putText(image, rgb_text, (35, legend_y+5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        legend_y += 25
                
                # Write frame directly to video
                video.write(image)
                
                # Update progress
                progress = int((frame_idx + 1) / actual_frames_to_generate * 100)
                self.progress_updated.emit(progress)
            
            # Release video writer
            video.release()
            
            # Clean up temporary directory
            shutil.rmtree(self.temp_dir)
            
            self.finished.emit(self.output_path)
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
            # Clean up temporary directory if it exists
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
