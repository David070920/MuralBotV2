"""
G-code visualization tab for MuralBot application.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                           QGroupBox, QFileDialog, QCheckBox, QSplitter, 
                           QProgressBar, QMessageBox, QSpinBox, QFormLayout)
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal
import cv2
import tempfile
import shutil
import multiprocessing

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
        self.gcode_lines = gcode_text.split('\n')
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
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        # Regular expressions for parsing
        color_re = re.compile(r'; Color: RGB\((\d+), (\d+), (\d+)\)')
        
        for line in self.gcode_lines:
            line = line.strip()
            
            # Skip empty lines and pure comments
            if not line or line.startswith(';'):
                # Check for color comment
                color_match = color_re.search(line)
                if color_match:
                    r, g, b = map(int, color_match.groups())
                    self.current_color = (r, g, b)
                continue
            
            # Remove inline comments
            if ';' in line:
                line = line[:line.index(';')].strip()
            
            # Parse G-code commands
            parts = line.split()
            if not parts:
                continue
            
            command = parts[0]
            
            # Handle G0/G1 (movement)
            if command in ['G0', 'G1']:
                for param in parts[1:]:
                    if param.startswith('X'):
                        current_pos[0] = float(param[1:])
                    elif param.startswith('Y'):
                        current_pos[1] = float(param[1:])
                    elif param.startswith('Z'):
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
        if min_x != float('inf'):
            self.bounds = [min_x, min_y, max_x, max_y]

    def get_frame(self, frame_index):
        """Get the position and state at a specific frame."""
        if 0 <= frame_index < len(self.positions):
            return self.positions[frame_index], self.colors[frame_index]
        return None, None

class AnimationGeneratorThread(QThread):
    """Thread for generating animation frames."""
    progress_updated = pyqtSignal(int)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, simulator, output_path, fps=30, dpi=100, duration=30, skip_frames=False):
        super().__init__()
        self.simulator = simulator
        self.output_path = output_path
        self.fps = fps
        self.dpi = dpi
        self.duration = duration  # seconds
        self.temp_dir = None
        self.skip_frames = skip_frames  # Option to skip frames for faster generation
    
    def run(self):
        try:
            # Create temporary directory for frames
            self.temp_dir = tempfile.mkdtemp()
            
            # Calculate frames based on duration and fps
            num_positions = len(self.simulator.positions)
            if num_positions == 0:
                self.error.emit("No G-code positions found")
                return
                
            total_frames = self.fps * self.duration
            
            # If skip_frames is True, reduce the number of frames for faster generation
            if self.skip_frames:
                frame_step = max(1, total_frames // 100)  # Generate only ~100 frames
                frame_indices = range(0, total_frames, frame_step)
                total_frames = len(frame_indices)
            else:
                frame_indices = range(total_frames)
            
            # Create figure for animation
            fig = Figure(figsize=(10, 8), dpi=self.dpi)
            ax = fig.add_subplot(111)
            
            # Set up the axes
            ax.set_xlabel('X (mm)')
            ax.set_ylabel('Y (mm)')
            ax.set_title('G-code Visualization')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Get the bounds for the plot
            min_x, min_y, max_x, max_y = self.simulator.bounds
            margin = (max(max_x - min_x, max_y - min_y) * 0.05) or 10
            ax.set_xlim(min_x - margin, max_x + margin)
            ax.set_ylim(min_y - margin, max_y + margin)
            
            # Determine the frame indices to generate
            frames_to_generate = frame_indices if self.skip_frames else range(total_frames)
            
            # Generate frames
            for frame_idx, i in enumerate(frames_to_generate):
                # Calculate the corresponding position index
                position_idx = min(int(i * num_positions / total_frames), num_positions - 1)
                
                # Clear previous plot
                ax.clear()
                
                # Re-setup the axes
                ax.set_xlabel('X (mm)')
                ax.set_ylabel('Y (mm)')
                ax.set_title('G-code Visualization')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xlim(min_x - margin, max_x + margin)
                ax.set_ylim(min_y - margin, max_y + margin)
                
                # Draw path up to current position with colors
                positions = self.simulator.positions[:position_idx+1]
                colors = self.simulator.colors[:position_idx+1]
                
                if positions and len(positions) > 1:
                    # Create a colored path by drawing segments with appropriate colors
                    for j in range(1, len(positions)):
                        x_values = [positions[j-1][0], positions[j][0]]
                        y_values = [positions[j-1][1], positions[j][1]]
                        r, g, b = colors[j]
                        color_normalized = (r/255, g/255, b/255)
                        ax.plot(x_values, y_values, '-', color=color_normalized, 
                               alpha=0.7, linewidth=0.8)
                
                # Count visible dots
                visible_dot_count = 0
                for j, pos in enumerate(self.simulator.positions[:position_idx+1]):
                    if pos[2] < 1.0:  # Check if it's a spray position
                        visible_dot_count += 1
                
                # Show visible dots
                dots = self.simulator.dots[:visible_dot_count]
                dot_colors = self.simulator.dot_colors[:visible_dot_count]
                
                if dots:
                    x_values = [dot[0] for dot in dots]
                    y_values = [dot[1] for dot in dots]
                    rgba_colors = [(r/255, g/255, b/255, 0.7) for r, g, b in dot_colors]
                    ax.scatter(x_values, y_values, s=30, c=rgba_colors, marker='o', edgecolors='none')
                
                # Show current position
                if position_idx < num_positions:
                    pos, color = self.simulator.get_frame(position_idx)
                    if pos:
                        r, g, b = color
                        color_normalized = (r/255, g/255, b/255)
                        ax.plot(pos[0], pos[1], 'o', markersize=8, markeredgecolor='black', markerfacecolor=color_normalized)
                
                # Draw the origin
                ax.plot(0, 0, 'kx', markersize=8)
                
                # Add color legend
                unique_colors = {}
                for j, color in enumerate(self.simulator.dot_colors[:visible_dot_count]):
                    color_key = tuple(color)
                    if color_key not in unique_colors and len(unique_colors) < 10:  # Limit to 10 colors in legend
                        unique_colors[color_key] = f"RGB{color_key}"
                
                # Add color swatches to the legend
                handles = []
                labels = []
                for color, label in unique_colors.items():
                    r, g, b = color
                    color_normalized = (r/255, g/255, b/255)
                    handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=color_normalized, markersize=10))
                    labels.append(label)
                
                if handles:
                    ax.legend(handles, labels, loc='upper right', 
                             title="Color Palette", framealpha=0.7)
                
                # Save the frame
                frame_path = os.path.join(self.temp_dir, f'frame_{frame_idx:05d}.png')
                fig.savefig(frame_path)
                
                # Update progress
                progress = int((frame_idx + 1) / len(frames_to_generate) * 100)
                self.progress_updated.emit(progress)
            
            # Generate video from frames using OpenCV
            first_frame = cv2.imread(os.path.join(self.temp_dir, 'frame_00000.png'))
            if first_frame is None:
                self.error.emit("Failed to read generated frames")
                return
                
            height, width, _ = first_frame.shape
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            
            # If we skipped frames during generation, need to handle it during video creation
            if self.skip_frames:
                # For each frame in the output video
                for i in range(total_frames):
                    # Find the closest generated frame
                    closest_idx = min(range(len(frames_to_generate)), 
                                    key=lambda x: abs(frames_to_generate[x] - i))
                    frame_path = os.path.join(self.temp_dir, f'frame_{closest_idx:05d}.png')
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        video.write(frame)
            else:
                # Use all generated frames in order
                for i in range(len(frames_to_generate)):
                    frame_path = os.path.join(self.temp_dir, f'frame_{i:05d}.png')
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        video.write(frame)
            
            video.release()
            
            # Clean up temporary directory
            shutil.rmtree(self.temp_dir)
            
            self.finished.emit(self.output_path)
        
        except Exception as e:
            self.error.emit(str(e))
            # Clean up temporary directory if it exists
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

class GcodeTab(QWidget):
    """Tab for G-code visualization."""
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.simulator = GcodeSimulator()
        self.gcode_file_path = None
        self.animation_thread = None
        
        self.init_ui()
    
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
        
        self.fast_mode_check = QCheckBox("Fast Generation Mode")
        self.fast_mode_check.setChecked(True)
        self.fast_mode_check.setToolTip("Generates fewer frames for faster processing")
        settings_layout.addRow("", self.fast_mode_check)
        
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
            "Note: The G-code text is not displayed to improve performance."
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
                with open(file_path, 'r') as f:
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
            
            # Start the animation generation thread
            self.animation_thread = AnimationGeneratorThread(
                self.simulator,
                output_path,
                fps=self.fps_spinbox.value(),
                duration=self.duration_spinbox.value(),
                skip_frames=self.fast_mode_check.isChecked()
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
            
            if platform.system() == 'Windows':
                os.startfile(output_path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', output_path))
            else:  # Linux
                subprocess.call(('xdg-open', output_path))
    
    def animation_generation_error(self, error_message):
        """Handle errors in animation generation."""
        QMessageBox.warning(self, "Error", f"Error generating animation: {error_message}")
        self.generate_btn.setEnabled(True)
