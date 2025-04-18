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
                           QGroupBox, QSlider, QFileDialog, QCheckBox, QSplitter, QTextEdit)
from PyQt5.QtCore import Qt, QTimer, QSettings

class MplCanvas(FigureCanvas):
    """Matplotlib canvas for visualization."""
    
    def __init__(self, parent=None, width=8, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        # Configure axes
        self.axes.set_xlabel('X (mm)')
        self.axes.set_ylabel('Y (mm)')
        self.axes.set_title('G-code Visualization')
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        super(MplCanvas, self).__init__(self.fig)

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
                        current_pos[2] = float(param[2:])
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

class GcodeTab(QWidget):
    """Tab for G-code visualization."""
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.simulator = GcodeSimulator()
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.gcode_file_path = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Create the visualization splitter
        vis_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - G-code text display
        gcode_group = QGroupBox("G-code")
        gcode_layout = QVBoxLayout()
        
        # G-code text editor
        self.gcode_text = QTextEdit()
        self.gcode_text.setReadOnly(True)
        self.gcode_text.setLineWrapMode(QTextEdit.NoWrap)
        self.gcode_text.setFont(self.font())
        
        # Load G-code button
        load_btn = QPushButton("Load G-code File")
        load_btn.clicked.connect(self.load_gcode_file)
        
        gcode_layout.addWidget(load_btn)
        gcode_layout.addWidget(self.gcode_text)
        gcode_group.setLayout(gcode_layout)
        
        # Right panel - Visualization
        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        
        # Matplotlib canvas
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        
        # Visualization options
        options_layout = QHBoxLayout()
        
        self.show_path_check = QCheckBox("Show Path")
        self.show_path_check.setChecked(True)
        self.show_path_check.stateChanged.connect(self.update_visualization)
        
        self.show_dots_check = QCheckBox("Show Dots")
        self.show_dots_check.setChecked(True)
        self.show_dots_check.stateChanged.connect(self.update_visualization)
        
        options_layout.addWidget(self.show_path_check)
        options_layout.addWidget(self.show_dots_check)
        
        # Playback controls
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        
        self.reset_btn = QPushButton("⏮ Reset")
        self.reset_btn.clicked.connect(self.reset_animation)
        self.reset_btn.setEnabled(False)
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(0)
        self.position_slider.setEnabled(False)
        self.position_slider.valueChanged.connect(self.slider_changed)
        
        self.position_label = QLabel("0 / 0")
        
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.reset_btn)
        playback_layout.addWidget(self.position_slider)
        playback_layout.addWidget(self.position_label)
        
        # Add components to visualization layout
        vis_layout.addWidget(self.canvas)
        vis_layout.addLayout(options_layout)
        vis_layout.addLayout(playback_layout)
        vis_group.setLayout(vis_layout)
        
        # Add components to splitter
        vis_splitter.addWidget(gcode_group)
        vis_splitter.addWidget(vis_group)
        vis_splitter.setSizes([300, 700])
        
        main_layout.addWidget(vis_splitter)
        self.setLayout(main_layout)
    
    def load_gcode(self, gcode_text, file_path=None):
        """Load G-code text and initialize the visualization."""
        # Update the text display
        self.gcode_text.setText(gcode_text)
        
        # Save file path if provided
        if file_path:
            self.gcode_file_path = file_path
        
        # Parse the G-code
        self.total_frames = self.simulator.load_gcode(gcode_text)
        self.current_frame = 0
        
        # Update UI
        self.position_slider.setMaximum(self.total_frames - 1 if self.total_frames > 0 else 0)
        self.position_slider.setValue(0)
        self.position_label.setText(f"0 / {self.total_frames}")
        
        # Enable controls
        self.play_btn.setEnabled(self.total_frames > 0)
        self.reset_btn.setEnabled(self.total_frames > 0)
        self.position_slider.setEnabled(self.total_frames > 0)
        
        # Update the visualization
        self.update_visualization()
    
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
                print(f"Error loading G-code file: {str(e)}")
    
    def update_visualization(self):
        """Update the visualization based on current settings and frame."""
        # Clear the plot
        self.canvas.axes.clear()
        
        # Set up the axes
        self.canvas.axes.set_xlabel('X (mm)')
        self.canvas.axes.set_ylabel('Y (mm)')
        self.canvas.axes.set_title('G-code Visualization')
        self.canvas.axes.grid(True, linestyle='--', alpha=0.7)
        
        # Get the bounds for the plot
        min_x, min_y, max_x, max_y = self.simulator.bounds
        # Add margin
        margin = (max(max_x - min_x, max_y - min_y) * 0.05) or 10
        self.canvas.axes.set_xlim(min_x - margin, max_x + margin)
        self.canvas.axes.set_ylim(min_y - margin, max_y + margin)
        
        # Show path if enabled
        if self.show_path_check.isChecked() and self.total_frames > 0:
            # Get positions up to current frame
            positions = self.simulator.positions[:self.current_frame+1]
            if positions:
                x_values = [pos[0] for pos in positions]
                y_values = [pos[1] for pos in positions]
                self.canvas.axes.plot(x_values, y_values, 'b-', alpha=0.5, linewidth=0.8)
        
        # Show dots if enabled
        if self.show_dots_check.isChecked() and self.simulator.dots:
            # Count visible dots based on current frame
            visible_dot_count = 0
            for i, (pos, color) in enumerate(zip(self.simulator.positions[:self.current_frame+1], 
                                                self.simulator.colors[:self.current_frame+1])):
                if pos[2] < 1.0:  # Check if it's a spray position
                    visible_dot_count += 1
            
            # Show visible dots
            dots = self.simulator.dots[:visible_dot_count]
            colors = self.simulator.dot_colors[:visible_dot_count]
            
            if dots:
                x_values = [dot[0] for dot in dots]
                y_values = [dot[1] for dot in dots]
                # Normalize RGB values to 0-1 range for matplotlib
                rgba_colors = [(r/255, g/255, b/255, 0.7) for r, g, b in colors]
                self.canvas.axes.scatter(x_values, y_values, s=30, c=rgba_colors, 
                                        marker='o', edgecolors='none')
        
        # Show current position
        if self.total_frames > 0 and self.current_frame < self.total_frames:
            pos, color = self.simulator.get_frame(self.current_frame)
            if pos:
                # Normalize RGB values to 0-1 range
                r, g, b = color
                color_normalized = (r/255, g/255, b/255)
                self.canvas.axes.plot(pos[0], pos[1], 'ro', markersize=8, 
                                     markeredgecolor='black', markerfacecolor=color_normalized)
        
        # Draw the origin
        self.canvas.axes.plot(0, 0, 'kx', markersize=8)
        
        # Redraw the canvas
        self.canvas.draw()
    
    def next_frame(self):
        """Move to the next frame in the animation."""
        if self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.position_slider.setValue(self.current_frame)
            self.position_label.setText(f"{self.current_frame} / {self.total_frames}")
            self.update_visualization()
        else:
            # Stop playback at the end
            self.is_playing = False
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
    
    def toggle_play(self):
        """Toggle animation playback."""
        if self.is_playing:
            self.is_playing = False
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.is_playing = True
            # Restart if at the end
            if self.current_frame >= self.total_frames - 1:
                self.current_frame = 0
                self.position_slider.setValue(0)
            
            self.play_timer.start(50)  # 50 ms per frame
            self.play_btn.setText("⏸ Pause")
    
    def reset_animation(self):
        """Reset animation to the beginning."""
        self.current_frame = 0
        self.position_slider.setValue(0)
        self.position_label.setText(f"0 / {self.total_frames}")
        self.update_visualization()
    
    def slider_changed(self, value):
        """Handle slider value changes."""
        self.current_frame = value
        self.position_label.setText(f"{value} / {self.total_frames}")
        self.update_visualization()
