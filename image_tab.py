"""
Image processing tab for MuralBot application.
"""

import os
import cv2
import numpy as np
from PIL import Image
# Replace direct ImageQt import with proper import from PyQt5
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QGroupBox, QFormLayout, QTableWidget,
                            QTableWidgetItem, QHeaderView, QSplitter, QMessageBox)
from PyQt5.QtCore import pyqtSignal, Qt, QSettings
from PyQt5.QtGui import QPixmap, QColor

from image_processor import ImageProcessor

class ColorPaletteTable(QTableWidget):
    """Table widget for displaying color palette."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(4)
        self.setHorizontalHeaderLabels(["Color", "RGB", "Batch", "Count"])
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    
    def set_palette(self, palette, batches, color_counts):
        """Update the table with color palette information."""
        self.setRowCount(len(palette))
        for i, color in enumerate(palette):
            # Color cell
            color_item = QTableWidgetItem("")
            color_item.setBackground(QColor(*color))
            self.setItem(i, 0, color_item)
            
            # RGB value cell
            rgb_item = QTableWidgetItem(f"({color[0]}, {color[1]}, {color[2]})")
            self.setItem(i, 1, rgb_item)
            
            # Batch number cell
            batch_item = QTableWidgetItem(f"{batches[i]}")
            self.setItem(i, 2, batch_item)
            
            # Color count cell
            count_item = QTableWidgetItem(f"{color_counts[i]}")
            self.setItem(i, 3, count_item)

class ImageTab(QWidget):
    """Tab for image processing and G-code generation."""
    
    # Signal emitted when G-code is generated
    gcode_generated = pyqtSignal(str, str)
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.input_image_path = None
        self.processed_image = None
        self.config = {}
        
        # Initialize the image processor
        self.image_processor = ImageProcessor()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Top section - Image and Palette
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Image preview group
        image_group = QGroupBox("Image Preview")
        image_layout = QVBoxLayout()
        
        img_buttons_layout = QHBoxLayout()
        self.load_image_btn = QPushButton("Load Image")
        self.load_image_btn.clicked.connect(self.load_image)
        img_buttons_layout.addWidget(self.load_image_btn)
        
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("border: 1px solid #cccccc;")
        
        image_layout.addLayout(img_buttons_layout)
        image_layout.addWidget(self.image_label)
        image_group.setLayout(image_layout)
        
        # Color palette group
        palette_group = QGroupBox("Color Palette")
        palette_layout = QVBoxLayout()
        
        self.color_table = ColorPaletteTable()
        palette_layout.addWidget(self.color_table)
        
        palette_group.setLayout(palette_layout)
        
        # Add image and palette to the splitter
        top_splitter.addWidget(image_group)
        top_splitter.addWidget(palette_group)
        top_splitter.setSizes([600, 400])
        
        # Bottom section - Processing controls
        process_group = QGroupBox("Image Processing and G-code Generation")
        process_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Buttons
        process_buttons_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Processing")
        self.preview_btn.clicked.connect(self.preview_processing)
        self.preview_btn.setEnabled(False)
        
        self.generate_btn = QPushButton("Generate G-code")
        self.generate_btn.clicked.connect(self.generate_gcode)
        self.generate_btn.setEnabled(False)
        
        process_buttons_layout.addWidget(self.preview_btn)
        process_buttons_layout.addWidget(self.generate_btn)
        
        # Add controls to process layout
        process_layout.addLayout(process_buttons_layout)
        process_layout.addWidget(self.progress_bar)
        
        process_group.setLayout(process_layout)
        
        # Add all components to the main layout
        main_layout.addWidget(top_splitter, 3)
        main_layout.addWidget(process_group, 1)
        
        self.setLayout(main_layout)
    
    def update_config(self, config):
        """Update the configuration from the config tab."""
        self.config = config
        self.image_processor.set_config(config)
        
        # Enable preview button if an image is loaded
        if self.input_image_path:
            self.preview_btn.setEnabled(True)
    
    def load_image(self):
        """Load an image file and display it."""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        
        if file_path:
            self.input_image_path = file_path
            
            # Load and display the image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale the image to fit within the label
                pixmap = pixmap.scaled(
                    self.image_label.width(), self.image_label.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.image_label.setPixmap(pixmap)
                
                # Enable the preview button if config is loaded
                if self.config:
                    self.preview_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Error", "Failed to load the image.")
    
    def preview_processing(self):
        """Preview the image processing without generating G-code."""
        if not self.input_image_path or not self.config:
            return
        
        try:
            # Process the image and get the palette
            self.processed_image, palette, batches, color_counts = self.image_processor.process_image(
                self.input_image_path, preview_only=True
            )
            
            # Update the palette table
            self.color_table.set_palette(palette, batches, color_counts)
            
            # Display the processed image
            height, width, _ = self.processed_image.shape
            
            # Convert numpy array to QImage directly
            # Format: RGB888 - 3 bytes per pixel (R,G,B)
            bytes_per_line = 3 * width
            q_image = QImage(self.processed_image.data, width, height, 
                            bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(pixmap)
            
            # Enable the generate button
            self.generate_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error processing image: {str(e)}")
    
    def generate_gcode(self):
        """Generate G-code from the processed image."""
        if not self.input_image_path or not self.config:
            return
        
        try:
            # Set progress to 0
            self.progress_bar.setValue(0)
            
            # Process the image and generate G-code
            gcode, output_info = self.image_processor.generate_gcode(
                self.input_image_path, 
                progress_callback=self.update_progress
            )
            
            # Save the G-code to a file
            file_name = os.path.splitext(os.path.basename(self.input_image_path))[0]
            gcode_file_path, _ = QFileDialog.getSaveFileName(
                self, "Save G-code", f"{file_name}.gcode", 
                "G-code Files (*.gcode);;All Files (*)"
            )
            
            if gcode_file_path:
                with open(gcode_file_path, 'w') as f:
                    f.write(gcode)
                
                # Emit signal to load the G-code visualization
                self.gcode_generated.emit(gcode, gcode_file_path)
                
                QMessageBox.information(
                    self, "Success", 
                    f"G-code has been generated and saved to {gcode_file_path}"
                )
            
            # Set progress to 100
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generating G-code: {str(e)}")
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(int(value))
