#!/usr/bin/env python3
"""
MuralBot - Image to G-code Generator for Wall Painting Robot
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QMessageBox, QWidget,
                            QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                            QFileDialog, QProgressBar, QGroupBox, QFormLayout, QTableWidget,
                            QTableWidgetItem, QHeaderView, QSplitter, QScrollArea,
                            QGridLayout, QRadioButton, QButtonGroup)
from PyQt5.QtCore import QSettings, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QColor, QImage

from config_tab import ConfigTab
from image_processor import ImageProcessor

# Define ColorPaletteTable class
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

# Define ImageTab class
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
        self.dithering_previews = {}
        self.selected_dithering_method = None
        
        # Available dithering methods
        self.dithering_methods = [
            "Floyd-Steinberg",
            "Ordered",
            "Jarvis-Judice-Ninke",
            "Stucki",
            "Atkinson",
            "Sierra",
            "Enhanced Floyd-Steinberg",
            "Blue Noise",
            "Pattern",
            "Halftone",
            "None"
        ]
        
        # Initialize the image processor
        self.image_processor = ImageProcessor()
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Top section - Image and Palette
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Image preview group
        image_group = QGroupBox("Original Image")
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
        
        # Dithering preview section
        dithering_group = QGroupBox("Dithering Previews")
        dithering_layout = QVBoxLayout()
        
        # Create a scroll area for the dithering previews
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.dithering_grid = QGridLayout(scroll_content)
        
        # We'll populate this grid with previews when the user clicks "Preview Processing"
        scroll_area.setWidget(scroll_content)
        dithering_layout.addWidget(scroll_area)
        
        # Create a button group for the radio buttons
        self.dithering_button_group = QButtonGroup(self)
        self.dithering_button_group.buttonClicked.connect(self.on_dithering_selected)
        
        dithering_group.setLayout(dithering_layout)
        
        # Bottom section - Processing controls
        process_group = QGroupBox("Image Processing and G-code Generation")
        process_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        # Buttons
        process_buttons_layout = QHBoxLayout()
        self.preview_btn = QPushButton("Preview Dithering Options")
        self.preview_btn.clicked.connect(self.preview_all_dithering)
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
        main_layout.addWidget(dithering_group, 3)
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
    
    def preview_all_dithering(self):
        """Generate previews for all dithering methods."""
        if not self.input_image_path or not self.config:
            return
        
        try:
            # Clear previous previews
            self.clear_dithering_grid()
            
            # Store the original dithering method
            original_method = self.config.get("dithering_method", "Floyd-Steinberg")
            
            # Create a progress step for each dithering method
            total_methods = len(self.dithering_methods)
            progress_step = 100 / total_methods
            
            # Generate a preview for each dithering method
            for i, method in enumerate(self.dithering_methods):
                # Update progress
                self.progress_bar.setValue(int(i * progress_step))
                
                # Update config with current dithering method
                self.config["dithering_method"] = method
                self.image_processor.set_config(self.config)
                
                # Process the image with this dithering method
                processed_image, palette, batches, color_counts = self.image_processor.process_image(
                    self.input_image_path, preview_only=True
                )
                
                # Store the palette for the first method (they should all be similar)
                if i == 0:
                    self.color_table.set_palette(palette, batches, color_counts)
                
                # Add the preview to the grid
                self.add_dithering_preview(method, processed_image)
                
                # Store this processed image for later use
                self.dithering_previews[method] = processed_image
            
            # Reset to the original method for consistency
            self.config["dithering_method"] = original_method
            self.image_processor.set_config(self.config)
            
            # Set progress to 100%
            self.progress_bar.setValue(100)
            
            # Enable the generate button only after a dithering method is selected
            self.generate_btn.setEnabled(False)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generating dithering previews: {str(e)}")
    
    def clear_dithering_grid(self):
        """Clear all items from the dithering grid."""
        # Remove all widgets from the grid
        while self.dithering_grid.count():
            item = self.dithering_grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Clear the button group
        for button in self.dithering_button_group.buttons():
            self.dithering_button_group.removeButton(button)
    
    def add_dithering_preview(self, method, image):
        """Add a preview for a specific dithering method to the grid."""
        # Calculate grid position (3 columns)
        row = len(self.dithering_button_group.buttons()) // 3
        col = len(self.dithering_button_group.buttons()) % 3
        
        # Create a container for the preview and radio button
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # Create the preview label
        preview_label = QLabel()
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setMinimumSize(200, 150)
        preview_label.setStyleSheet("border: 1px solid #cccccc;")
        
        # Convert the image to a pixmap and display it
        height, width, _ = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        pixmap = pixmap.scaled(
            200, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        preview_label.setPixmap(pixmap)
        
        # Create the radio button
        radio_button = QRadioButton(method)
        self.dithering_button_group.addButton(radio_button)
        
        # Add the preview and radio button to the container
        container_layout.addWidget(preview_label)
        container_layout.addWidget(radio_button)
        
        # Add the container to the grid
        self.dithering_grid.addWidget(container, row, col)
    
    def on_dithering_selected(self, button):
        """Handle selection of a dithering method."""
        # Get the text from the radio button (which is the method name)
        self.selected_dithering_method = button.text()
        
        # Update the config with the selected method
        self.config["dithering_method"] = self.selected_dithering_method
        self.image_processor.set_config(self.config)
        
        # Enable the generate button
        self.generate_btn.setEnabled(True)
    
    def generate_gcode(self):
        """Generate G-code from the processed image using the selected dithering method."""
        if not self.input_image_path or not self.config or not self.selected_dithering_method:
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
                    f"G-code has been generated with {self.selected_dithering_method} dithering and saved to {gcode_file_path}"
                )
            
            # Set progress to 100
            self.progress_bar.setValue(100)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error generating G-code: {str(e)}")
    
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_bar.setValue(int(value))

from gcode_tab import GcodeTab

class MuralBotApp(QMainWindow):
    """Main application window for MuralBot."""
    
    def __init__(self):
        super().__init__()
        
        self.settings = QSettings("MuralBot", "MuralBotV2")
        
        self.setWindowTitle("MuralBot - Mural Painting Robot Controller")
        self.setGeometry(100, 100, 1200, 800)
        
        # Create the tab widget
        self.tabs = QTabWidget()
        
        # Create the tabs
        self.config_tab = ConfigTab(self.settings)
        self.image_tab = ImageTab(self.settings)
        self.gcode_tab = GcodeTab(self.settings)
        
        # Add tabs to widget
        self.tabs.addTab(self.config_tab, "Configuration")
        self.tabs.addTab(self.image_tab, "Image Processing")
        self.tabs.addTab(self.gcode_tab, "G-Code Visualization")
        
        # Set tab widget as the central widget
        self.setCentralWidget(self.tabs)
          # Connect signals between tabs
        self.config_tab.config_updated.connect(self.image_tab.update_config)
        self.config_tab.config_updated.connect(self.gcode_tab.update_config)
        self.image_tab.gcode_generated.connect(self.gcode_tab.load_gcode)
        
        # Load settings
        self.load_settings()
        
    def load_settings(self):
        """Load application settings."""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
    
    def closeEvent(self, event):
        """Save settings when closing the application."""
        # Save window geometry
        self.settings.setValue("geometry", self.saveGeometry())
        # Call parent class closeEvent
        super().closeEvent(event)

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = MuralBotApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
