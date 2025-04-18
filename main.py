#!/usr/bin/env python3
"""
MuralBot - Image to G-code Generator for Wall Painting Robot
"""

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
from PyQt5.QtCore import QSettings

from config_tab import ConfigTab
from image_tab import ImageTab
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
