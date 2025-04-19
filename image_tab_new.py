"""
Image processing tab for MuralBot application.
"""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

class ImageTabNew(QWidget):
    """Tab for image processing and G-code generation."""
    
    # Signal emitted when G-code is generated
    gcode_generated = pyqtSignal(str, str)
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        print("ImageTab initialized")
