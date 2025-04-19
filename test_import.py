#!/usr/bin/env python3
"""Test file for checking imports."""

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal

class TestClass(QWidget):
    """Test class for checking imports."""
    
    test_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        print("TestClass initialized")
