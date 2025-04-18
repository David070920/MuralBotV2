"""
Configuration tab for MuralBot application.
"""

from PyQt5.QtWidgets import (QWidget, QFormLayout, QLabel, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QVBoxLayout, QPushButton,
                            QGroupBox, QHBoxLayout, QCheckBox, QComboBox)
from PyQt5.QtCore import pyqtSignal, QSettings

class ConfigTab(QWidget):
    """Tab for robot and painting configuration settings."""
    
    # Signal emitted when configuration is updated
    config_updated = pyqtSignal(dict)
    
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Wall dimensions group
        wall_group = QGroupBox("Wall Dimensions")
        wall_layout = QFormLayout()
        
        self.wall_width = QDoubleSpinBox()
        self.wall_width.setRange(0.1, 100.0)
        self.wall_width.setValue(2.0)
        self.wall_width.setSuffix(" m")
        self.wall_width.setDecimals(2)
        
        self.wall_height = QDoubleSpinBox()
        self.wall_height.setRange(0.1, 100.0)
        self.wall_height.setValue(2.0)
        self.wall_height.setSuffix(" m")
        self.wall_height.setDecimals(2)
        
        wall_layout.addRow("Wall Width:", self.wall_width)
        wall_layout.addRow("Wall Height:", self.wall_height)
        wall_group.setLayout(wall_layout)
        
        # Robot settings group
        robot_group = QGroupBox("Robot Settings")
        robot_layout = QFormLayout()
        
        self.total_colors = QSpinBox()
        self.total_colors.setRange(1, 100)
        self.total_colors.setValue(8)
        
        self.colors_per_batch = QSpinBox()
        self.colors_per_batch.setRange(1, 50)
        self.colors_per_batch.setValue(4)
        
        self.dot_resolution = QDoubleSpinBox()
        self.dot_resolution.setRange(1.0, 100.0)
        self.dot_resolution.setValue(20.0)
        self.dot_resolution.setSuffix(" mm")
        self.dot_resolution.setDecimals(1)
        
        self.speed_xy = QSpinBox()
        self.speed_xy.setRange(10, 5000)
        self.speed_xy.setValue(1000)
        self.speed_xy.setSuffix(" mm/min")
        
        self.speed_z = QSpinBox()
        self.speed_z.setRange(10, 5000)
        self.speed_z.setValue(500)
        self.speed_z.setSuffix(" mm/min")
        
        self.spray_delay = QDoubleSpinBox()
        self.spray_delay.setRange(0.0, 5.0)
        self.spray_delay.setValue(0.2)
        self.spray_delay.setSuffix(" sec")
        self.spray_delay.setDecimals(2)
        
        robot_layout.addRow("Total Colors to Use:", self.total_colors)
        robot_layout.addRow("Colors per Batch:", self.colors_per_batch)
        robot_layout.addRow("Spray Dot Diameter:", self.dot_resolution)
        robot_layout.addRow("XY Movement Speed:", self.speed_xy)
        robot_layout.addRow("Z Movement Speed:", self.speed_z)
        robot_layout.addRow("Spray Delay:", self.spray_delay)
        robot_group.setLayout(robot_layout)
        
        # Advanced settings group
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()
        
        self.dithering_method = QComboBox()
        self.dithering_method.addItems(["None", "Floyd-Steinberg", "Ordered"])
        
        self.optimize_path = QCheckBox()
        self.optimize_path.setChecked(True)
        
        self.border_margin = QDoubleSpinBox()
        self.border_margin.setRange(0.0, 500.0)
        self.border_margin.setValue(50.0)
        self.border_margin.setSuffix(" mm")
        self.border_margin.setDecimals(1)
        
        self.home_position = QComboBox()
        self.home_position.addItems(["Bottom Left", "Bottom Right", "Top Left", "Top Right"])
        
        advanced_layout.addRow("Dithering Method:", self.dithering_method)
        advanced_layout.addRow("Optimize Path:", self.optimize_path)
        advanced_layout.addRow("Border Margin:", self.border_margin)
        advanced_layout.addRow("Home Position:", self.home_position)
        advanced_group.setLayout(advanced_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_config)
        self.defaults_button = QPushButton("Restore Defaults")
        self.defaults_button.clicked.connect(self.restore_defaults)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.defaults_button)
        
        # Add all layouts to main layout
        main_layout.addWidget(wall_group)
        main_layout.addWidget(robot_group)
        main_layout.addWidget(advanced_group)
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def get_config(self):
        """Get the current configuration values."""
        return {
            "wall_width": self.wall_width.value(),
            "wall_height": self.wall_height.value(),
            "total_colors": self.total_colors.value(),
            "colors_per_batch": self.colors_per_batch.value(),
            "dot_resolution": self.dot_resolution.value(),
            "speed_xy": self.speed_xy.value(),
            "speed_z": self.speed_z.value(),
            "spray_delay": self.spray_delay.value(),
            "dithering_method": self.dithering_method.currentText(),
            "optimize_path": self.optimize_path.isChecked(),
            "border_margin": self.border_margin.value(),
            "home_position": self.home_position.currentText()
        }
    
    def save_config(self):
        """Save configuration to settings and emit signal."""
        config = self.get_config()
        
        # Save to QSettings
        for key, value in config.items():
            self.settings.setValue(f"config/{key}", value)
        
        # Emit signal for other tabs
        self.config_updated.emit(config)
    
    def load_settings(self):
        """Load settings from QSettings."""
        if self.settings.contains("config/wall_width"):
            self.wall_width.setValue(float(self.settings.value("config/wall_width")))
        if self.settings.contains("config/wall_height"):
            self.wall_height.setValue(float(self.settings.value("config/wall_height")))
        if self.settings.contains("config/total_colors"):
            self.total_colors.setValue(int(self.settings.value("config/total_colors")))
        if self.settings.contains("config/colors_per_batch"):
            self.colors_per_batch.setValue(int(self.settings.value("config/colors_per_batch")))
        if self.settings.contains("config/dot_resolution"):
            self.dot_resolution.setValue(float(self.settings.value("config/dot_resolution")))
        if self.settings.contains("config/speed_xy"):
            self.speed_xy.setValue(int(self.settings.value("config/speed_xy")))
        if self.settings.contains("config/speed_z"):
            self.speed_z.setValue(int(self.settings.value("config/speed_z")))
        if self.settings.contains("config/spray_delay"):
            self.spray_delay.setValue(float(self.settings.value("config/spray_delay")))
        if self.settings.contains("config/dithering_method"):
            self.dithering_method.setCurrentText(self.settings.value("config/dithering_method"))
        if self.settings.contains("config/optimize_path"):
            self.optimize_path.setChecked(self.settings.value("config/optimize_path") == "true")
        if self.settings.contains("config/border_margin"):
            self.border_margin.setValue(float(self.settings.value("config/border_margin")))
        if self.settings.contains("config/home_position"):
            self.home_position.setCurrentText(self.settings.value("config/home_position"))
    
    def restore_defaults(self):
        """Restore default configuration values."""
        self.wall_width.setValue(2.0)
        self.wall_height.setValue(2.0)
        self.total_colors.setValue(8)
        self.colors_per_batch.setValue(4)
        self.dot_resolution.setValue(20.0)
        self.speed_xy.setValue(1000)
        self.speed_z.setValue(500)
        self.spray_delay.setValue(0.2)
        self.dithering_method.setCurrentText("Floyd-Steinberg")
        self.optimize_path.setChecked(True)
        self.border_margin.setValue(50.0)
        self.home_position.setCurrentText("Bottom Left")
