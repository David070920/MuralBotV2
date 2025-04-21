"""
Configuration tab for MuralBot application.
"""

from PyQt5.QtWidgets import (QWidget, QFormLayout, QLabel, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QVBoxLayout, QPushButton,
                            QGroupBox, QHBoxLayout, QCheckBox, QComboBox,
                            QTableWidget, QTableWidgetItem, QHeaderView)
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
        
        # Color Mode Settings group
        color_mode_group = QGroupBox("Color Mode Settings")
        color_mode_layout = QFormLayout()
        
        self.color_mode = QComboBox()
        self.color_mode.addItems(["Automatic", "Manual RGB", "Default Colors"])
        self.color_mode.currentIndexChanged.connect(self.update_color_mode)
        
        # Color table for manual RGB input
        self.color_table = QTableWidget(0, 3)
        self.color_table.setHorizontalHeaderLabels(["Red", "Green", "Blue"])
        self.color_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.color_table.setVisible(False)
        
        # Add/Remove buttons for manual color table
        color_buttons_layout = QHBoxLayout()
        self.add_color_btn = QPushButton("Add Color")
        self.add_color_btn.clicked.connect(self.add_color_row)
        self.remove_color_btn = QPushButton("Remove Selected")
        self.remove_color_btn.clicked.connect(self.remove_color_row)
        color_buttons_layout.addWidget(self.add_color_btn)
        color_buttons_layout.addWidget(self.remove_color_btn)
        
        # Hide buttons initially
        self.add_color_btn.setVisible(False)
        self.remove_color_btn.setVisible(False)
        
        color_mode_layout.addRow("Color Selection Mode:", self.color_mode)
        color_mode_layout.addWidget(self.color_table)
        color_mode_layout.addRow(color_buttons_layout)
        color_mode_group.setLayout(color_mode_layout)
        
        # Advanced settings group
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout()
        
        self.optimize_path = QCheckBox()
        self.optimize_path.setChecked(True)
        
        self.border_margin = QDoubleSpinBox()
        self.border_margin.setRange(0.0, 500.0)
        self.border_margin.setValue(50.0)
        self.border_margin.setSuffix(" mm")
        self.border_margin.setDecimals(1)
        
        self.home_position = QComboBox()
        self.home_position.addItems(["Bottom Left", "Bottom Right", "Top Left", "Top Right"])
        
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
        main_layout.addWidget(color_mode_group)
        main_layout.addWidget(advanced_group)
        main_layout.addLayout(button_layout)
        main_layout.addStretch()
        
        self.setLayout(main_layout)
    
    def get_config(self):
        """Get the current configuration values."""
        config = {
            "wall_width": self.wall_width.value(),
            "wall_height": self.wall_height.value(),
            "total_colors": self.total_colors.value(),
            "colors_per_batch": self.colors_per_batch.value(),
            "dot_resolution": self.dot_resolution.value(),
            "speed_xy": self.speed_xy.value(),
            "speed_z": self.speed_z.value(),
            "spray_delay": self.spray_delay.value(),
            "optimize_path": self.optimize_path.isChecked(),
            "border_margin": self.border_margin.value(),
            "home_position": self.home_position.currentText(),
            "color_mode": self.color_mode.currentText()
        }
        
        # Add manual colors if in manual RGB mode
        if self.color_mode.currentText() == "Manual RGB":
            manual_colors = []
            for row in range(self.color_table.rowCount()):
                color = []
                for col in range(3):
                    spin_box = self.color_table.cellWidget(row, col)
                    color.append(spin_box.value())
                manual_colors.append(color)
            config["manual_colors"] = manual_colors
        
        # Add default colors if in Default Colors mode
        elif self.color_mode.currentText() == "Default Colors":
            # Red, Green, Yellow, Blue, Black, White
            config["default_colors"] = [
                [255, 0, 0],     # Red
                [0, 255, 0],     # Green
                [255, 255, 0],   # Yellow
                [0, 0, 255],     # Blue
                [0, 0, 0],       # Black
                [255, 255, 255]  # White
            ]
        
        return config
    
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
        if self.settings.contains("config/optimize_path"):
            self.optimize_path.setChecked(self.settings.value("config/optimize_path") == "true")
        if self.settings.contains("config/border_margin"):
            self.border_margin.setValue(float(self.settings.value("config/border_margin")))
        if self.settings.contains("config/home_position"):
            self.home_position.setCurrentText(self.settings.value("config/home_position"))
        if self.settings.contains("config/color_mode"):
            self.color_mode.setCurrentText(self.settings.value("config/color_mode"))
            
            # Load manual colors if in Manual RGB mode
            if self.color_mode.currentText() == "Manual RGB" and self.settings.contains("config/manual_colors"):
                manual_colors = self.settings.value("config/manual_colors")
                if manual_colors:
                    # Clear existing rows
                    while self.color_table.rowCount() > 0:
                        self.color_table.removeRow(0)
                    
                    # Add saved colors
                    for color in manual_colors:
                        row = self.color_table.rowCount()
                        self.color_table.insertRow(row)
                        for col in range(3):
                            spin_box = QSpinBox()
                            spin_box.setRange(0, 255)
                            spin_box.setValue(color[col])
                            self.color_table.setCellWidget(row, col, spin_box)
            
            # Update UI based on loaded color mode
            self.update_color_mode()
    
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
        self.optimize_path.setChecked(True)
        self.border_margin.setValue(50.0)
        self.home_position.setCurrentText("Bottom Left")
    
    def update_color_mode(self):
        """Update UI based on selected color mode"""
        mode = self.color_mode.currentText()
        
        # Hide color table and buttons by default
        self.color_table.setVisible(False)
        self.add_color_btn.setVisible(False)
        self.remove_color_btn.setVisible(False)
        
        # Show relevant UI elements based on mode
        if mode == "Manual RGB":
            self.color_table.setVisible(True)
            self.add_color_btn.setVisible(True)
            self.remove_color_btn.setVisible(True)
            
            # Initialize with at least one row if empty
            if self.color_table.rowCount() == 0:
                self.add_color_row()
                
        elif mode == "Default Colors":
            # Default colors are predefined (red, green, yellow, blue, black, white)
            # No additional UI needed as these are hardcoded
            pass
            
        # Update total colors spin box based on mode
        if mode == "Automatic":
            self.total_colors.setEnabled(True)
        elif mode == "Manual RGB":
            self.total_colors.setValue(self.color_table.rowCount())
            self.total_colors.setEnabled(False)
        elif mode == "Default Colors":
            self.total_colors.setValue(6)  # 6 default colors
            self.total_colors.setEnabled(False)
    
    def add_color_row(self):
        """Add a row to the color table for manual RGB input"""
        row = self.color_table.rowCount()
        self.color_table.insertRow(row)
        
        # Add RGB spin boxes (0-255)
        for col in range(3):
            spin_box = QSpinBox()
            spin_box.setRange(0, 255)
            # Set default values to make a random color
            default_values = [255, 0, 0]  # Default to red
            spin_box.setValue(default_values[col])
            self.color_table.setCellWidget(row, col, spin_box)
        
        # Update total colors to match row count
        self.total_colors.setValue(self.color_table.rowCount())
    
    def remove_color_row(self):
        """Remove selected row from color table"""
        selected_rows = self.color_table.selectedIndexes()
        if selected_rows:
            row = selected_rows[0].row()
            self.color_table.removeRow(row)
            # Update total colors to match row count
            self.total_colors.setValue(self.color_table.rowCount())
            
            # Make sure we have at least one color
            if self.color_table.rowCount() == 0:
                self.add_color_row()
