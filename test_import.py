"""
Test file to check if GcodeTab can be imported properly.
"""

try:
    from gcode_tab import GcodeTab
    print("SUCCESS: GcodeTab was successfully imported!")
except ImportError as e:
    print(f"FAILED: Could not import GcodeTab. Error: {e}")
