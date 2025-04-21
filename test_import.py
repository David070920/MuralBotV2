"""
Simple test file to verify that the ConfigTab can be imported correctly.
"""

try:
    # Attempt to import ConfigTab from config_tab
    from config_tab import ConfigTab
    print("SUCCESS: ConfigTab was successfully imported!")
    print(f"ConfigTab class: {ConfigTab}")
except ImportError as e:
    print(f"FAILED: Could not import ConfigTab. Error: {e}")
    
    # Try to import the module directly to see what's in it
    try:
        import config_tab
        print(f"The config_tab module was found and contains: {dir(config_tab)}")
        print(f"Module location: {config_tab.__file__}")
    except ImportError as e:
        print(f"Could not import the config_tab module at all. Error: {e}")
