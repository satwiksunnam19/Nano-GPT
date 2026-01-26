"""
Simple command-line argument parser for train.py
Allows overriding config values via command line arguments
Usage: python train.py --batch_size=4 --device=mps
"""
import sys

# Parse command line arguments
for arg in sys.argv[1:]:
    if arg.startswith('--'):
        # Remove '--' and split on '='
        key_value = arg[2:].split('=')
        if len(key_value) == 2:
            key, value = key_value
            # Try to convert to appropriate type
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                # Try to convert to number (int or float)
                try:
                    # Try int first
                    if '.' not in value and 'e' not in value.lower():
                        value = int(value)
                    else:
                        # Try float (handles 1e-3, 0.001, etc.)
                        value = float(value)
                except ValueError:
                    # Keep as string if conversion fails
                    pass
            # Set the global variable
            globals()[key] = value
            print(f"Override: {key} = {value}")
