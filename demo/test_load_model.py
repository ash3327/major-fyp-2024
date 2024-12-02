"""
THIS FILE CAN ONLY BE RUN FROM THE ROOT DIRECTORY
"""
try:
    import sys
    sys.path.append(".")
    from libs import ConfigLoader
    from libs.models import ModelLib
except ImportError:
    print("Warning: This script (test_load_model.py) can only be executed from the root directory of the project.")
    sys.exit(1)


configs = ConfigLoader()
model = ModelLib(configs)