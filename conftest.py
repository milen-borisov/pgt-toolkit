import sys
import os

# Get the directory of the current conftest.py file (which should be the root of the project)
# and add the 'src' folder to the Python path.
sys.path.append(os.path.join(os.path.dirname(__file__), "src/pgt_toolkit"))
