import os
import sys

from dotenv import load_dotenv

# Load .env if it exists
load_dotenv()

print("CONFTEST")
print(sys.path)

# Ensure the project root is in the path for Jupyter
sys.path.insert(0, os.getcwd())
