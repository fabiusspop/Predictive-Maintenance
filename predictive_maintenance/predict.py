import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from prediction.predict_failures import main as predict_main

if __name__ == "__main__":
    predict_main() 