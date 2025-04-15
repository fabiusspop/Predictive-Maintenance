import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from training.visualize_model import main as visualize_main

if __name__ == "__main__":
    visualize_main() 