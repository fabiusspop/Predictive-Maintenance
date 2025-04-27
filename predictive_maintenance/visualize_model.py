import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from training.visualize_model import main as visualize_main

if __name__ == "__main__":
    visualize_main() 