import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_preprocessing.visualize_preprocessed import main as visualize_main

if __name__ == "__main__":
    visualize_main() 