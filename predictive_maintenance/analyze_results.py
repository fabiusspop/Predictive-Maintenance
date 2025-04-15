import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from prediction.analyze_results import main as analyze_main

if __name__ == "__main__":
    analyze_main() 