import os
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data_preprocessing.preprocessor import SensorDataPreprocessor

def main():
    parser = argparse.ArgumentParser(description='Preprocess IoT sensor data for predictive maintenance')
    
    parser.add_argument('--input', type=str, default='data/sensor_data.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--output', type=str, default='data/preprocessed_data.csv',
                        help='Path to save the preprocessed data')
    parser.add_argument('--windows', type=str, default='5,10,20',
                        help='Comma-separated list of window sizes for rolling features')
    
    args = parser.parse_args()
    
    window_sizes = [int(w) for w in args.windows.split(',')]
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {args.input}...")
    preprocessor = SensorDataPreprocessor()
    
    data = preprocessor.load_data(args.input)
    print(f"Loaded data with shape: {data.shape}")
    
    print("Extracting features...")
    features_df = preprocessor.extract_features(data)
    
    print(f"Creating rolling features with windows {window_sizes}...")
    processed_df = preprocessor.create_rolling_features(features_df, window_sizes)
    
    preprocessor.save_preprocessed_data(processed_df, args.output)
    
    print("\nPreprocessing summary:")
    print(f"Original data shape: {data.shape}")
    print(f"Preprocessed data shape: {processed_df.shape}")
    print(f"New features added: {processed_df.shape[1] - data.shape[1]}")
    print("\nTarget distribution:")
    print(processed_df['target'].value_counts())
    print("\nFeature types:")
    print(processed_df.dtypes.value_counts())

if __name__ == "__main__":
    main() 