import os
import sys
import argparse
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from data_generation.sensor_data_generator import SensorDataGenerator
from data_generation.visualize_data import plot_sensor_data

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic IoT sensor data for predictive maintenance')
    
    parser.add_argument('--normal', type=int, default=5000,
                        help='Number of normal data points per sensor type')
    parser.add_argument('--failure', type=int, default=500,
                        help='Number of failure data points per sensor type')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations of the data')
    parser.add_argument('--output', type=str, default='data',
                        help='Output directory for data files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create plots directory if visualizing
    plots_dir = os.path.join(args.output, 'plots')
    if args.visualize:
        os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Generating data ({args.normal} normal samples, {args.failure} failure samples)...")
    generator = SensorDataGenerator(args.output)
    dataset = generator.generate_dataset(normal_samples=args.normal, failure_samples=args.failure)
    generator.save_dataset(dataset)
    
    if args.visualize:
        print("Generating visualizations...")
        for sensor_type in ["temperature", "humidity", "soil_moisture"]:
            plot_sensor_data(dataset, sensor_type, plots_dir)
    
    print("Data generation complete!")
    print(f"Dataset shape: {dataset.shape}")
    print("Status distribution:")
    print(dataset['status'].value_counts())

if __name__ == "__main__":
    main() 