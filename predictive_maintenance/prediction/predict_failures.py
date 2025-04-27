import os
import sys
import argparse
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from data_generation.sensor_data_generator import SensorDataGenerator
from prediction.predictor import MaintenancePredictor

def generate_test_data(output_file="data/test_data.csv", normal_samples=1000, failure_samples=200):
    """Generate test data for prediction testing
    
    Args:
        output_file (str): Path to save the generated data
        normal_samples (int): Number of normal samples to generate
        failure_samples (int): Number of failure samples to generate
        
    Returns:
        pd.DataFrame: Generated test data
    """
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    generator = SensorDataGenerator(output_dir)
    
    # generate data
    print(f"Generating test data ({normal_samples} normal, {failure_samples} failure samples)...")
    data = generator.generate_dataset(
        normal_samples=normal_samples,
        failure_samples=failure_samples
    )
    
    generator.save_dataset(data, os.path.basename(output_file))
    
    print(f"Test data saved to {output_file}")
    return data

def main():
    parser = argparse.ArgumentParser(description='Predict sensor failures using the trained model')
    
    parser.add_argument('--model', type=str, default='models/random_forest.joblib',
                        help='Path to the trained model file')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to input data (if not provided, generate new data)')
    parser.add_argument('--output-dir', type=str, default='data/predictions',
                        help='Directory to save prediction results')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for failure prediction')
    parser.add_argument('--generate-normal', type=int, default=1000,
                        help='Number of normal samples to generate (if generating data)')
    parser.add_argument('--generate-failure', type=int, default=200,
                        help='Number of failure samples to generate (if generating data)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.input is None:
        # generate new data
        test_data_path = os.path.join(args.output_dir, "test_data.csv")
        test_data = generate_test_data(
            test_data_path,
            args.generate_normal,
            args.generate_failure
        )
    else:
        # load existing data
        if not os.path.exists(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        print(f"Loading test data from {args.input}...")
        test_data = pd.read_csv(args.input)
        test_data_path = args.input
    
    # predictor
    print(f"Loading model from {args.model}...")
    predictor = MaintenancePredictor(args.model, threshold=args.threshold)
    
    print("Preprocessing test data...")
    processed_data = predictor.preprocess_data(test_data)
    
    print("Making predictions...")
    predictor.predict(processed_data)
    
    results = predictor.get_prediction_results(test_data)
    
    results_path = os.path.join(args.output_dir, "prediction_results.csv")
    predictor.save_prediction_results(results, results_path)
    
    # prediction visualization
    plots_dir = os.path.join(args.output_dir, "plots")
    print(f"Creating prediction visualizations in {plots_dir}...")
    predictor.visualize_predictions(results, plots_dir)
    
    # maintenance report
    report_path = os.path.join(args.output_dir, "maintenance_report.json")
    print(f"Generating maintenance report...")
    predictor.generate_maintenance_report(results, report_path)
    
    # statistics
    print("\nPrediction Summary:")
    print(f"Total samples: {len(results)}")
    print(f"Predicted failures: {(results['predicted_status'] == 'failure').sum()}")
    print(f"Sensors needing maintenance: {len(predictor.get_sensors_needing_maintenance(results))}")
    
    # If true labels -> calc acc
    if 'status' in results.columns:
        true_status = results['status'].apply(lambda x: 1 if x != 'normal' else 0)
        pred_status = results['predicted_status'].apply(lambda x: 1 if x == 'failure' else 0)
        
        accuracy = (true_status == pred_status).mean()
        print(f"Prediction accuracy: {accuracy:.4f}")
        
        true_positive = ((true_status == 1) & (pred_status == 1)).sum()
        false_positive = ((true_status == 0) & (pred_status == 1)).sum()
        true_negative = ((true_status == 0) & (pred_status == 0)).sum()
        false_negative = ((true_status == 1) & (pred_status == 0)).sum()
        
        print(f"True positives: {true_positive}")
        print(f"False positives: {false_positive}")
        print(f"True negatives: {true_negative}")
        print(f"False negatives: {false_negative}")
    
    print(f"\nPrediction results saved to {results_path}")
    print(f"Maintenance report saved to {report_path}")
    print(f"Prediction plots saved to {plots_dir}")

if __name__ == "__main__":
    main() 