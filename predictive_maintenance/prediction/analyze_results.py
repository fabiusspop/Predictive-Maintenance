import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

def load_prediction_results(results_path):
    """Load prediction results from CSV
    
    Args:
        results_path (str): Path to results CSV
        
    Returns:
        pd.DataFrame: Prediction results
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    results = pd.read_csv(results_path)
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in results.columns:
        results['timestamp'] = pd.to_datetime(results['timestamp'])
    
    return results

def load_maintenance_report(report_path):
    """Load maintenance report from JSON
    
    Args:
        report_path (str): Path to report JSON
        
    Returns:
        dict: Maintenance report
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file not found: {report_path}")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    return report

def analyze_prediction_accuracy(results):
    """Analyze prediction accuracy
    
    Args:
        results (pd.DataFrame): Prediction results
        
    Returns:
        dict: Accuracy metrics
    """
    # Check if we have true labels
    if 'status' not in results.columns:
        return None
    
    # Convert true status to binary
    true_status = results['status'].apply(lambda x: 1 if x != 'normal' else 0)
    
    # Convert predicted status to binary
    pred_status = results['predicted_status'].apply(lambda x: 1 if x == 'failure' else 0)
    
    # Calculate metrics
    accuracy = (true_status == pred_status).mean()
    true_positive = ((true_status == 1) & (pred_status == 1)).sum()
    false_positive = ((true_status == 0) & (pred_status == 1)).sum()
    true_negative = ((true_status == 0) & (pred_status == 0)).sum()
    false_negative = ((true_status == 1) & (pred_status == 0)).sum()
    
    # Calculate precision, recall, F1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "total_samples": len(results),
        "predicted_failures": pred_status.sum(),
        "actual_failures": true_status.sum()
    }
    
    return metrics

def analyze_sensor_types(results):
    """Analyze predictions by sensor type
    
    Args:
        results (pd.DataFrame): Prediction results
        
    Returns:
        pd.DataFrame: Prediction metrics by sensor type
    """
    if 'sensor_type' not in results.columns:
        return None
    
    # Group by sensor type
    sensor_types = []
    
    for sensor_type, group in results.groupby('sensor_type'):
        # Count total, predicted failures, and actual failures
        total = len(group)
        pred_failures = (group['predicted_status'] == 'failure').sum()
        
        # Calculate actual failures if available
        if 'status' in group.columns:
            actual_failures = (group['status'] != 'normal').sum()
            true_positive = ((group['status'] != 'normal') & (group['predicted_status'] == 'failure')).sum()
            accuracy = ((group['status'] != 'normal') == (group['predicted_status'] == 'failure')).mean()
        else:
            actual_failures = None
            true_positive = None
            accuracy = None
        
        sensor_types.append({
            "sensor_type": sensor_type,
            "total_readings": total,
            "predicted_failures": pred_failures,
            "actual_failures": actual_failures,
            "true_positives": true_positive,
            "accuracy": accuracy,
            "failure_rate": pred_failures / total
        })
    
    return pd.DataFrame(sensor_types)

def plot_confusion_matrix(metrics, output_dir="plots"):
    """Plot confusion matrix
    
    Args:
        metrics (dict): Accuracy metrics
        output_dir (str): Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create confusion matrix
    cm = [
        [metrics['true_negative'], metrics['false_positive']],
        [metrics['false_negative'], metrics['true_positive']]
    ]
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Failure'],
        yticklabels=['Normal', 'Failure']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Add accuracy metrics
    plt.figtext(
        0.5, 
        0.01, 
        f"Accuracy: {metrics['accuracy']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}",
        ha='center'
    )
    
    # Save
    filepath = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    print(f"Confusion matrix saved to {filepath}")

def plot_sensor_type_metrics(sensor_metrics, output_dir="plots"):
    """Plot sensor type metrics
    
    Args:
        sensor_metrics (pd.DataFrame): Metrics by sensor type
        output_dir (str): Directory to save the plot
    """
    if sensor_metrics is None or len(sensor_metrics) == 0:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot failure rates by sensor type
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sensor_type', y='failure_rate', data=sensor_metrics)
    plt.title('Failure Rate by Sensor Type')
    plt.xlabel('Sensor Type')
    plt.ylabel('Failure Rate')
    
    # Save
    filepath = os.path.join(output_dir, "failure_rate_by_sensor_type.png")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    
    print(f"Failure rate plot saved to {filepath}")
    
    # If we have accuracy data, plot that too
    if 'accuracy' in sensor_metrics.columns and sensor_metrics['accuracy'].notna().any():
        plt.figure(figsize=(10, 6))
        sns.barplot(x='sensor_type', y='accuracy', data=sensor_metrics)
        plt.title('Prediction Accuracy by Sensor Type')
        plt.xlabel('Sensor Type')
        plt.ylabel('Accuracy')
        
        # Save
        filepath = os.path.join(output_dir, "accuracy_by_sensor_type.png")
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()
        
        print(f"Accuracy plot saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Analyze prediction results')
    
    parser.add_argument('--results', type=str, default='data/predictions/prediction_results.csv',
                        help='Path to prediction results CSV')
    parser.add_argument('--report', type=str, default='data/predictions/maintenance_report.json',
                        help='Path to maintenance report JSON')
    parser.add_argument('--output-dir', type=str, default='data/predictions/analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prediction results
    print(f"Loading prediction results from {args.results}...")
    results = load_prediction_results(args.results)
    
    # Load maintenance report
    print(f"Loading maintenance report from {args.report}...")
    report = load_maintenance_report(args.report)
    
    # Analyze prediction accuracy
    metrics = analyze_prediction_accuracy(results)
    if metrics:
        print("\nPrediction Accuracy Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"True Positives: {metrics['true_positive']}")
        print(f"False Positives: {metrics['false_positive']}")
        print(f"True Negatives: {metrics['true_negative']}")
        print(f"False Negatives: {metrics['false_negative']}")
        
        # Plot confusion matrix
        plot_confusion_matrix(metrics, args.output_dir)
    
    # Analyze by sensor type
    sensor_metrics = analyze_sensor_types(results)
    if sensor_metrics is not None:
        print("\nSensor Type Metrics:")
        print(sensor_metrics)
        
        # Plot sensor type metrics
        plot_sensor_type_metrics(sensor_metrics, args.output_dir)
    
    # Print maintenance report summary
    print("\nMaintenance Report Summary:")
    print(f"Total Sensors: {report['total_sensors']}")
    print(f"Sensors Needing Maintenance: {report['sensors_needing_maintenance']}")
    print("Failures by Sensor Type:")
    for sensor_type, count in report['failures_by_sensor_type'].items():
        print(f"  {sensor_type}: {count}")
    
    # Save analysis results to text file
    summary_path = os.path.join(args.output_dir, "prediction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Prediction Analysis Summary\n")
        f.write("==========================\n\n")
        
        if metrics:
            f.write("Prediction Accuracy Metrics:\n")
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall: {metrics['recall']:.4f}\n")
            f.write(f"F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"True Positives: {metrics['true_positive']}\n")
            f.write(f"False Positives: {metrics['false_positive']}\n")
            f.write(f"True Negatives: {metrics['true_negative']}\n")
            f.write(f"False Negatives: {metrics['false_negative']}\n\n")
        
        f.write("Maintenance Report Summary:\n")
        f.write(f"Total Sensors: {report['total_sensors']}\n")
        f.write(f"Sensors Needing Maintenance: {report['sensors_needing_maintenance']}\n")
        f.write("Failures by Sensor Type:\n")
        for sensor_type, count in report['failures_by_sensor_type'].items():
            f.write(f"  {sensor_type}: {count}\n")
    
    print(f"\nAnalysis summary saved to {summary_path}")

if __name__ == "__main__":
    main() 