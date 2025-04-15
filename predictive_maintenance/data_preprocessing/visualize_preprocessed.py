import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import seaborn as sns

def load_preprocessed_data(filepath):
    """Load preprocessed data from CSV
    
    Args:
        filepath (str): Path to preprocessed data CSV
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessed data not found at {filepath}")
    
    data = pd.read_csv(filepath)
    # Convert timestamp to datetime
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    return data

def plot_feature_distributions(data, output_dir="plots"):
    """Create histograms of feature distributions
    
    Args:
        data (pd.DataFrame): Preprocessed data
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numerical_features = [f for f in numerical_features if f not in ['target', 'sensor_number']]
    
    # Create feature distribution plots
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(numerical_features[:9], 1):  # Plot first 9 features
        plt.subplot(3, 3, i)
        sns.histplot(data=data, x=feature, hue='target', bins=30, alpha=0.6, kde=True)
        plt.title(f"{feature} Distribution")
        plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"))
    plt.close()
    
    print(f"Feature distribution plot saved to {os.path.join(output_dir, 'feature_distributions.png')}")

def plot_correlation_matrix(data, output_dir="plots"):
    """Create correlation matrix heatmap
    
    Args:
        data (pd.DataFrame): Preprocessed data
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Calculate correlation matrix
    corr_matrix = data[numerical_features].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()
    
    print(f"Correlation matrix saved to {os.path.join(output_dir, 'correlation_matrix.png')}")

def plot_rolling_features(data, output_dir="plots"):
    """Plot rolling features for each sensor type
    
    Args:
        data (pd.DataFrame): Preprocessed data
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique sensor types
    sensor_types = data['sensor_type'].unique()
    
    for sensor_type in sensor_types:
        # Filter data for this sensor type
        sensor_data = data[data['sensor_type'] == sensor_type]
        
        # Get a sample sensor for this type
        sample_sensor_id = sensor_data['sensor_id'].iloc[0]
        sample_data = sensor_data[sensor_data['sensor_id'] == sample_sensor_id]
        
        # Plot rolling mean features
        plt.figure(figsize=(15, 8))
        
        rolling_cols = [col for col in sample_data.columns if col.startswith('rolling_mean')]
        
        for col in rolling_cols:
            plt.plot(
                sample_data['timestamp'],
                sample_data[col],
                label=col,
                alpha=0.7
            )
        
        plt.plot(
            sample_data['timestamp'],
            sample_data['value'],
            label='Original Value',
            color='black',
            alpha=0.5,
            linestyle='--'
        )
        
        plt.title(f"Rolling Means for {sample_sensor_id}")
        plt.xlabel("Timestamp")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{sensor_type}_rolling_means.png"))
        plt.close()
        
        print(f"Rolling means plot saved to {os.path.join(output_dir, f'{sensor_type}_rolling_means.png')}")

def main():
    parser = argparse.ArgumentParser(description='Visualize preprocessed sensor data')
    
    parser.add_argument('--input', type=str, default='data/preprocessed_data.csv',
                        help='Path to the preprocessed data CSV file')
    parser.add_argument('--output', type=str, default='data/plots/preprocessed',
                        help='Directory to save visualization plots')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.input}...")
    data = load_preprocessed_data(args.input)
    
    # Create visualizations
    print("Creating feature distribution plots...")
    plot_feature_distributions(data, args.output)
    
    print("Creating correlation matrix...")
    plot_correlation_matrix(data, args.output)
    
    print("Creating rolling feature plots...")
    plot_rolling_features(data, args.output)
    
    print("Visualization complete!")

if __name__ == "__main__":
    main() 