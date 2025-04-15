import matplotlib.pyplot as plt
import pandas as pd
import os
from data_generation.sensor_data_generator import SensorDataGenerator

def plot_sensor_data(data, sensor_type, output_dir="plots"):
    """
    Create plots of sensor data to visualize normal vs. failure patterns
    
    Args:
        data (pd.DataFrame): Sensor data
        sensor_type (str): Type of sensor to plot
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter data for the specific sensor type
    sensor_data = data[data['sensor_id'].str.startswith(sensor_type)]
    
    # Get unique sensor IDs for this type
    sensor_ids = sensor_data['sensor_id'].unique()
    
    for sensor_id in sensor_ids:
        # Filter data for this specific sensor
        single_sensor_data = sensor_data[sensor_data['sensor_id'] == sensor_id]
        
        # Skip if not enough data
        if len(single_sensor_data) < 10:
            continue
        
        plt.figure(figsize=(12, 6))
        
        # Group by status and plot each group
        status_groups = single_sensor_data.groupby('status')
        
        for status, group in status_groups:
            plt.scatter(
                group['timestamp'], 
                group['value'], 
                label=status,
                alpha=0.7,
                s=20
            )
        
        # Add reference lines for normal operating range
        generator = SensorDataGenerator()
        # Extract the sensor type properly
        sensor_key = sensor_type
        # Handle special case for soil_moisture
        if sensor_type.startswith('soil_moisture'):
            sensor_key = 'soil_moisture'
            
        sensor_range = generator.sensor_types[sensor_key]
        
        plt.axhline(y=sensor_range['min'], color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=sensor_range['max'], color='green', linestyle='--', alpha=0.5)
        
        plt.title(f"Sensor {sensor_id} Data")
        plt.xlabel("Timestamp")
        plt.ylabel(f"Value ({sensor_range['unit']})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis to show dates nicely
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        plt.tight_layout()
        file_path = os.path.join(output_dir, f"{sensor_id}_data.png")
        plt.savefig(file_path)
        plt.close()
        
        print(f"Plot saved to {file_path}")

def main():
    # Create data directory if it doesn't exist
    os.makedirs("../../data", exist_ok=True)
    
    # Create plots directory
    plots_dir = "../../data/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate sensor data
    generator = SensorDataGenerator("../../data")
    dataset = generator.generate_dataset(normal_samples=2000, failure_samples=300)
    generator.save_dataset(dataset, filename="sensor_data.csv")
    
    # Plot data for each sensor type
    for sensor_type in ["temperature", "humidity", "soil_moisture"]:
        plot_sensor_data(dataset, sensor_type, plots_dir)
    
    print("Data generation and visualization complete.")

if __name__ == "__main__":
    main() 