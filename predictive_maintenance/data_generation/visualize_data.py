import matplotlib.pyplot as plt
import pandas as pd
import os
from data_generation.sensor_data_generator import SensorDataGenerator

def plot_sensor_data(data, sensor_type, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    sensor_data = data[data['sensor_id'].str.startswith(sensor_type)]
    
    sensor_ids = sensor_data['sensor_id'].unique()
    
    for sensor_id in sensor_ids:
        single_sensor_data = sensor_data[sensor_data['sensor_id'] == sensor_id]
        
        if len(single_sensor_data) < 10:
            continue
        
        plt.figure(figsize=(12, 6))
        
        status_groups = single_sensor_data.groupby('status')
        
        for status, group in status_groups:
            plt.scatter(
                group['timestamp'], 
                group['value'], 
                label=status,
                alpha=0.7,
                s=20
            )
        
        generator = SensorDataGenerator()
        sensor_key = sensor_type
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
        
        plt.gcf().autofmt_xdate()
        
        plt.tight_layout()
        file_path = os.path.join(output_dir, f"{sensor_id}_data.png")
        plt.savefig(file_path)
        plt.close()
        
        print(f"Plot saved to {file_path}")

def main():
    os.makedirs("../../data", exist_ok=True)
    
    plots_dir = "../../data/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    generator = SensorDataGenerator("../../data")
    dataset = generator.generate_dataset(normal_samples=2000, failure_samples=300)
    generator.save_dataset(dataset, filename="sensor_data.csv")
    
    for sensor_type in ["temperature", "humidity", "soil_moisture"]:
        plot_sensor_data(dataset, sensor_type, plots_dir)
    
    print("Data generation and visualization complete.")

if __name__ == "__main__":
    main()