import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

class SensorDataGenerator:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # sensor types and their normal operating ranges
        self.sensor_types = {
            "temperature": {"min": 20, "max": 30, "unit": "°C"},
            "humidity": {"min": 40, "max": 70, "unit": "%"},
            "soil_moisture": {"min": 30, "max": 60, "unit": "%"}
        }
    
    def generate_normal_data(self, sensor_type, num_samples=1000, days=30):
        """Generate normal operating data for a specific sensor type.
        
        Args:
            sensor_type (str): Type of sensor
            num_samples (int): Number of data points to generate
            days (int): Number of days to spread the data over
            
        Returns:
            Generated sensor data
        """
        if sensor_type not in self.sensor_types:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        sensor_range = self.sensor_types[sensor_type]
        min_val, max_val = sensor_range["min"], sensor_range["max"]
        
        # timestamps spanning the specified number of days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = [start_time + timedelta(
            seconds=np.random.randint(0, days * 24 * 60 * 60)
        ) for _ in range(num_samples)]
        timestamps.sort()
        
        # normal values with some random noise
        mean = (min_val + max_val) / 2
        std_dev = (max_val - min_val) / 6  
        values = np.random.normal(mean, std_dev, num_samples)
        
        # Generate sensor IDs
        sensor_ids = [f"{sensor_type}_{np.random.randint(1, 6)}" for _ in range(num_samples)]
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "sensor_id": sensor_ids,
            "value": values,
            "unit": sensor_range["unit"],
            "status": "normal"
        })
        
        return df
    
    def generate_failure_data(self, sensor_type, num_samples=100, failure_type="drift"):
        """Generate data indicating sensor failure.
        
        Args:
            sensor_type (str): Type of sensor
            num_samples (int): Number of data points to generate
            failure_type (str): Type of failure pattern
            
        Returns:
            Generated failure data
        """
        if sensor_type not in self.sensor_types:
            raise ValueError(f"Unknown sensor type: {sensor_type}")
        
        sensor_range = self.sensor_types[sensor_type]
        min_val, max_val = sensor_range["min"], sensor_range["max"]
        
        # Generate timestamps for the last 3 days
        end_time = datetime.now()
        start_time = end_time - timedelta(days=3)
        timestamps = [start_time + timedelta(
            seconds=np.random.randint(0, 3 * 24 * 60 * 60)
        ) for _ in range(num_samples)]
        timestamps.sort()
        
        # Generate sensor ID (use just one sensor ID for the failure data)
        sensor_id = f"{sensor_type}_{np.random.randint(1, 6)}"
        sensor_ids = [sensor_id] * num_samples
        
        # Generate values based on failure type
        if failure_type == "drift":
            # Gradual drift beyond normal range
            base = np.linspace(max_val, max_val * 1.5, num_samples)
            noise = np.random.normal(0, (max_val - min_val) / 20, num_samples)
            values = base + noise
        elif failure_type == "spike":
            # Random spikes well beyond normal range
            mean = (min_val + max_val) / 2
            std_dev = (max_val - min_val) / 6
            values = np.random.normal(mean, std_dev, num_samples)
            # Add spikes to ~30% of values
            spike_indices = np.random.choice(num_samples, size=int(num_samples * 0.3), replace=False)
            spike_direction = np.random.choice([-1, 1], size=len(spike_indices))
            for i, direction in zip(spike_indices, spike_direction):
                if direction > 0:
                    values[i] = max_val * (1.5 + np.random.random() * 0.5)
                else:
                    values[i] = min_val * (0.5 - np.random.random() * 0.3)
        else: 
            # Excessive noise
            mean = (min_val + max_val) / 2
            std_dev = (max_val - min_val) / 2  # much higher
            values = np.random.normal(mean, std_dev, num_samples)
        
        df = pd.DataFrame({
            "timestamp": timestamps,
            "sensor_id": sensor_ids,
            "value": values,
            "unit": sensor_range["unit"],
            "status": f"failure_{failure_type}"
        })
        
        return df
    
    def generate_dataset(self, normal_samples=1000, failure_samples=200, include_failures=True):
        """Generate a complete dataset for all sensor types.
        
        Args:
            normal_samples (int): Number of normal samples per sensor type
            failure_samples (int): Number of failure samples per sensor type
            include_failures (bool): Whether to include failure data
            
        Returns:
            combined dataset
        """
        all_data = []
        
        # normal data for each sensor type
        for sensor_type in self.sensor_types:
            normal_data = self.generate_normal_data(sensor_type, normal_samples)
            all_data.append(normal_data)
            
            # failure data if requested
            if include_failures:
                failure_types = ["drift", "spike", "noise"]
                for failure_type in failure_types:
                    failure_data = self.generate_failure_data(
                        sensor_type, 
                        num_samples=failure_samples // len(failure_types),
                        failure_type=failure_type
                    )
                    all_data.append(failure_data)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.sort_values("timestamp", inplace=True)
        
        return combined_df
    
    def save_dataset(self, dataset, filename="sensor_data.csv"):
        """Save the generated dataset to a CSV file.
        """
        filepath = os.path.join(self.output_dir, filename)
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")


if __name__ == "__main__":
    generator = SensorDataGenerator("../../data")
    dataset = generator.generate_dataset(normal_samples=5000, failure_samples=500)
    generator.save_dataset(dataset)
    
    print(f"Dataset shape: {dataset.shape}")
    print(f"Sensor types: {dataset['sensor_id'].nunique()}")
    print("Status distribution:")
    print(dataset['status'].value_counts()) 