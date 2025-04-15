import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class MaintenancePredictor:
    """Class for making predictive maintenance predictions on sensor data"""
    
    def __init__(self, model_path, threshold=0.5):
        """Initialize the predictor
        
        Args:
            model_path (str): Path to the trained model file
            threshold (float): Probability threshold for positive class
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.threshold = threshold
        self.predictions = None
        self.probabilities = None
    
    def preprocess_data(self, data):
        """Preprocess new sensor data for prediction
        
        Args:
            data (pd.DataFrame): Raw sensor data
            
        Returns:
            pd.DataFrame: Preprocessed data ready for prediction
        """
        # Copy data to avoid modifying the original
        df = data.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Extract sensor type and number
        if 'sensor_id' in df.columns:
            # Handle special case for soil_moisture
            df['sensor_type'] = df['sensor_id'].apply(self._extract_sensor_type)
            df['sensor_number'] = df['sensor_id'].apply(self._extract_sensor_number)
        
        # Create rolling features if there are multiple readings per sensor
        if 'value' in df.columns and 'timestamp' in df.columns and 'sensor_id' in df.columns:
            df = self._create_rolling_features(df)
        
        # Convert categorical features to one-hot encoding
        if 'sensor_type' in df.columns and 'sensor_id' in df.columns and 'unit' in df.columns:
            categorical_cols = ['sensor_type', 'sensor_id', 'unit']
            df = pd.get_dummies(df, columns=categorical_cols)
        
        return df
    
    def _extract_sensor_type(self, sensor_id):
        """Extract sensor type from sensor_id
        
        Args:
            sensor_id (str): Sensor ID string
            
        Returns:
            str: Sensor type
        """
        # Handle special case for soil_moisture
        if sensor_id.startswith('soil_moisture'):
            return 'soil_moisture'
        else:
            return sensor_id.split('_')[0]
    
    def _extract_sensor_number(self, sensor_id):
        """Extract sensor number from sensor_id
        
        Args:
            sensor_id (str): Sensor ID string
            
        Returns:
            int: Sensor number
        """
        # Handle special case for soil_moisture
        if sensor_id.startswith('soil_moisture'):
            # For soil_moisture_X, we want to get X
            return int(sensor_id.split('_')[2])
        else:
            return int(sensor_id.split('_')[1])
    
    def _create_rolling_features(self, data, window_sizes=[5, 10, 20]):
        """Create rolling window features for time series data
        
        Args:
            data (pd.DataFrame): Sensor data
            window_sizes (list): List of window sizes for rolling calculations
            
        Returns:
            pd.DataFrame: DataFrame with additional rolling features
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Sort by timestamp and group by sensor_id
        df = df.sort_values(['sensor_id', 'timestamp'])
        
        # For each group (sensor_id), calculate rolling statistics
        for window in window_sizes:
            grouped = df.groupby('sensor_id')
            
            # Calculate rolling mean
            df[f'rolling_mean_{window}'] = grouped['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Calculate rolling standard deviation
            df[f'rolling_std_{window}'] = grouped['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Calculate rate of change
            df[f'rolling_roc_{window}'] = grouped['value'].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / y.iloc[0] if y.iloc[0] != 0 else 0
                )
            )
        
        # Fill NaN values that may have been created
        df = df.fillna(0)
        
        return df
    
    def predict(self, data, drop_cols=None):
        """Make predictions on preprocessed data
        
        Args:
            data (pd.DataFrame): Preprocessed sensor data
            drop_cols (list): Columns to drop before prediction
            
        Returns:
            np.array: Binary predictions (0=normal, 1=failure)
        """
        # Make a copy of the data
        X = data.copy()
        
        # Drop columns that aren't features
        if drop_cols is None:
            drop_cols = ['timestamp', 'status', 'target', 'failure_type']
        
        for col in drop_cols:
            if col in X.columns:
                X = X.drop(col, axis=1)
        
        # Check if the model has predict_proba method
        if hasattr(self.model, 'predict_proba'):
            # Get probabilities
            self.probabilities = self.model.predict_proba(X)[:, 1]
            # Apply threshold
            self.predictions = (self.probabilities >= self.threshold).astype(int)
        else:
            # Direct predictions
            self.predictions = self.model.predict(X)
            self.probabilities = None
        
        return self.predictions
    
    def get_prediction_results(self, data, predictions=None, probabilities=None):
        """Combine original data with predictions
        
        Args:
            data (pd.DataFrame): Original data
            predictions (np.array): Binary predictions (optional)
            probabilities (np.array): Prediction probabilities (optional)
            
        Returns:
            pd.DataFrame: Original data with predictions
        """
        # Use the stored predictions if none provided
        if predictions is None:
            predictions = self.predictions
        
        if probabilities is None:
            probabilities = self.probabilities
        
        if predictions is None:
            raise ValueError("No predictions available. Run predict() first.")
        
        # Copy the data
        results = data.copy()
        
        # Add predictions
        results['predicted_status'] = np.where(predictions == 1, 'failure', 'normal')
        
        # Add probabilities if available
        if probabilities is not None:
            results['failure_probability'] = probabilities
        
        # Make sure sensor_type is available for reporting
        if 'sensor_type' not in results.columns and 'sensor_id' in results.columns:
            results['sensor_type'] = results['sensor_id'].apply(self._extract_sensor_type)
        
        return results
    
    def save_prediction_results(self, results, output_path):
        """Save prediction results to CSV
        
        Args:
            results (pd.DataFrame): Prediction results
            output_path (str): Path to save the CSV file
        """
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        results.to_csv(output_path, index=False)
        print(f"Prediction results saved to {output_path}")
    
    def visualize_predictions(self, results, output_dir="plots"):
        """Create visualizations of prediction results
        
        Args:
            results (pd.DataFrame): Prediction results
            output_dir (str): Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Time series plot of values with predicted status
        if 'timestamp' in results.columns and 'value' in results.columns:
            # Group by sensor_id
            sensor_ids = results['sensor_id'].unique()
            
            for sensor_id in sensor_ids:
                # Filter data for this sensor
                sensor_data = results[results['sensor_id'] == sensor_id]
                
                # Create plot
                plt.figure(figsize=(12, 6))
                
                # Plot normal and failure points with different colors
                normal_data = sensor_data[sensor_data['predicted_status'] == 'normal']
                failure_data = sensor_data[sensor_data['predicted_status'] == 'failure']
                
                plt.scatter(
                    normal_data['timestamp'], 
                    normal_data['value'],
                    color='green',
                    label='Normal',
                    alpha=0.7
                )
                
                plt.scatter(
                    failure_data['timestamp'], 
                    failure_data['value'],
                    color='red',
                    label='Failure',
                    alpha=0.7
                )
                
                # Add failure probability if available
                if 'failure_probability' in sensor_data.columns:
                    ax2 = plt.gca().twinx()
                    ax2.plot(
                        sensor_data['timestamp'],
                        sensor_data['failure_probability'],
                        color='blue',
                        linestyle='--',
                        alpha=0.5,
                        label='Failure Probability'
                    )
                    ax2.set_ylabel('Failure Probability')
                    ax2.set_ylim(0, 1.05)
                
                plt.title(f"Prediction Results for {sensor_id}")
                plt.xlabel("Timestamp")
                plt.ylabel("Value")
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Format x-axis to show dates nicely
                plt.gcf().autofmt_xdate()
                
                # Save the plot
                plt.tight_layout()
                filepath = os.path.join(output_dir, f"{sensor_id}_predictions.png")
                plt.savefig(filepath)
                plt.close()
                
                print(f"Prediction plot saved to {filepath}")
        
        # 2. Distribution of failure probabilities
        if 'failure_probability' in results.columns:
            plt.figure(figsize=(10, 6))
            
            # Create histogram
            sns.histplot(
                data=results,
                x='failure_probability',
                hue='predicted_status',
                bins=20,
                alpha=0.7,
                kde=True
            )
            
            plt.title("Distribution of Failure Probabilities")
            plt.xlabel("Failure Probability")
            plt.ylabel("Count")
            plt.grid(True, alpha=0.3)
            
            # Add threshold line
            plt.axvline(x=self.threshold, color='red', linestyle='--', alpha=0.7)
            plt.text(
                self.threshold + 0.01, 
                plt.ylim()[1] * 0.9, 
                f'Threshold: {self.threshold}',
                color='red'
            )
            
            # Save the plot
            plt.tight_layout()
            filepath = os.path.join(output_dir, "failure_probability_distribution.png")
            plt.savefig(filepath)
            plt.close()
            
            print(f"Probability distribution plot saved to {filepath}")
    
    def get_sensors_needing_maintenance(self, results, sort_by='failure_probability', ascending=False):
        """Get a list of sensors that need maintenance
        
        Args:
            results (pd.DataFrame): Prediction results
            sort_by (str): Column to sort by
            ascending (bool): Sort ascending or descending
            
        Returns:
            pd.DataFrame: Sensors needing maintenance
        """
        # Filter for failures
        maintenance_needed = results[results['predicted_status'] == 'failure']
        
        # Get the latest reading for each sensor
        if 'timestamp' in maintenance_needed.columns:
            latest_readings = maintenance_needed.sort_values('timestamp').groupby('sensor_id').tail(1)
        else:
            latest_readings = maintenance_needed.groupby('sensor_id').first()
        
        # Sort by the specified column
        if sort_by in latest_readings.columns:
            sorted_readings = latest_readings.sort_values(sort_by, ascending=ascending)
        else:
            sorted_readings = latest_readings
        
        return sorted_readings
    
    def generate_maintenance_report(self, results, output_path=None):
        """Generate a maintenance report
        
        Args:
            results (pd.DataFrame): Prediction results
            output_path (str): Path to save the report (optional)
            
        Returns:
            str: Report in JSON format
        """
        # Make sure sensor_type is available
        if 'sensor_type' not in results.columns and 'sensor_id' in results.columns:
            results['sensor_type'] = results['sensor_id'].apply(self._extract_sensor_type)
        
        # Get sensors needing maintenance
        maintenance_sensors = self.get_sensors_needing_maintenance(results)
        
        # Count failures by sensor type
        if 'sensor_type' in results.columns:
            failures_by_type = results[results['predicted_status'] == 'failure'].groupby('sensor_type').size()
            failures_by_type = failures_by_type.to_dict()
        else:
            failures_by_type = {"unknown": len(results[results['predicted_status'] == 'failure'])}
        
        # Create report
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_sensors": len(results['sensor_id'].unique()),
            "sensors_needing_maintenance": len(maintenance_sensors['sensor_id'].unique()),
            "failures_by_sensor_type": failures_by_type,
            "maintenance_list": []
        }
        
        # Add each sensor to the maintenance list
        for _, row in maintenance_sensors.iterrows():
            # Handle timestamp safely
            timestamp_str = None
            if 'timestamp' in row:
                # Try to convert to ISO format if it's a datetime
                try:
                    if isinstance(row['timestamp'], (pd.Timestamp, datetime)):
                        timestamp_str = row['timestamp'].isoformat()
                    elif isinstance(row['timestamp'], str):
                        timestamp_str = row['timestamp']
                except:
                    # If conversion fails, just use the string representation
                    timestamp_str = str(row['timestamp'])
            
            sensor_info = {
                "sensor_id": row['sensor_id'],
                "sensor_type": row['sensor_type'] if 'sensor_type' in row else self._extract_sensor_type(row['sensor_id']),
                "latest_value": float(row['value']) if 'value' in row else None,
                "failure_probability": float(row['failure_probability']) if 'failure_probability' in row else None,
                "timestamp": timestamp_str
            }
            report["maintenance_list"].append(sensor_info)
        
        # Convert to JSON
        report_json = json.dumps(report, indent=2)
        
        # Save if output path is provided
        if output_path:
            # Ensure directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_json)
            
            print(f"Maintenance report saved to {output_path}")
        
        return report_json 