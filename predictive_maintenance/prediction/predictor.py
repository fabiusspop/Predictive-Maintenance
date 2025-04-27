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
        """
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
        df = data.copy()
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        if 'sensor_id' in df.columns:
            df['sensor_type'] = df['sensor_id'].apply(self._extract_sensor_type)
            df['sensor_number'] = df['sensor_id'].apply(self._extract_sensor_number)
        
        # if multiple readings per sensor -> create rolling features
        if 'value' in df.columns and 'timestamp' in df.columns and 'sensor_id' in df.columns:
            df = self._create_rolling_features(df)
        
        # one hot enc
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
        if sensor_id.startswith('soil_moisture'):
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
        df = data.copy()
        
        df = df.sort_values(['sensor_id', 'timestamp'])
        
        # for each sensor_id -> calc rolling stats
        for window in window_sizes:
            grouped = df.groupby('sensor_id')
            
            # rolling mean
            df[f'rolling_mean_{window}'] = grouped['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # rolling std
            df[f'rolling_std_{window}'] = grouped['value'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # rate of change
            df[f'rolling_roc_{window}'] = grouped['value'].transform(
                lambda x: x.rolling(window=window, min_periods=2).apply(
                    lambda y: (y.iloc[-1] - y.iloc[0]) / y.iloc[0] if y.iloc[0] != 0 else 0
                )
            )
        
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
        X = data.copy()
        
        if drop_cols is None:
            drop_cols = ['timestamp', 'status', 'target', 'failure_type']
        
        for col in drop_cols:
            if col in X.columns:
                X = X.drop(col, axis=1)
        
        if hasattr(self.model, 'predict_proba'):
            self.probabilities = self.model.predict_proba(X)[:, 1]
            self.predictions = (self.probabilities >= self.threshold).astype(int)
        else:
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
        if predictions is None:
            predictions = self.predictions
        
        if probabilities is None:
            probabilities = self.probabilities
        
        if predictions is None:
            raise ValueError("No predictions available. Run predict() first.")
        
        results = data.copy()
        
        # predictions
        results['predicted_status'] = np.where(predictions == 1, 'failure', 'normal')
        
        # probabilities if available
        if probabilities is not None:
            results['failure_probability'] = probabilities
        
        if 'sensor_type' not in results.columns and 'sensor_id' in results.columns:
            results['sensor_type'] = results['sensor_id'].apply(self._extract_sensor_type)
        
        return results
    
    def save_prediction_results(self, results, output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
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
            sensor_ids = results['sensor_id'].unique()
            
            for sensor_id in sensor_ids:
                sensor_data = results[results['sensor_id'] == sensor_id]
                
                plt.figure(figsize=(12, 6))
                
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
                
                plt.gcf().autofmt_xdate()
                
                plt.tight_layout()
                filepath = os.path.join(output_dir, f"{sensor_id}_predictions.png")
                plt.savefig(filepath)
                plt.close()
                
                print(f"Prediction plot saved to {filepath}")
        
        # 2. Distribution of failure probabilities
        if 'failure_probability' in results.columns:
            plt.figure(figsize=(10, 6))
            
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
            
            plt.axvline(x=self.threshold, color='red', linestyle='--', alpha=0.7)
            plt.text(
                self.threshold + 0.01, 
                plt.ylim()[1] * 0.9, 
                f'Threshold: {self.threshold}',
                color='red'
            )
            
            plt.tight_layout()
            filepath = os.path.join(output_dir, "failure_probability_distribution.png")
            plt.savefig(filepath)
            plt.close()
            
            print(f"Probability distribution plot saved to {filepath}")
    
    def get_sensors_needing_maintenance(self, results, sort_by='failure_probability', ascending=False):
        """Get a list of sensors requiring maintenance
        
        Args:
            results (pd.DataFrame): Prediction results
            sort_by (str): Column to sort by
            ascending (bool): Sort ascending or descending
            
        Returns:
            pd.DataFrame
        """
        maintenance_needed = results[results['predicted_status'] == 'failure']
        
        # latest reading for each sensor
        if 'timestamp' in maintenance_needed.columns:
            latest_readings = maintenance_needed.sort_values('timestamp').groupby('sensor_id').tail(1)
        else:
            latest_readings = maintenance_needed.groupby('sensor_id').first()
        
        if sort_by in latest_readings.columns:
            sorted_readings = latest_readings.sort_values(sort_by, ascending=ascending)
        else:
            sorted_readings = latest_readings
        
        return sorted_readings
    
    def generate_maintenance_report(self, results, output_path=None):
        """Generate a maintenance report
            
        Returns:
            str: Report in JSON format
        """
        if 'sensor_type' not in results.columns and 'sensor_id' in results.columns:
            results['sensor_type'] = results['sensor_id'].apply(self._extract_sensor_type)
        
        maintenance_sensors = self.get_sensors_needing_maintenance(results)
        
        # counting failures by sensor type
        if 'sensor_type' in results.columns:
            failures_by_type = results[results['predicted_status'] == 'failure'].groupby('sensor_type').size()
            failures_by_type = failures_by_type.to_dict()
        else:
            failures_by_type = {"unknown": len(results[results['predicted_status'] == 'failure'])}
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "total_sensors": len(results['sensor_id'].unique()),
            "sensors_needing_maintenance": len(maintenance_sensors['sensor_id'].unique()),
            "failures_by_sensor_type": failures_by_type,
            "maintenance_list": []
        }
        
        # adding sensors to list
        for _, row in maintenance_sensors.iterrows():
            timestamp_str = None
            if 'timestamp' in row:
                try:
                    if isinstance(row['timestamp'], (pd.Timestamp, datetime)):
                        timestamp_str = row['timestamp'].isoformat()
                    elif isinstance(row['timestamp'], str):
                        timestamp_str = row['timestamp']
                except:
                    timestamp_str = str(row['timestamp'])
            
            sensor_info = {
                "sensor_id": row['sensor_id'],
                "sensor_type": row['sensor_type'] if 'sensor_type' in row else self._extract_sensor_type(row['sensor_id']),
                "latest_value": float(row['value']) if 'value' in row else None,
                "failure_probability": float(row['failure_probability']) if 'failure_probability' in row else None,
                "timestamp": timestamp_str
            }
            report["maintenance_list"].append(sensor_info)
        
        report_json = json.dumps(report, indent=2)
        
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_json)
            
            print(f"Maintenance report saved to {output_path}")
        
        return report_json 