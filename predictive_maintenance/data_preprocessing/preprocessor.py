import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

class SensorDataPreprocessor:
    def __init__(self):
        self.scaler = None
        self.categorical_encoder = None
        self.feature_pipeline = None
    
    def load_data(self, data_path):
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        data = pd.read_csv(data_path)
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return data
    
    def extract_features(self, data):
        """Extract features from raw sensor data
        
        Args:
            data
            
        Returns:
            df with extracted features
        """
        df = data.copy()
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        df['sensor_type'] = df['sensor_id'].apply(self._extract_sensor_type)
        
        df['sensor_number'] = df['sensor_id'].apply(self._extract_sensor_number)
        
        # (1 failure, 0 normal)
        df['target'] = df['status'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # failure type -> multi-class classification)
        df['failure_type'] = df['status'].apply(lambda x: x if x == 'normal' else x.split('_')[1])
        
        return df
    
    def _extract_sensor_type(self, sensor_id):
        """Extract sensor type from sensor_id
        
        Args:
            sensor_id 
            
        Returns:
            Sensor type
        """
        if sensor_id.startswith('soil_moisture'):
            return 'soil_moisture'
        else:
            return sensor_id.split('_')[0]
    
    def _extract_sensor_number(self, sensor_id):
    
        if sensor_id.startswith('soil_moisture'):
            return int(sensor_id.split('_')[2])
        else:
            return int(sensor_id.split('_')[1])
    
    def create_rolling_features(self, data, window_sizes=[5, 10, 20], group_by='sensor_id'):
        """Create rolling window features for time series data
        
        Args:
            data (pd.DataFrame): Preprocessed sensor data
            window_sizes (list): List of window sizes for rolling calculations
            group_by (str): Column to group by before calculating rolling features
            
        Returns:
            df with additional rolling features
        """
        df = data.copy()
        
        # sort by timestamp and group by sensor_id
        df = df.sort_values(['sensor_id', 'timestamp'])
        
        # calculate rolling statistics for each group
        for window in window_sizes:
            grouped = df.groupby(group_by)
            
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
    
    def prepare_for_training(self, data, categorical_features=None, numerical_features=None):
        """Prepare data for model training by scaling numerical features
        and encoding categorical features
        
        Args:
            data (pd.DataFrame): Preprocessed data
            categorical_features (list): List of categorical feature names
            numerical_features (list): List of numerical feature names
            
        Returns:
            tuple: (X, y) prepared for training
        """
        if categorical_features is None:
            categorical_features = ['sensor_type', 'sensor_id']
            
        if numerical_features is None:
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numerical_features = [f for f in numerical_features 
                                if f not in ['target'] and 'timestamp' not in f]
        
        # preprocessing for numerical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        # preprocessing for categorical features
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
    
        self.feature_pipeline = preprocessor
        
        X = data.drop(['timestamp', 'status', 'target', 'failure_type'], axis=1)
        y = data['target']
        
        return X, y
    
    def prepare_multiclass(self, data, categorical_features=None, numerical_features=None):
        """Prepare data for multi-class classification (predicting failure type)
        
        Args:
            data (pd.DataFrame): Preprocessed data
            categorical_features (list): List of categorical feature names
            numerical_features (list): List of numerical feature names
            
        Returns:
            tuple: (X, y) prepared for training
        """
        X, _ = self.prepare_for_training(data, categorical_features, numerical_features)
        y = data['failure_type']
        
        return X, y
    
    def transform_new_data(self, data):
        """Transform new data using the fitted pipeline
        
        Args:
            data (pd.DataFrame): New data to transform
            
        Returns:
            np.ndarray: Transformed features
        """
        if self.feature_pipeline is None:
            raise ValueError("Pipeline not fitted yet. Call prepare_for_training first.")
        
        X = data.drop(['timestamp', 'status', 'target', 'failure_type'], axis=1, errors='ignore')
        
        return self.feature_pipeline.transform(X)
    
    def save_preprocessed_data(self, data, output_path):
        """Save preprocessed data to CSV
        
        Args:
            data (pd.DataFrame): Preprocessed data
            output_path
        """
        data.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")


if __name__ == "__main__":
    preprocessor = SensorDataPreprocessor()

    data_path = "../../data/sensor_data.csv"
    data = preprocessor.load_data(data_path)
    
    features_df = preprocessor.extract_features(data)
    processed_df = preprocessor.create_rolling_features(features_df)
    
    preprocessor.save_preprocessed_data(processed_df, "../../data/preprocessed_data.csv")
    
    X, y = preprocessor.prepare_for_training(processed_df)
    
    print(f"Original data shape: {data.shape}")
    print(f"Preprocessed data shape: {processed_df.shape}")
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    print(f"Feature names: {X.columns.tolist()}")
    print(f"Target distribution:\n{y.value_counts()}") 