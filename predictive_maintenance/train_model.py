import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent))

from data_preprocessing.preprocessor import SensorDataPreprocessor
from training.model_trainer import RandomForestTrainer

def plot_confusion_matrix(cm, classes, output_dir="plots", filename="confusion_matrix.png"):
    """Plot and save confusion matrix
    
    Args:
        cm (array): Confusion matrix array
        classes (list): Class labels
        output_dir (str): Directory to save the plot
        filename (str): Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Confusion matrix saved to {filepath}")

def plot_feature_importance(importance_df, output_dir="plots", top_n=10, filename="feature_importance.png"):
    """Plot and save feature importance
    
    Args:
        importance_df (pd.DataFrame): DataFrame with feature importance
        output_dir (str): Directory to save the plot
        top_n (int): Number of top features to display
        filename (str): Name of the output file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Feature importance plot saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Train predictive maintenance model')
    
    parser.add_argument('--input', type=str, default='data/preprocessed_data.csv',
                        help='Path to the preprocessed data CSV file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save the trained model')
    parser.add_argument('--plots-dir', type=str, default='data/plots/training',
                        help='Directory to save training plots')
    parser.add_argument('--tune', action='store_true',
                        help='Perform hyperparameter tuning')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    
    # Load preprocessed data
    print(f"Loading preprocessed data from {args.input}...")
    preprocessor = SensorDataPreprocessor()
    data = preprocessor.load_data(args.input)
    
    # Split data before preprocessing to prevent data leakage
    print("Splitting data into train and test sets...")
    train_data, test_data = train_test_split(
        data, test_size=args.test_size, random_state=42, 
        stratify=data['target']
    )
    
    # Store feature names for later use
    drop_cols = ['timestamp', 'status', 'target', 'failure_type']
    feature_names = [col for col in train_data.columns if col not in drop_cols]
    
    # Prepare data for training
    print("Preparing data for training...")
    X_train = train_data.drop(drop_cols, axis=1)
    y_train = train_data['target']
    
    # Prepare test data
    X_test = test_data.drop(drop_cols, axis=1)
    y_test = test_data['target']
    
    # Manually preprocess data for scikit-learn compatibility
    # Convert categorical features to one-hot encoding
    categorical_features = ['sensor_type', 'sensor_id', 'unit']
    numerical_features = [f for f in feature_names if f not in categorical_features]
    
    # One-hot encode categorical features
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_features)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_features)
    
    # Ensure train and test have the same columns
    for col in X_train_encoded.columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0
    
    # Reorder columns to match
    X_test_encoded = X_test_encoded[X_train_encoded.columns]
    
    # Create and train model
    print("Training Random Forest model...")
    trainer = RandomForestTrainer(model_dir=args.model_dir)
    
    if args.tune:
        print("Performing hyperparameter tuning (this may take a while)...")
        results = trainer.train_with_hyperparameter_tuning(X_train_encoded, y_train, test_size=0)
        print(f"Best parameters: {results['best_params']}")
    else:
        # Train on the full training set, evaluate on the test set
        # We use test_size=0 here because we already split the data
        trainer._create_model()
        trainer.model.fit(X_train_encoded, y_train)
        y_pred = trainer.model.predict(X_test_encoded)
        
        # Create results dictionary
        results = {
            "accuracy": np.mean(y_pred == y_test),
            "classification_report": None,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "X_test": X_test_encoded,
            "y_test": y_test,
            "y_pred": y_pred
        }
    
    # Print training results
    print(f"\nModel accuracy: {results['accuracy']:.4f}")
    
    # Save the model
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}")
    
    # Get and plot feature importance
    importance_df = trainer.get_feature_importance(X_train_encoded.columns)
    plot_feature_importance(importance_df, args.plots_dir)
    
    # Save top features to CSV
    importance_df.to_csv(os.path.join(args.plots_dir, 'feature_importance.csv'), index=False)
    print(f"Feature importance saved to {os.path.join(args.plots_dir, 'feature_importance.csv')}")
    
    # Plot confusion matrix
    class_names = ["Normal", "Failure"]
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        args.plots_dir
    )
    
    # Save the test data for later evaluation
    test_data_path = os.path.join(args.model_dir, 'test_data.csv')
    test_data.to_csv(test_data_path, index=False)
    print(f"Test data saved to {test_data_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main() 