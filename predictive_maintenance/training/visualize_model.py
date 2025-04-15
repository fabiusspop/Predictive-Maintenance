import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, classification_report
import os
import joblib
import argparse

def load_model(model_path):
    """Load a trained model
    
    Args:
        model_path (str): Path to the model file
        
    Returns:
        object: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return joblib.load(model_path)

def plot_roc_curve(y_true, y_score, output_dir="plots", filename="roc_curve.png"):
    """Plot ROC curve
    
    Args:
        y_true (array): True labels
        y_score (array): Predicted probabilities
        output_dir (str): Directory to save the plot
        filename (str): Filename for the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"ROC curve saved to {filepath}")
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_score, output_dir="plots", filename="precision_recall_curve.png"):
    """Plot precision-recall curve
    
    Args:
        y_true (array): True labels
        y_score (array): Predicted probabilities
        output_dir (str): Directory to save the plot
        filename (str): Filename for the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    plt.close()
    
    print(f"Precision-recall curve saved to {filepath}")
    
    return pr_auc

def plot_error_analysis(y_true, y_pred, X_test, feature_names, output_dir="plots"):
    """Plot error analysis visualizations
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        X_test (array): Test features
        feature_names (list): Feature names
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert X_test to DataFrame for easier handling
    if not isinstance(X_test, pd.DataFrame):
        X_test = pd.DataFrame(X_test, columns=feature_names)
    
    # Create DataFrame with predictions and true values
    results_df = X_test.copy()
    results_df['true'] = y_true
    results_df['predicted'] = y_pred
    results_df['error'] = y_true != y_pred
    
    # Get misclassified samples
    misclassified = results_df[results_df['error']]
    
    # If no misclassifications, return
    if len(misclassified) == 0:
        print("No misclassifications to analyze")
        return
    
    # Get numeric features
    numeric_features = results_df.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = [f for f in numeric_features if f not in ['true', 'predicted', 'error']]
    
    # Plot distribution of top 5 features for correct vs. incorrect predictions
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(numeric_features[:5], 1):
        plt.subplot(2, 3, i)
        sns.kdeplot(
            data=results_df, x=feature, hue='error',
            palette={True: 'red', False: 'green'},
            common_norm=False, alpha=0.7
        )
        plt.title(f"{feature} Distribution")
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, "error_analysis_features.png")
    plt.savefig(filepath)
    plt.close()
    
    print(f"Error analysis plot saved to {filepath}")

def evaluate_model(model_path, test_data_path, output_dir="plots"):
    """Evaluate a trained model and generate visualizations
    
    Args:
        model_path (str): Path to the trained model
        test_data_path (str): Path to test data CSV
        output_dir (str): Directory to save plots
    """
    # Load model
    model = load_model(model_path)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    
    # Prepare test data
    drop_cols = ['timestamp', 'status', 'target', 'failure_type']
    X_test = test_data.drop(drop_cols, axis=1, errors='ignore')
    y_test = test_data['target']
    
    # Process categorical features
    categorical_features = ['sensor_type', 'sensor_id', 'unit']
    if all(col in X_test.columns for col in categorical_features):
        X_test = pd.get_dummies(X_test, columns=categorical_features)
    
    # Make sure X_test has the right format for the model
    # The model might expect specific columns after one-hot encoding
    try:
        y_pred = model.predict(X_test)
    except (ValueError, KeyError) as e:
        print(f"Error making predictions: {e}")
        print("This might be due to mismatched columns after one-hot encoding.")
        print("Using only numerical features as a fallback.")
        
        # Use only numerical features as a fallback
        numerical_features = X_test.select_dtypes(include=['int64', 'float64']).columns
        X_test = X_test[numerical_features]
        y_pred = model.predict(X_test)
    
    # Calculate and print classification report
    report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(report)
    
    # Get feature names
    feature_names = X_test.columns.tolist()
    
    # Get prediction probabilities if available
    try:
        y_score = model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
        
        # Plot ROC curve
        roc_auc = plot_roc_curve(y_test, y_score, output_dir)
        
        # Plot precision-recall curve
        pr_auc = plot_precision_recall_curve(y_test, y_score, output_dir)
        
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
    except (AttributeError, IndexError) as e:
        print(f"Model does not support probability predictions: {e}")
        print("Skipping ROC and PR curves")
    
    # Plot error analysis
    plot_error_analysis(y_test, y_pred, X_test, feature_names, output_dir)
    
    # Return accuracy
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description='Visualize model evaluation results')
    
    parser.add_argument('--model', type=str, default='models/random_forest.joblib',
                        help='Path to the trained model file')
    parser.add_argument('--test-data', type=str, default='data/preprocessed_data.csv',
                        help='Path to the test data CSV file')
    parser.add_argument('--output', type=str, default='data/plots/evaluation',
                        help='Directory to save evaluation plots')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # Evaluate model and generate visualizations
    print(f"Evaluating model {args.model} on {args.test_data}...")
    evaluate_model(args.model, args.test_data, args.output)
    
    print("Evaluation complete. Plots saved to", args.output)

if __name__ == "__main__":
    main() 