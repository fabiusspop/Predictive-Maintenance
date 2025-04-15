import numpy as np
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class ModelTrainer:
    """Base class for training predictive maintenance models"""
    
    def __init__(self, model_dir="models"):
        """Initialize the model trainer
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.model = None
        self.model_name = "base_model"
        os.makedirs(model_dir, exist_ok=True)
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train a model on the provided data
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing training results
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Create and train the model
        self._create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Compile results
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        return results
    
    def _create_model(self):
        """Create the model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_model")
    
    def save_model(self, filename=None):
        """Save the trained model to disk
        
        Args:
            filename (str): Name to save the model as
            
        Returns:
            str: Path to the saved model
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        if filename is None:
            filename = f"{self.model_name}.joblib"
        
        model_path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, model_path)
        
        return model_path
    
    def load_model(self, filename=None):
        """Load a trained model from disk
        
        Args:
            filename (str): Name of the model file
            
        Returns:
            object: Loaded model
        """
        if filename is None:
            filename = f"{self.model_name}.joblib"
        
        model_path = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        
        return self.model
    
    def predict(self, X):
        """Make predictions using the trained model
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Model predictions
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities if the model supports it
        
        Args:
            X: Feature matrix
            
        Returns:
            array: Prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model has been trained or loaded yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("This model does not support probability predictions")


class RandomForestTrainer(ModelTrainer):
    """Class for training Random Forest models for predictive maintenance"""
    
    def __init__(self, model_dir="models", n_estimators=100, max_depth=None):
        """Initialize the Random Forest trainer
        
        Args:
            model_dir (str): Directory to save trained models
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
        """
        super().__init__(model_dir)
        self.model_name = "random_forest"
        self.n_estimators = n_estimators
        self.max_depth = max_depth
    
    def _create_model(self):
        """Create a Random Forest Classifier"""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
    
    def train_with_hyperparameter_tuning(self, X, y, test_size=0.2, random_state=42, cv=5):
        """Train with hyperparameter tuning using GridSearchCV
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Dictionary containing training results
        """
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Define base model
        base_model = RandomForestClassifier(random_state=42)
        
        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            verbose=1,
            scoring='accuracy'
        )
        
        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)
        
        # Get best model
        self.model = grid_search.best_estimator_
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        
        # Compile results
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "best_params": grid_search.best_params_,
            "cv_results": grid_search.cv_results_,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        return results
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance from the trained Random Forest
        
        Args:
            feature_names (list): List of feature names
            
        Returns:
            pd.DataFrame: DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("No model has been trained yet")
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # If feature names are not provided, use generic names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create DataFrame with feature importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    # Create and train a Random Forest model
    trainer = RandomForestTrainer(model_dir="../models")
    results = trainer.train(X, y)
    
    # Print training results
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Save the model
    model_path = trainer.save_model()
    print(f"Model saved to: {model_path}")
    
    # Get feature importance
    feature_importance = trainer.get_feature_importance()
    print("\nFeature Importance (top 5):")
    print(feature_importance.head(5)) 