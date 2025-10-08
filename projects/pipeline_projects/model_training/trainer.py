"""
ML Pipeline - Model Training Module
模型训练模块，支持多种机器学习算法的训练、验证和超参数优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Comprehensive model training and evaluation pipeline
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize model trainer with configuration
        
        Args:
            config: Configuration dictionary with training parameters
        """
        self.config = config or self._default_config()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.training_history = []
        
    def _default_config(self) -> dict:
        """Default configuration for model training"""
        return {
            'task_type': 'classification',  # 'classification' or 'regression'
            'cv_folds': 5,
            'random_state': 42,
            'test_size': 0.2,
            'scoring': 'accuracy',  # For classification: 'accuracy', 'f1', 'roc_auc'
                                   # For regression: 'neg_mean_squared_error', 'r2'
            'hyperparameter_tuning': True,
            'tuning_method': 'grid',  # 'grid' or 'random'
            'n_iter': 100,  # For random search
            'save_models': True,
            'model_dir': 'models/',
            'verbose': True
        }
    
    def get_available_models(self) -> dict:
        """
        Get available models based on task type
        
        Returns:
            Dictionary of available models
        """
        if self.config['task_type'] == 'classification':
            return {
                'logistic_regression': LogisticRegression(random_state=self.config['random_state']),
                'random_forest': RandomForestClassifier(random_state=self.config['random_state']),
                'svm': SVC(random_state=self.config['random_state']),
                'decision_tree': DecisionTreeClassifier(random_state=self.config['random_state']),
                'neural_network': MLPClassifier(random_state=self.config['random_state']),
                'xgboost': xgb.XGBClassifier(random_state=self.config['random_state']),
                'lightgbm': lgb.LGBMClassifier(random_state=self.config['random_state'])
            }
        else:  # regression
            return {
                'linear_regression': LinearRegression(),
                'random_forest': RandomForestRegressor(random_state=self.config['random_state']),
                'svr': SVR(),
                'decision_tree': DecisionTreeRegressor(random_state=self.config['random_state']),
                'neural_network': MLPRegressor(random_state=self.config['random_state']),
                'xgboost': xgb.XGBRegressor(random_state=self.config['random_state']),
                'lightgbm': lgb.LGBMRegressor(random_state=self.config['random_state'])
            }
    
    def get_hyperparameter_grids(self) -> dict:
        """
        Get hyperparameter grids for tuning
        
        Returns:
            Dictionary of hyperparameter grids
        """
        if self.config['task_type'] == 'classification':
            return {
                'logistic_regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'svm': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            }
        else:  # regression
            return {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'svr': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                },
                'xgboost': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'lightgbm': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100]
                }
            }
    
    def train_single_model(self, model_name: str, model, X_train: np.ndarray, 
                          y_train: np.ndarray, X_val: np.ndarray = None, 
                          y_val: np.ndarray = None) -> dict:
        """
        Train a single model
        
        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        if self.config['verbose']:
            print(f"🚀 Training {model_name}...")
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, 
                                  cv=self.config['cv_folds'], 
                                  scoring=self.config['scoring'])
        
        # Validation score (if validation set provided)
        val_score = None
        if X_val is not None and y_val is not None:
            if self.config['task_type'] == 'classification':
                val_score = model.score(X_val, y_val)
            else:
                y_pred = model.predict(X_val)
                val_score = r2_score(y_val, y_pred)
        
        result = {
            'model_name': model_name,
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'val_score': val_score,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update best model
        if val_score is not None and val_score > self.best_score:
            self.best_score = val_score
            self.best_model = model
        elif val_score is None and cv_scores.mean() > self.best_score:
            self.best_score = cv_scores.mean()
            self.best_model = model
        
        return result
    
    def hyperparameter_tuning(self, model_name: str, model, param_grid: dict,
                            X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Perform hyperparameter tuning
        
        Args:
            model_name: Name of the model
            model: Model instance
            param_grid: Parameter grid for tuning
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Dictionary with tuning results
        """
        if self.config['verbose']:
            print(f"🔧 Tuning hyperparameters for {model_name}...")
        
        if self.config['tuning_method'] == 'grid':
            search = GridSearchCV(
                model, param_grid, 
                cv=self.config['cv_folds'],
                scoring=self.config['scoring'],
                n_jobs=-1,
                verbose=0
            )
        else:  # random search
            search = RandomizedSearchCV(
                model, param_grid,
                n_iter=self.config['n_iter'],
                cv=self.config['cv_folds'],
                scoring=self.config['scoring'],
                n_jobs=-1,
                random_state=self.config['random_state'],
                verbose=0
            )
        
        search.fit(X_train, y_train)
        
        result = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': search.cv_results_
        }
        
        return result
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train all available models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with all training results
        """
        if self.config['verbose']:
            print("🎯 Starting comprehensive model training...")
        
        available_models = self.get_available_models()
        param_grids = self.get_hyperparameter_grids()
        
        all_results = {}
        
        for model_name, model in available_models.items():
            try:
                # Hyperparameter tuning if enabled
                if self.config['hyperparameter_tuning'] and model_name in param_grids:
                    tuning_result = self.hyperparameter_tuning(
                        model_name, model, param_grids[model_name], X_train, y_train
                    )
                    best_model = tuning_result['best_model']
                    tuning_info = tuning_result
                else:
                    best_model = model
                    tuning_info = None
                
                # Train the best model
                result = self.train_single_model(
                    model_name, best_model, X_train, y_train, X_val, y_val
                )
                
                result['tuning_info'] = tuning_info
                all_results[model_name] = result
                
                # Store model
                self.models[model_name] = best_model
                
            except Exception as e:
                if self.config['verbose']:
                    print(f"❌ Error training {model_name}: {str(e)}")
                continue
        
        self.results = all_results
        return all_results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        if self.config['task_type'] == 'classification':
            # Classification metrics
            accuracy = model.score(X_test, y_test)
            
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except:
                auc_score = None
            
            metrics = {
                'accuracy': accuracy,
                'auc_score': auc_score,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': rmse
            }
        
        return metrics
    
    def plot_model_comparison(self, save_path: str = None):
        """
        Plot comparison of all trained models
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.results:
            print("❌ No results to plot. Train models first.")
            return
        
        # Extract scores for plotting
        model_names = list(self.results.keys())
        cv_scores = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        val_scores = [self.results[name]['val_score'] for name in model_names if self.results[name]['val_score'] is not None]
        val_names = [name for name in model_names if self.results[name]['val_score'] is not None]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. CV Scores comparison
        axes[0, 0].bar(model_names, cv_scores, yerr=cv_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Cross-Validation Scores')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Validation Scores comparison (if available)
        if val_scores:
            axes[0, 1].bar(val_names, val_scores, alpha=0.7, color='orange')
            axes[0, 1].set_title('Validation Scores')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No validation scores available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Validation Scores')
        
        # 3. Training time comparison
        training_times = [self.results[name]['training_time'] for name in model_names]
        axes[1, 0].bar(model_names, training_times, alpha=0.7, color='green')
        axes[1, 0].set_title('Training Time')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Score vs Time scatter plot
        axes[1, 1].scatter(training_times, cv_scores, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[1, 1].annotate(name, (training_times[i], cv_scores[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('CV Score')
        axes[1, 1].set_title('Score vs Training Time')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_models(self, model_dir: str = None):
        """
        Save trained models
        
        Args:
            model_dir: Directory to save models
        """
        if not model_dir:
            model_dir = self.config['model_dir']
        
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)
            
            if self.config['verbose']:
                print(f"💾 Saved {model_name} to {model_path}")
        
        # Save results
        results_path = os.path.join(model_dir, "training_results.json")
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {}
            for name, result in self.results.items():
                json_result = result.copy()
                if 'cv_scores' in json_result:
                    json_result['cv_scores'] = json_result['cv_scores'].tolist()
                json_results[name] = json_result
            
            json.dump(json_results, f, indent=2, default=str)
        
        if self.config['verbose']:
            print(f"💾 Saved training results to {results_path}")
    
    def load_models(self, model_dir: str):
        """
        Load trained models
        
        Args:
            model_dir: Directory containing saved models
        """
        import os
        import glob
        
        model_files = glob.glob(os.path.join(model_dir, "*.joblib"))
        
        for model_file in model_files:
            model_name = os.path.basename(model_file).replace('.joblib', '')
            model = joblib.load(model_file)
            self.models[model_name] = model
            
            if self.config['verbose']:
                print(f"📂 Loaded {model_name} from {model_file}")
    
    def get_model_summary(self) -> pd.DataFrame:
        """
        Get summary of all trained models
        
        Returns:
            DataFrame with model summary
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std'],
                'Val_Score': result['val_score'],
                'Training_Time': result['training_time']
            })
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('CV_Mean', ascending=False)
        
        return df

# Example usage
def example_usage():
    """Example usage of ModelTrainer"""
    from sklearn.datasets import make_classification, make_regression
    
    # Create sample data
    if True:  # Set to True for classification, False for regression
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        task_type = 'classification'
    else:
        X, y = make_regression(n_samples=1000, n_features=20, random_state=42)
        task_type = 'regression'
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Initialize trainer
    config = {
        'task_type': task_type,
        'cv_folds': 5,
        'random_state': 42,
        'scoring': 'accuracy' if task_type == 'classification' else 'r2',
        'hyperparameter_tuning': True,
        'tuning_method': 'random',
        'n_iter': 20,
        'verbose': True
    }
    
    trainer = ModelTrainer(config)
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Get model summary
    summary = trainer.get_model_summary()
    print("\n📊 Model Summary:")
    print(summary)
    
    # Plot comparison
    trainer.plot_model_comparison()
    
    # Evaluate best model on test set
    if trainer.best_model:
        test_metrics = trainer.evaluate_model(trainer.best_model, X_test, y_test)
        print(f"\n🎯 Best Model Test Performance:")
        for metric, value in test_metrics.items():
            if metric not in ['classification_report', 'confusion_matrix']:
                print(f"{metric}: {value:.4f}")
    
    # Save models
    trainer.save_models('example_models/')

if __name__ == "__main__":
    example_usage()
