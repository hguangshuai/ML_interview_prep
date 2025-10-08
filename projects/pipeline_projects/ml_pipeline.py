"""
Complete ML Pipeline - End-to-End Machine Learning Workflow
从数据准备到模型部署的完整机器学习流水线
"""

import os
import sys
import json
import yaml
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Add pipeline modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_preparation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_training'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'model_deployment'))

from preprocessor import DataPreprocessor
from trainer import ModelTrainer
from deployer import ModelDeployer

class MLPipeline:
    """
    Complete end-to-end machine learning pipeline
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize ML pipeline with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.pipeline_logs = []
        self.results = {}
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config.get('data_preprocessing', {}))
        self.trainer = ModelTrainer(self.config.get('model_training', {}))
        self.deployer = ModelDeployer(self.config.get('model_deployment', {}))
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            return self._default_config()
    
    def _default_config(self) -> dict:
        """Default pipeline configuration"""
        return {
            'pipeline_name': 'ml_pipeline',
            'data_preprocessing': {
                'test_size': 0.2,
                'random_state': 42,
                'scaling_method': 'standard',
                'encoding_method': 'onehot',
                'imputation_method': 'mean',
                'handle_outliers': True,
                'outlier_method': 'iqr'
            },
            'model_training': {
                'task_type': 'classification',
                'cv_folds': 5,
                'random_state': 42,
                'scoring': 'accuracy',
                'hyperparameter_tuning': True,
                'tuning_method': 'random',
                'n_iter': 50,
                'verbose': True
            },
            'model_deployment': {
                'deployment_type': 'api',
                'model_format': 'joblib',
                'api_port': 5000,
                'monitoring_enabled': True,
                'version_control': True
            },
            'output_dir': 'pipeline_output/',
            'log_level': 'INFO'
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = os.path.join(self.config['output_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.get('log_level', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'pipeline.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _log_step(self, step: str, status: str, details: str = ""):
        """Log pipeline step"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            'status': status,
            'details': details
        }
        self.pipeline_logs.append(log_entry)
        
        if status == 'success':
            self.logger.info(f"✅ {step}: {details}")
        elif status == 'error':
            self.logger.error(f"❌ {step}: {details}")
        else:
            self.logger.info(f"🔄 {step}: {details}")
    
    def load_data(self, data_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from file
        
        Args:
            data_path: Path to data file
            **kwargs: Additional arguments for data loading
            
        Returns:
            Loaded DataFrame
        """
        try:
            self._log_step('data_loading', 'start', f"Loading data from {data_path}")
            
            df = self.preprocessor.load_data(data_path, **kwargs)
            
            self._log_step('data_loading', 'success', 
                         f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
            
            return df
            
        except Exception as e:
            self._log_step('data_loading', 'error', str(e))
            raise
    
    def explore_data(self, df: pd.DataFrame, save_plots: bool = True):
        """
        Explore and visualize data
        
        Args:
            df: Input DataFrame
            save_plots: Whether to save plots
        """
        try:
            self._log_step('data_exploration', 'start', "Exploring data")
            
            # Get exploration results
            exploration_results = self.preprocessor.explore_data(df)
            
            # Save exploration results
            output_dir = os.path.join(self.config['output_dir'], 'data_exploration')
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'exploration_results.json'), 'w') as f:
                json.dump(exploration_results, f, indent=2, default=str)
            
            # Create visualizations
            if save_plots:
                plot_path = os.path.join(output_dir, 'data_visualization.png')
                self.preprocessor.visualize_data(df, save_path=plot_path)
            
            self.results['data_exploration'] = exploration_results
            
            self._log_step('data_exploration', 'success', 
                         f"Exploration completed. Quality score: {exploration_results.get('quality_score', 'N/A')}")
            
        except Exception as e:
            self._log_step('data_exploration', 'error', str(e))
            raise
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str = None) -> tuple:
        """
        Preprocess data for training
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            self._log_step('data_preprocessing', 'start', "Preprocessing data")
            
            # Preprocess data
            X_train, X_test, y_train, y_test = self.preprocessor.preprocess_pipeline(
                df, target_column=target_column, fit=True
            )
            
            # Save preprocessed data
            output_dir = os.path.join(self.config['output_dir'], 'preprocessed_data')
            os.makedirs(output_dir, exist_ok=True)
            
            X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
            X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
            
            if y_train is not None:
                y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
                y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
            
            self.results['preprocessing'] = {
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape,
                'y_train_shape': y_train.shape if y_train is not None else None,
                'y_test_shape': y_test.shape if y_test is not None else None,
                'feature_names': self.preprocessor.feature_names
            }
            
            self._log_step('data_preprocessing', 'success', 
                         f"Preprocessed data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self._log_step('data_preprocessing', 'error', str(e))
            raise
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train multiple models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training results
        """
        try:
            self._log_step('model_training', 'start', "Training models")
            
            # Train all models
            training_results = self.trainer.train_all_models(X_train, y_train, X_val, y_val)
            
            # Get model summary
            model_summary = self.trainer.get_model_summary()
            
            # Save training results
            output_dir = os.path.join(self.config['output_dir'], 'model_training')
            os.makedirs(output_dir, exist_ok=True)
            
            model_summary.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)
            
            # Save models
            self.trainer.save_models(os.path.join(output_dir, 'models'))
            
            # Create comparison plots
            plot_path = os.path.join(output_dir, 'model_comparison.png')
            self.trainer.plot_model_comparison(save_path=plot_path)
            
            self.results['model_training'] = {
                'best_model': self.trainer.best_model.__class__.__name__ if self.trainer.best_model else None,
                'best_score': self.trainer.best_score,
                'model_summary': model_summary.to_dict('records'),
                'training_results': training_results
            }
            
            self._log_step('model_training', 'success', 
                         f"Training completed. Best model: {self.results['model_training']['best_model']}")
            
            return training_results
            
        except Exception as e:
            self._log_step('model_training', 'error', str(e))
            raise
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate best model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            self._log_step('model_evaluation', 'start', "Evaluating model")
            
            if not self.trainer.best_model:
                raise ValueError("No trained model found. Train models first.")
            
            # Evaluate best model
            evaluation_metrics = self.trainer.evaluate_model(
                self.trainer.best_model, X_test, y_test
            )
            
            # Save evaluation results
            output_dir = os.path.join(self.config['output_dir'], 'model_evaluation')
            os.makedirs(output_dir, exist_ok=True)
            
            with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
                json.dump(evaluation_metrics, f, indent=2, default=str)
            
            self.results['model_evaluation'] = evaluation_metrics
            
            self._log_step('model_evaluation', 'success', 
                         f"Evaluation completed. Metrics: {evaluation_metrics}")
            
            return evaluation_metrics
            
        except Exception as e:
            self._log_step('model_evaluation', 'error', str(e))
            raise
    
    def deploy_model(self, model_name: str = None, deploy_type: str = None) -> str:
        """
        Deploy the best model
        
        Args:
            model_name: Name for deployed model
            deploy_type: Type of deployment ('api' or 'docker')
            
        Returns:
            Deployment information
        """
        try:
            self._log_step('model_deployment', 'start', "Deploying model")
            
            if not self.trainer.best_model:
                raise ValueError("No trained model found. Train models first.")
            
            if not model_name:
                model_name = f"{self.config['pipeline_name']}_best_model"
            
            if not deploy_type:
                deploy_type = self.config['model_deployment']['deployment_type']
            
            # Package model
            model_path = self.deployer.package_model(
                self.trainer.best_model,
                model_name,
                metadata={
                    'pipeline_name': self.config['pipeline_name'],
                    'training_config': self.config['model_training'],
                    'evaluation_metrics': self.results.get('model_evaluation', {})
                }
            )
            
            # Deploy model
            if deploy_type == 'api':
                self.deployer.deploy_api(model_name, model_path)
            elif deploy_type == 'docker':
                self.deployer.deploy_docker(model_name, model_path)
            else:
                raise ValueError(f"Unsupported deployment type: {deploy_type}")
            
            deployment_info = {
                'model_name': model_name,
                'model_path': model_path,
                'deployment_type': deploy_type,
                'deployed_at': datetime.now().isoformat()
            }
            
            self.results['model_deployment'] = deployment_info
            
            self._log_step('model_deployment', 'success', 
                         f"Model deployed as {deploy_type} service")
            
            return model_path
            
        except Exception as e:
            self._log_step('model_deployment', 'error', str(e))
            raise
    
    def run_full_pipeline(self, data_path: str, target_column: str = None,
                         deploy: bool = False) -> dict:
        """
        Run complete ML pipeline
        
        Args:
            data_path: Path to data file
            target_column: Name of target column
            deploy: Whether to deploy the model
            
        Returns:
            Dictionary with pipeline results
        """
        try:
            self._log_step('full_pipeline', 'start', "Starting full ML pipeline")
            
            # Step 1: Load data
            df = self.load_data(data_path)
            
            # Step 2: Explore data
            self.explore_data(df)
            
            # Step 3: Preprocess data
            X_train, X_test, y_train, y_test = self.preprocess_data(df, target_column)
            
            # Step 4: Train models
            training_results = self.train_models(X_train, y_train)
            
            # Step 5: Evaluate model
            evaluation_metrics = self.evaluate_model(X_test, y_test)
            
            # Step 6: Deploy model (if requested)
            if deploy:
                deployment_path = self.deploy_model()
                self.results['deployment_path'] = deployment_path
            
            # Save pipeline results
            self._save_pipeline_results()
            
            self._log_step('full_pipeline', 'success', "Full pipeline completed successfully")
            
            return self.results
            
        except Exception as e:
            self._log_step('full_pipeline', 'error', str(e))
            raise
    
    def _save_pipeline_results(self):
        """Save pipeline results and logs"""
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        with open(os.path.join(output_dir, 'pipeline_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save logs
        with open(os.path.join(output_dir, 'pipeline_logs.json'), 'w') as f:
            json.dump(self.pipeline_logs, f, indent=2, default=str)
        
        # Save configuration
        with open(os.path.join(output_dir, 'pipeline_config.json'), 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def get_pipeline_summary(self) -> dict:
        """Get summary of pipeline execution"""
        return {
            'pipeline_name': self.config['pipeline_name'],
            'total_steps': len(self.pipeline_logs),
            'successful_steps': len([log for log in self.pipeline_logs if log['status'] == 'success']),
            'failed_steps': len([log for log in self.pipeline_logs if log['status'] == 'error']),
            'execution_time': self.pipeline_logs[-1]['timestamp'] - self.pipeline_logs[0]['timestamp'] if self.pipeline_logs else None,
            'results_summary': {
                'data_shape': self.results.get('preprocessing', {}).get('X_train_shape'),
                'best_model': self.results.get('model_training', {}).get('best_model'),
                'best_score': self.results.get('model_training', {}).get('best_score'),
                'deployment_status': 'deployed' if 'model_deployment' in self.results else 'not_deployed'
            }
        }

def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Complete ML Pipeline")
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--target', type=str, help='Name of target column')
    parser.add_argument('--deploy', action='store_true', help='Deploy model after training')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MLPipeline(args.config)
    
    # Override output directory if specified
    if args.output:
        pipeline.config['output_dir'] = args.output
    
    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline(
            data_path=args.data,
            target_column=args.target,
            deploy=args.deploy
        )
        
        # Print summary
        summary = pipeline.get_pipeline_summary()
        print("\n" + "="*50)
        print("🎉 PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Pipeline Name: {summary['pipeline_name']}")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Successful Steps: {summary['successful_steps']}")
        print(f"Failed Steps: {summary['failed_steps']}")
        print(f"Best Model: {summary['results_summary']['best_model']}")
        print(f"Best Score: {summary['results_summary']['best_score']}")
        print(f"Deployment Status: {summary['results_summary']['deployment_status']}")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
