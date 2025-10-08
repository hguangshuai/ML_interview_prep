"""
ML Pipeline - Model Deployment Module
模型部署模块，支持模型打包、API服务、监控和版本管理
"""

import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from flask import Flask, request, jsonify
import docker
import requests
from sklearn.metrics import accuracy_score, mean_squared_error
import psutil
import time

class ModelDeployer:
    """
    Comprehensive model deployment pipeline
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize model deployer with configuration
        
        Args:
            config: Configuration dictionary with deployment parameters
        """
        self.config = config or self._default_config()
        self.deployed_models = {}
        self.model_versions = {}
        self.deployment_logs = []
        
        # Setup logging
        self._setup_logging()
        
    def _default_config(self) -> dict:
        """Default configuration for model deployment"""
        return {
            'deployment_type': 'api',  # 'api', 'batch', 'streaming'
            'model_format': 'joblib',  # 'joblib', 'pickle', 'onnx'
            'api_port': 5000,
            'api_host': '0.0.0.0',
            'max_request_size': '16MB',
            'timeout': 30,
            'model_dir': 'deployed_models/',
            'log_dir': 'deployment_logs/',
            'monitoring_enabled': True,
            'version_control': True,
            'health_check_interval': 60,
            'performance_threshold': 0.8
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.config['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'deployment.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def package_model(self, model, model_name: str, metadata: dict = None) -> str:
        """
        Package model with metadata for deployment
        
        Args:
            model: Trained model object
            model_name: Name of the model
            metadata: Additional metadata about the model
            
        Returns:
            Path to packaged model
        """
        self.logger.info(f"📦 Packaging model: {model_name}")
        
        # Create model directory
        model_dir = os.path.join(self.config['model_dir'], model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate version
        version = self._generate_version(model_name)
        version_dir = os.path.join(model_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, f"{model_name}.{self.config['model_format']}")
        
        if self.config['model_format'] == 'joblib':
            joblib.dump(model, model_path)
        elif self.config['model_format'] == 'pickle':
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported model format: {self.config['model_format']}")
        
        # Save metadata
        model_metadata = {
            'model_name': model_name,
            'version': version,
            'created_at': datetime.now().isoformat(),
            'model_format': self.config['model_format'],
            'model_path': model_path,
            'metadata': metadata or {}
        }
        
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        # Update version tracking
        if model_name not in self.model_versions:
            self.model_versions[model_name] = []
        self.model_versions[model_name].append(version)
        
        self.logger.info(f"✅ Model packaged: {model_path}")
        return model_path
    
    def _generate_version(self, model_name: str) -> str:
        """Generate version number for model"""
        if model_name not in self.model_versions:
            return "v1.0.0"
        
        # Simple version increment
        current_versions = self.model_versions[model_name]
        version_number = len(current_versions) + 1
        return f"v{version_number}.0.0"
    
    def create_api_service(self, model_name: str, model_path: str) -> Flask:
        """
        Create Flask API service for model
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model
            
        Returns:
            Flask application
        """
        self.logger.info(f"🚀 Creating API service for {model_name}")
        
        app = Flask(__name__)
        
        # Load model
        if self.config['model_format'] == 'joblib':
            model = joblib.load(model_path)
        elif self.config['model_format'] == 'pickle':
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # Store model in deployed models
        self.deployed_models[model_name] = {
            'model': model,
            'model_path': model_path,
            'deployed_at': datetime.now().isoformat(),
            'request_count': 0,
            'error_count': 0
        }
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            })
        
        @app.route('/predict', methods=['POST'])
        def predict():
            """Prediction endpoint"""
            try:
                # Increment request count
                self.deployed_models[model_name]['request_count'] += 1
                
                # Get input data
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Convert to appropriate format
                if 'features' in data:
                    features = np.array(data['features'])
                elif 'data' in data:
                    features = np.array(data['data'])
                else:
                    return jsonify({'error': 'No features provided'}), 400
                
                # Make prediction
                prediction = model.predict(features)
                
                # Handle different prediction formats
                if hasattr(prediction, 'tolist'):
                    prediction = prediction.tolist()
                
                # Log prediction
                self._log_prediction(model_name, features, prediction)
                
                return jsonify({
                    'prediction': prediction,
                    'model_name': model_name,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                # Increment error count
                self.deployed_models[model_name]['error_count'] += 1
                
                self.logger.error(f"Prediction error: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/model_info', methods=['GET'])
        def model_info():
            """Get model information"""
            return jsonify({
                'model_name': model_name,
                'model_path': model_path,
                'deployed_at': self.deployed_models[model_name]['deployed_at'],
                'request_count': self.deployed_models[model_name]['request_count'],
                'error_count': self.deployed_models[model_name]['error_count']
            })
        
        return app
    
    def deploy_api(self, model_name: str, model_path: str, port: int = None):
        """
        Deploy model as API service
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model
            port: Port number for API
        """
        if not port:
            port = self.config['api_port']
        
        app = self.create_api_service(model_name, model_path)
        
        self.logger.info(f"🌐 Deploying API service on port {port}")
        
        # Start monitoring if enabled
        if self.config['monitoring_enabled']:
            self._start_monitoring(model_name)
        
        # Run Flask app
        app.run(host=self.config['api_host'], port=port, debug=False)
    
    def create_docker_image(self, model_name: str, model_path: str) -> str:
        """
        Create Docker image for model deployment
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model
            
        Returns:
            Docker image name
        """
        self.logger.info(f"🐳 Creating Docker image for {model_name}")
        
        # Create Dockerfile
        dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy model and application
COPY {model_name}_api.py .
COPY {os.path.dirname(model_path)}/ ./models/

# Expose port
EXPOSE {self.config['api_port']}

# Run application
CMD ["python", "{model_name}_api.py"]
"""
        
        # Create API script
        api_script = f"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model_path = os.path.join('models', '{os.path.basename(model_path)}')
model = joblib.load(model_path)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({{'status': 'healthy'}})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features'])
        prediction = model.predict(features)
        return jsonify({{'prediction': prediction.tolist()}})
    except Exception as e:
        return jsonify({{'error': str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port={self.config['api_port']})
"""
        
        # Create deployment directory
        deploy_dir = f"docker_deploy_{model_name}"
        os.makedirs(deploy_dir, exist_ok=True)
        
        # Write files
        with open(os.path.join(deploy_dir, 'Dockerfile'), 'w') as f:
            f.write(dockerfile_content)
        
        with open(os.path.join(deploy_dir, f'{model_name}_api.py'), 'w') as f:
            f.write(api_script)
        
        # Create requirements.txt
        requirements = """
flask==2.3.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
"""
        with open(os.path.join(deploy_dir, 'requirements.txt'), 'w') as f:
            f.write(requirements)
        
        # Build Docker image
        try:
            client = docker.from_env()
            image_name = f"{model_name}:latest"
            
            # Build image
            image, build_logs = client.images.build(
                path=deploy_dir,
                tag=image_name,
                rm=True
            )
            
            self.logger.info(f"✅ Docker image created: {image_name}")
            return image_name
            
        except Exception as e:
            self.logger.error(f"Docker build failed: {str(e)}")
            raise
    
    def deploy_docker(self, model_name: str, model_path: str, port: int = None):
        """
        Deploy model using Docker
        
        Args:
            model_name: Name of the model
            model_path: Path to saved model
            port: Port number for API
        """
        if not port:
            port = self.config['api_port']
        
        # Create Docker image
        image_name = self.create_docker_image(model_name, model_path)
        
        # Run container
        try:
            client = docker.from_env()
            container = client.containers.run(
                image_name,
                ports={f'{self.config["api_port"]}/tcp': port},
                detach=True,
                name=f"{model_name}_api"
            )
            
            self.logger.info(f"🚀 Docker container started: {container.name}")
            
        except Exception as e:
            self.logger.error(f"Docker deployment failed: {str(e)}")
            raise
    
    def _log_prediction(self, model_name: str, features: np.ndarray, prediction: Any):
        """Log prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'features_shape': features.shape,
            'prediction': str(prediction),
            'request_id': f"{model_name}_{int(time.time())}"
        }
        
        self.deployment_logs.append(log_entry)
        
        # Keep only recent logs
        if len(self.deployment_logs) > 1000:
            self.deployment_logs = self.deployment_logs[-1000:]
    
    def _start_monitoring(self, model_name: str):
        """Start monitoring for deployed model"""
        self.logger.info(f"📊 Starting monitoring for {model_name}")
        
        # This would typically run in a separate thread
        # For now, we'll just log the start
        pass
    
    def get_model_performance(self, model_name: str) -> dict:
        """
        Get performance metrics for deployed model
        
        Args:
            model_name: Name of the deployed model
            
        Returns:
            Dictionary with performance metrics
        """
        if model_name not in self.deployed_models:
            return {'error': 'Model not deployed'}
        
        model_info = self.deployed_models[model_name]
        
        # Calculate metrics
        total_requests = model_info['request_count']
        error_count = model_info['error_count']
        success_rate = (total_requests - error_count) / total_requests if total_requests > 0 else 0
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        performance = {
            'model_name': model_name,
            'total_requests': total_requests,
            'error_count': error_count,
            'success_rate': success_rate,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_percent,
            'uptime': datetime.now().isoformat(),
            'status': 'healthy' if success_rate > self.config['performance_threshold'] else 'degraded'
        }
        
        return performance
    
    def rollback_model(self, model_name: str, version: str = None):
        """
        Rollback model to previous version
        
        Args:
            model_name: Name of the model
            version: Specific version to rollback to (if None, rollback to previous)
        """
        self.logger.info(f"🔄 Rolling back model {model_name}")
        
        if model_name not in self.model_versions:
            raise ValueError(f"No versions found for model {model_name}")
        
        if not version:
            # Rollback to previous version
            if len(self.model_versions[model_name]) < 2:
                raise ValueError("No previous version to rollback to")
            version = self.model_versions[model_name][-2]
        
        # Update deployed model
        model_path = os.path.join(
            self.config['model_dir'], 
            model_name, 
            version, 
            f"{model_name}.{self.config['model_format']}"
        )
        
        if os.path.exists(model_path):
            self.deployed_models[model_name]['model_path'] = model_path
            self.logger.info(f"✅ Rolled back to version {version}")
        else:
            raise ValueError(f"Version {version} not found")
    
    def get_deployment_status(self) -> dict:
        """
        Get status of all deployed models
        
        Returns:
            Dictionary with deployment status
        """
        status = {
            'total_models': len(self.deployed_models),
            'models': {}
        }
        
        for model_name in self.deployed_models:
            status['models'][model_name] = self.get_model_performance(model_name)
        
        return status
    
    def cleanup_old_versions(self, model_name: str, keep_versions: int = 5):
        """
        Clean up old model versions
        
        Args:
            model_name: Name of the model
            keep_versions: Number of versions to keep
        """
        if model_name not in self.model_versions:
            return
        
        versions = self.model_versions[model_name]
        if len(versions) <= keep_versions:
            return
        
        # Remove old versions
        versions_to_remove = versions[:-keep_versions]
        
        for version in versions_to_remove:
            version_dir = os.path.join(
                self.config['model_dir'], 
                model_name, 
                version
            )
            
            if os.path.exists(version_dir):
                import shutil
                shutil.rmtree(version_dir)
                self.logger.info(f"🗑️  Removed old version: {version}")
        
        # Update version list
        self.model_versions[model_name] = versions[-keep_versions:]

# Example usage
def example_usage():
    """Example usage of ModelDeployer"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Create sample model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Initialize deployer
    config = {
        'deployment_type': 'api',
        'model_format': 'joblib',
        'api_port': 5000,
        'model_dir': 'example_models/',
        'monitoring_enabled': True
    }
    
    deployer = ModelDeployer(config)
    
    # Package model
    model_path = deployer.package_model(
        model, 
        'random_forest_classifier',
        metadata={'task': 'classification', 'features': 20}
    )
    
    print(f"📦 Model packaged: {model_path}")
    
    # Get deployment status
    status = deployer.get_deployment_status()
    print(f"📊 Deployment status: {status}")
    
    # Note: In a real scenario, you would deploy the API service
    # deployer.deploy_api('random_forest_classifier', model_path)

if __name__ == "__main__":
    example_usage()
