"""
ML Pipeline - Data Preparation Module
数据准备和预处理模块，包含数据清洗、特征工程、数据分割等功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize data preprocessor with configuration
        
        Args:
            config: Configuration dictionary with preprocessing parameters
        """
        self.config = config or self._default_config()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.feature_names = None
        
    def _default_config(self) -> dict:
        """Default configuration for data preprocessing"""
        return {
            'test_size': 0.2,
            'random_state': 42,
            'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
            'encoding_method': 'onehot',   # 'onehot', 'label', 'target'
            'imputation_method': 'mean',   # 'mean', 'median', 'mode', 'constant'
            'handle_outliers': True,
            'outlier_method': 'iqr',       # 'iqr', 'zscore', 'isolation'
            'feature_selection': False,
            'correlation_threshold': 0.95
        }
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        file_extension = file_path.split('.')[-1].lower()
        
        if file_extension == 'csv':
            return pd.read_csv(file_path, **kwargs)
        elif file_extension in ['xlsx', 'xls']:
            return pd.read_excel(file_path, **kwargs)
        elif file_extension == 'json':
            return pd.read_json(file_path, **kwargs)
        elif file_extension == 'parquet':
            return pd.read_parquet(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def explore_data(self, df: pd.DataFrame) -> dict:
        """
        Comprehensive data exploration
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing exploration results
        """
        exploration_results = {
            'shape': df.shape,
            'dtypes': df.dtypes,
            'missing_values': df.isnull().sum(),
            'missing_percentage': (df.isnull().sum() / len(df)) * 100,
            'numeric_summary': df.describe(),
            'categorical_summary': df.describe(include=['object']),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return exploration_results
    
    def visualize_data(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive data visualizations
        
        Args:
            df: Input DataFrame
            save_path: Optional path to save plots
        """
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Missing values heatmap
        plt.subplot(3, 3, 1)
        sns.heatmap(df.isnull(), cbar=True, yticklabels=False)
        plt.title('Missing Values Heatmap')
        
        # 2. Data types distribution
        plt.subplot(3, 3, 2)
        df.dtypes.value_counts().plot(kind='bar')
        plt.title('Data Types Distribution')
        plt.xticks(rotation=45)
        
        # 3. Missing values by column
        plt.subplot(3, 3, 3)
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            missing_data.plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.xticks(rotation=45)
        
        # 4. Numeric features distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plt.subplot(3, 3, 4)
            df[numeric_cols].hist(bins=20, alpha=0.7)
            plt.title('Numeric Features Distribution')
        
        # 5. Correlation heatmap
        if len(numeric_cols) > 1:
            plt.subplot(3, 3, 5)
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Heatmap')
        
        # 6. Outliers detection
        if len(numeric_cols) > 0:
            plt.subplot(3, 3, 6)
            df[numeric_cols].boxplot()
            plt.title('Outliers Detection')
            plt.xticks(rotation=45)
        
        # 7. Categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            plt.subplot(3, 3, 7)
            df[categorical_cols].nunique().plot(kind='bar')
            plt.title('Unique Values in Categorical Features')
            plt.xticks(rotation=45)
        
        # 8. Target variable distribution (if exists)
        if 'target' in df.columns or 'label' in df.columns:
            target_col = 'target' if 'target' in df.columns else 'label'
            plt.subplot(3, 3, 8)
            df[target_col].value_counts().plot(kind='bar')
            plt.title(f'{target_col.title()} Distribution')
            plt.xticks(rotation=45)
        
        # 9. Data quality score
        plt.subplot(3, 3, 9)
        quality_score = self._calculate_data_quality_score(df)
        plt.bar(['Data Quality Score'], [quality_score])
        plt.title(f'Overall Data Quality: {quality_score:.2f}/10')
        plt.ylim(0, 10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        score = 10.0
        
        # Deduct points for missing values
        missing_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        score -= missing_percentage * 0.1
        
        # Deduct points for duplicates
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        score -= duplicate_percentage * 0.05
        
        # Deduct points for high correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr().abs()
            high_corr_pairs = (correlation_matrix > 0.95).sum().sum() - len(numeric_cols)
            score -= high_corr_pairs * 0.1
        
        return max(0, score)
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_processed = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        if len(numeric_cols) > 0:
            if self.config['imputation_method'] == 'mean':
                imputer = SimpleImputer(strategy='mean')
            elif self.config['imputation_method'] == 'median':
                imputer = SimpleImputer(strategy='median')
            elif self.config['imputation_method'] == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value=0)
            else:
                imputer = SimpleImputer(strategy='mean')
            
            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            self.imputers['numeric'] = imputer
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            if self.config['imputation_method'] == 'mode':
                imputer = SimpleImputer(strategy='most_frequent')
            elif self.config['imputation_method'] == 'constant':
                imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
            else:
                imputer = SimpleImputer(strategy='most_frequent')
            
            df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
            self.imputers['categorical'] = imputer
        
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numeric columns
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with outliers handled
        """
        if not self.config['handle_outliers']:
            return df
        
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.config['outlier_method'] == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
            
            elif self.config['outlier_method'] == 'zscore':
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                df_processed = df_processed[z_scores < 3]
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return df_processed
        
        if self.config['encoding_method'] == 'onehot':
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_features = encoder.fit_transform(df_processed[categorical_cols])
            
            # Create feature names
            feature_names = []
            for i, col in enumerate(categorical_cols):
                categories = encoder.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df_processed.index)
            
            # Drop original categorical columns and add encoded ones
            df_processed = df_processed.drop(categorical_cols, axis=1)
            df_processed = pd.concat([df_processed, encoded_df], axis=1)
            
            self.encoders['onehot'] = encoder
        
        elif self.config['encoding_method'] == 'label':
            for col in categorical_cols:
                encoder = LabelEncoder()
                df_processed[col] = encoder.fit_transform(df_processed[col].astype(str))
                self.encoders[f'label_{col}'] = encoder
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numeric features
        
        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            DataFrame with scaled features
        """
        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df_processed
        
        if fit:
            if self.config['scaling_method'] == 'standard':
                scaler = StandardScaler()
            elif self.config['scaling_method'] == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
            elif self.config['scaling_method'] == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
            self.scalers['main'] = scaler
        else:
            if 'main' in self.scalers:
                df_processed[numeric_cols] = self.scalers['main'].transform(df_processed[numeric_cols])
        
        return df_processed
    
    def split_data(self, df: pd.DataFrame, target_column: str = None) -> tuple:
        """
        Split data into training and testing sets
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if target_column and target_column in df.columns:
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y if len(y.unique()) > 1 else None
            )
            
            return X_train, X_test, y_train, y_test
        else:
            # If no target column, just split features
            X_train, X_test = train_test_split(
                df,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
            
            return X_train, X_test, None, None
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str = None, 
                          fit: bool = True) -> tuple:
        """
        Complete preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            fit: Whether to fit transformers (True for training, False for testing)
            
        Returns:
            Tuple of processed data
        """
        print("🔄 Starting data preprocessing pipeline...")
        
        # Step 1: Handle missing values
        print("📝 Handling missing values...")
        df_processed = self.handle_missing_values(df)
        
        # Step 2: Handle outliers
        print("🔍 Handling outliers...")
        df_processed = self.handle_outliers(df_processed)
        
        # Step 3: Encode categorical features
        print("🏷️  Encoding categorical features...")
        df_processed = self.encode_categorical_features(df_processed)
        
        # Step 4: Scale features
        print("📏 Scaling features...")
        df_processed = self.scale_features(df_processed, fit=fit)
        
        # Step 5: Split data
        print("✂️  Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(df_processed, target_column)
        
        # Store feature names
        if fit:
            self.feature_names = X_train.columns.tolist()
        
        print("✅ Data preprocessing completed!")
        
        return X_train, X_test, y_train, y_test

# Example usage
def example_usage():
    """Example usage of DataPreprocessor"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(5, 2, n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.choice(['X', 'Y'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    # Add some missing values and outliers
    data['feature1'][:50] = np.nan
    data['feature2'][-20:] = data['feature2'][-20:] * 10  # Outliers
    
    df = pd.DataFrame(data)
    
    # Initialize preprocessor
    config = {
        'test_size': 0.2,
        'random_state': 42,
        'scaling_method': 'standard',
        'encoding_method': 'onehot',
        'imputation_method': 'mean',
        'handle_outliers': True,
        'outlier_method': 'iqr'
    }
    
    preprocessor = DataPreprocessor(config)
    
    # Explore data
    exploration_results = preprocessor.explore_data(df)
    print("📊 Data Exploration Results:")
    print(f"Shape: {exploration_results['shape']}")
    print(f"Missing values: {exploration_results['missing_values'].sum()}")
    print(f"Duplicates: {exploration_results['duplicates']}")
    
    # Visualize data
    preprocessor.visualize_data(df)
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocessor.preprocess_pipeline(
        df, target_column='target', fit=True
    )
    
    print(f"\n📈 Final processed data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

if __name__ == "__main__":
    example_usage()
