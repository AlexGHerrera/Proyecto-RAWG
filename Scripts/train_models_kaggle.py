#!/usr/bin/env python3
"""
RAWG Game Success Prediction - Kaggle Training Script
Trains XGBoost and Neural Network models for game success prediction
Author: Alex G. Herrera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GameSuccessPredictor:
    """
    Complete pipeline for training game success prediction models
    """
    
    def __init__(self, data_path):
        """Initialize with data path"""
        self.data_path = data_path
        self.models = {}
        self.scalers = {}
        self.results = {}
        
    def load_and_prepare_data(self):
        """Load and prepare the training dataset"""
        print("🔄 Loading dataset...")
        
        # Try different file formats
        try:
            if self.data_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.data_path)
            elif self.data_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.data_path)
            else:
                self.df = pd.read_csv(self.data_path)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
            
        print(f"✅ Dataset loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        
        # Define features and target
        self.design_features = ['n_genres', 'n_platforms', 'n_tags', 'esrb_rating_id', 'estimated_hours', 'planned_year']
        self.target = 'success_score'
        
        # Check if all required columns exist
        missing_cols = [col for col in self.design_features + [self.target] if col not in self.df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
            
        # Prepare features and target
        self.X = self.df[self.design_features].copy()
        self.y = self.df[self.target].copy()
        
        # Handle missing values
        self.X = self.X.fillna(0)
        self.y = self.y.fillna(self.y.median())
        
        # Basic data validation
        print(f"📊 Features shape: {self.X.shape}")
        print(f"📊 Target range: {self.y.min():.4f} - {self.y.max():.4f}")
        print(f"📊 Target mean: {self.y.mean():.4f}")
        
        return True
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """Split data into train/validation/test sets"""
        print("🔄 Splitting data...")
        
        # First split: train+val vs test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        print(f"✅ Train: {self.X_train.shape[0]:,} | Val: {self.X_val.shape[0]:,} | Test: {self.X_test.shape[0]:,}")
        
        # Scale features for neural network
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def train_xgboost(self, tune_hyperparams=True):
        """Train XGBoost model with optional hyperparameter tuning"""
        print("🚀 Training XGBoost...")
        
        if tune_hyperparams:
            print("🔧 Tuning hyperparameters...")
            param_grid = {
                'n_estimators': [300, 500, 800],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_base = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                xgb_base, param_grid, cv=3, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            self.models['xgboost'] = grid_search.best_estimator_
            print(f"✅ Best XGB params: {grid_search.best_params_}")
        else:
            # Use default good parameters
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                n_jobs=-1
            )
            self.models['xgboost'].fit(self.X_train, self.y_train)
        
        # Predictions
        xgb_pred_val = self.models['xgboost'].predict(self.X_val)
        xgb_pred_test = self.models['xgboost'].predict(self.X_test)
        
        # Store results
        self.results['xgboost'] = {
            'val_pred': xgb_pred_val,
            'test_pred': xgb_pred_test,
            'feature_importance': dict(zip(self.design_features, self.models['xgboost'].feature_importances_))
        }
        
        print("✅ XGBoost training completed!")
        
    def train_neural_network(self, epochs=200, batch_size=32):
        """Train Neural Network model"""
        print("🧠 Training Neural Network...")
        
        # Build model architecture
        model = Sequential([
            Dense(128, activation='relu', input_shape=(len(self.design_features),)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            Dense(1, activation='sigmoid')  # Output 0-1 for success_score
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6
        )
        
        # Train model
        history = model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_val_scaled, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.models['neural_network'] = model
        
        # Predictions
        nn_pred_val = model.predict(self.X_val_scaled).flatten()
        nn_pred_test = model.predict(self.X_test_scaled).flatten()
        
        # Store results
        self.results['neural_network'] = {
            'val_pred': nn_pred_val,
            'test_pred': nn_pred_test,
            'history': history.history
        }
        
        print("✅ Neural Network training completed!")
        
    def train_baseline_models(self):
        """Train baseline models for comparison"""
        print("📊 Training baseline models...")
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model
        
        rf_pred_val = rf_model.predict(self.X_val)
        rf_pred_test = rf_model.predict(self.X_test)
        
        self.results['random_forest'] = {
            'val_pred': rf_pred_val,
            'test_pred': rf_pred_test,
            'feature_importance': dict(zip(self.design_features, rf_model.feature_importances_))
        }
        
        print("✅ Baseline models completed!")
        
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n🏆 MODEL EVALUATION RESULTS")
        print("=" * 60)
        
        evaluation_results = {}
        
        for model_name, results in self.results.items():
            # Validation metrics
            val_mse = mean_squared_error(self.y_val, results['val_pred'])
            val_rmse = np.sqrt(val_mse)
            val_mae = mean_absolute_error(self.y_val, results['val_pred'])
            val_r2 = r2_score(self.y_val, results['val_pred'])
            
            # Test metrics
            test_mse = mean_squared_error(self.y_test, results['test_pred'])
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(self.y_test, results['test_pred'])
            test_r2 = r2_score(self.y_test, results['test_pred'])
            
            evaluation_results[model_name] = {
                'val_rmse': val_rmse, 'val_mae': val_mae, 'val_r2': val_r2,
                'test_rmse': test_rmse, 'test_mae': test_mae, 'test_r2': test_r2
            }
            
            print(f"\n{model_name.upper():15}")
            print(f"  Validation  | RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | R²: {val_r2:.4f}")
            print(f"  Test        | RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
        
        self.evaluation_results = evaluation_results
        
        # Find best model
        best_model = min(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['test_rmse'])
        print(f"\n🥇 BEST MODEL: {best_model.upper()} (lowest test RMSE)")
        
        return best_model
        
    def plot_results(self):
        """Create comprehensive result visualizations"""
        print("📊 Creating visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Model Comparison - RMSE
        model_names = list(self.evaluation_results.keys())
        test_rmse = [self.evaluation_results[m]['test_rmse'] for m in model_names]
        
        bars = axes[0, 0].bar(model_names, test_rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 0].set_title('Model Comparison - Test RMSE', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, value in zip(bars, test_rmse):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. R² Comparison
        test_r2 = [self.evaluation_results[m]['test_r2'] for m in model_names]
        bars = axes[0, 1].bar(model_names, test_r2, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0, 1].set_title('Model Comparison - Test R²', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, test_r2):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Feature Importance (XGBoost)
        if 'xgboost' in self.results:
            importance = self.results['xgboost']['feature_importance']
            features = list(importance.keys())
            values = list(importance.values())
            
            axes[0, 2].barh(features, values, color='#ff7f0e')
            axes[0, 2].set_title('XGBoost Feature Importance', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Importance')
        
        # 4. Predictions vs Actual (XGBoost)
        if 'xgboost' in self.results:
            axes[1, 0].scatter(self.y_test, self.results['xgboost']['test_pred'], 
                              alpha=0.6, color='#ff7f0e')
            axes[1, 0].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 0].set_title('XGBoost: Predictions vs Actual', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Actual Success Score')
            axes[1, 0].set_ylabel('Predicted Success Score')
        
        # 5. Predictions vs Actual (Neural Network)
        if 'neural_network' in self.results:
            axes[1, 1].scatter(self.y_test, self.results['neural_network']['test_pred'], 
                              alpha=0.6, color='#2ca02c')
            axes[1, 1].plot([self.y_test.min(), self.y_test.max()], 
                           [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes[1, 1].set_title('Neural Network: Predictions vs Actual', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Actual Success Score')
            axes[1, 1].set_ylabel('Predicted Success Score')
        
        # 6. Training History (Neural Network)
        if 'neural_network' in self.results and 'history' in self.results['neural_network']:
            history = self.results['neural_network']['history']
            axes[1, 2].plot(history['loss'], label='Training Loss', color='#2ca02c')
            axes[1, 2].plot(history['val_loss'], label='Validation Loss', color='#ff7f0e')
            axes[1, 2].set_title('Neural Network Training History', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Loss (MSE)')
            axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_models(self):
        """Save trained models"""
        print("💾 Saving models...")
        
        # Save XGBoost
        if 'xgboost' in self.models:
            self.models['xgboost'].save_model('xgboost_model.json')
            print("✅ XGBoost model saved as 'xgboost_model.json'")
        
        # Save Neural Network
        if 'neural_network' in self.models:
            self.models['neural_network'].save('neural_network_model.h5')
            print("✅ Neural Network model saved as 'neural_network_model.h5'")
        
        # Save scaler
        import joblib
        joblib.dump(self.scaler, 'feature_scaler.pkl')
        print("✅ Feature scaler saved as 'feature_scaler.pkl'")
        
    def run_complete_pipeline(self, tune_xgb=False):
        """Run the complete training pipeline"""
        print("🚀 STARTING COMPLETE TRAINING PIPELINE")
        print("=" * 50)
        
        # Load and prepare data
        if not self.load_and_prepare_data():
            return False
            
        # Split data
        self.split_data()
        
        # Train models
        self.train_xgboost(tune_hyperparams=tune_xgb)
        self.train_neural_network()
        self.train_baseline_models()
        
        # Evaluate
        best_model = self.evaluate_models()
        
        # Visualize
        self.plot_results()
        
        # Save models
        self.save_models()
        
        print(f"\n🎉 PIPELINE COMPLETED! Best model: {best_model.upper()}")
        return True

def main():
    """Main execution function"""
    print("🎮 RAWG Game Success Prediction - Kaggle Training")
    print("=" * 50)
    
    # Configuration
    DATA_PATH = "/kaggle/input/rawg-games/training_dataset.csv"  # Adjust path for Kaggle
    TUNE_HYPERPARAMS = False  # Set to True for hyperparameter tuning (takes longer)
    
    # Alternative paths to try
    possible_paths = [
        "/kaggle/input/rawg-games/training_dataset.csv",
        "/kaggle/input/rawg-games/training_dataset.parquet",
        "/kaggle/input/rawg-games/design_features_dataset.csv",
        "training_dataset.csv",  # If uploaded directly
        "design_features_dataset.csv"
    ]
    
    # Find the correct data path
    import os
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if data_path is None:
        print("❌ No dataset found. Please upload one of:")
        for path in possible_paths:
            print(f"   - {path}")
        return
    
    print(f"📁 Using dataset: {data_path}")
    
    # Initialize and run pipeline
    predictor = GameSuccessPredictor(data_path)
    success = predictor.run_complete_pipeline(tune_xgb=TUNE_HYPERPARAMS)
    
    if success:
        print("\n✅ Training completed successfully!")
        print("📁 Generated files:")
        print("   - xgboost_model.json")
        print("   - neural_network_model.h5") 
        print("   - feature_scaler.pkl")
        print("   - model_results.png")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
