#!/usr/bin/env python3
"""
Advanced Model Training Script for Gift Recommendation AI
This script implements machine learning techniques to improve recommendation accuracy
"""

import os
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GiftRecommendationTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.training_history = []
        
    def load_training_data(self, csv_path: str = 'products.csv') -> pd.DataFrame:
        """Load and preprocess training data"""
        print("Loading training data...")
        
        # Load CSV data
        df = pd.read_csv(csv_path, on_bad_lines='skip', low_memory=False, skiprows=3)
        
        # Basic preprocessing
        df = df.dropna(subset=['name'])
        
        # Create synthetic training data based on product features
        training_data = self._create_synthetic_training_data(df)
        
        print(f"Loaded {len(training_data)} training samples")
        return training_data
    
    def _create_synthetic_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic training data for model training"""
        training_samples = []
        
        # Define different user profiles and scenarios
        user_profiles = [
            {'age': 8, 'gender': 'male', 'interests': ['toys', 'games'], 'budget': 'low'},
            {'age': 15, 'gender': 'female', 'interests': ['fashion', 'beauty'], 'budget': 'medium'},
            {'age': 25, 'gender': 'male', 'interests': ['technology', 'gaming'], 'budget': 'high'},
            {'age': 35, 'gender': 'female', 'interests': ['home', 'cooking'], 'budget': 'medium'},
            {'age': 50, 'gender': 'male', 'interests': ['books', 'gardening'], 'budget': 'high'},
            {'age': 65, 'gender': 'female', 'interests': ['health', 'wellness'], 'budget': 'medium'},
        ]
        
        occasions = ['birthday', 'christmas', 'anniversary', 'graduation', 'wedding']
        
        for _, product in df.iterrows():
            # Create multiple training samples for each product
            for profile in user_profiles:
                for occasion in occasions:
                    # Calculate relevance score based on product features
                    relevance_score = self._calculate_relevance_score(product, profile, occasion)
                    
                    if relevance_score > 0.3:  # Only keep relevant samples
                        sample = {
                            'product_id': product.name,
                            'product_name': product.get('name', ''),
                            'product_category': product.get('main_category', ''),
                            'product_price': self._extract_price(product.get('actual_price', 0)),
                            'product_rating': self._extract_rating(product.get('ratings', 0)),
                            'user_age': profile['age'],
                            'user_gender': profile['gender'],
                            'user_interests': ','.join(profile['interests']),
                            'user_budget': profile['budget'],
                            'occasion': occasion,
                            'relevance_score': relevance_score,
                            'purchase_likelihood': min(1.0, relevance_score * 0.8 + np.random.normal(0, 0.1))
                        }
                        training_samples.append(sample)
        
        return pd.DataFrame(training_samples)
    
    def _calculate_relevance_score(self, product: pd.Series, profile: Dict, occasion: str) -> float:
        """Calculate relevance score between product and user profile"""
        score = 0.0
        
        # Age appropriateness
        product_text = str(product.get('name', '')).lower() + ' ' + str(product.get('description', '')).lower()
        
        if profile['age'] < 13:
            if any(word in product_text for word in ['toy', 'kids', 'child', 'baby']):
                score += 0.4
        elif profile['age'] < 18:
            if any(word in product_text for word in ['teen', 'youth', 'adolescent']):
                score += 0.4
        else:
            if any(word in product_text for word in ['adult', 'professional', 'mature']):
                score += 0.4
        
        # Interest matching
        for interest in profile['interests']:
            if interest.lower() in product_text:
                score += 0.3
        
        # Gender matching
        if profile['gender'] == 'male' and any(word in product_text for word in ['men', 'male', 'guy']):
            score += 0.2
        elif profile['gender'] == 'female' and any(word in product_text for word in ['women', 'female', 'girl']):
            score += 0.2
        
        # Occasion matching
        if occasion in product_text:
            score += 0.3
        
        # Price appropriateness
        price = self._extract_price(product.get('actual_price', 0))
        if profile['budget'] == 'low' and price < 1000:
            score += 0.2
        elif profile['budget'] == 'medium' and 500 <= price <= 3000:
            score += 0.2
        elif profile['budget'] == 'high' and price > 2000:
            score += 0.2
        
        return min(1.0, score)
    
    def _extract_price(self, price_str) -> float:
        """Extract numeric price from string"""
        try:
            if pd.isna(price_str):
                return 0.0
            price_str = str(price_str).replace('₹', '').replace(',', '')
            return float(price_str)
        except:
            return 0.0
    
    def _extract_rating(self, rating_str) -> float:
        """Extract numeric rating from string"""
        try:
            if pd.isna(rating_str):
                return 0.0
            return float(rating_str)
        except:
            return 0.0
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model training"""
        print("Preparing features...")
        
        # Text features
        text_features = df['product_name'].fillna('') + ' ' + df['product_category'].fillna('') + ' ' + df['user_interests'].fillna('')
        
        # TF-IDF for text features
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        text_vectors = tfidf.fit_transform(text_features)
        
        # Numerical features
        numerical_features = df[['product_price', 'product_rating', 'user_age', 'relevance_score']].values
        
        # Categorical features
        gender_encoder = LabelEncoder()
        budget_encoder = LabelEncoder()
        occasion_encoder = LabelEncoder()
        
        gender_encoded = gender_encoder.fit_transform(df['user_gender'].fillna('unknown'))
        budget_encoded = budget_encoder.fit_transform(df['user_budget'].fillna('unknown'))
        occasion_encoded = occasion_encoder.fit_transform(df['occasion'].fillna('unknown'))
        
        categorical_features = np.column_stack([gender_encoded, budget_encoded, occasion_encoded])
        
        # Combine all features
        X = np.hstack([text_vectors.toarray(), numerical_features, categorical_features])
        y = df['purchase_likelihood'].values
        
        # Store encoders
        self.encoders['gender'] = gender_encoder
        self.encoders['budget'] = budget_encoder
        self.encoders['occasion'] = occasion_encoder
        self.encoders['tfidf'] = tfidf
        
        print(f"Feature matrix shape: {X.shape}")
        return X, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train multiple models for ensemble prediction"""
        print("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_score = rf_model.score(X_test_scaled, y_test)
        print(f"Random Forest R² Score: {rf_score:.4f}")
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_score = gb_model.score(X_test_scaled, y_test)
        print(f"Gradient Boosting R² Score: {gb_score:.4f}")
        
        # Store models
        self.models['random_forest'] = rf_model
        self.models['gradient_boosting'] = gb_model
        
        # Feature importance
        self.feature_importance['random_forest'] = rf_model.feature_importances_
        self.feature_importance['gradient_boosting'] = gb_model.feature_importances_
        
        # Training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'models_trained': list(self.models.keys()),
            'scores': {'random_forest': rf_score, 'gradient_boosting': gb_score},
            'data_shape': X.shape
        })
        
        print("Model training completed!")
    
    def save_models(self, output_dir: str = 'trained_models'):
        """Save trained models and encoders"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name}.joblib'))
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(output_dir, f'{name}_scaler.joblib'))
        
        # Save encoders
        for name, encoder in self.encoders.items():
            joblib.dump(encoder, os.path.join(output_dir, f'{name}_encoder.joblib'))
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save feature importance
        with open(os.path.join(output_dir, 'feature_importance.json'), 'w') as f:
            json.dump({k: v.tolist() for k, v in self.feature_importance.items()}, f, indent=2)
        
        print(f"Models saved to {output_dir}/")
    
    def predict_relevance(self, product_data: Dict, user_profile: Dict, occasion: str) -> float:
        """Predict relevance score for a product-user-occasion combination"""
        if not self.models:
            print("No trained models available. Please train models first.")
            return 0.0
        
        # Prepare input features
        text_features = f"{product_data.get('name', '')} {product_data.get('category', '')} {','.join(user_profile.get('interests', []))}"
        
        # Transform text features
        text_vector = self.encoders['tfidf'].transform([text_features]).toarray()
        
        # Numerical features
        numerical_features = np.array([[
            product_data.get('price', 0),
            product_data.get('rating', 0),
            user_profile.get('age', 25),
            0.5  # Default relevance score
        ]])
        
        # Categorical features
        gender_encoded = self.encoders['gender'].transform([user_profile.get('gender', 'unknown')])
        budget_encoded = self.encoders['budget'].transform([user_profile.get('budget', 'medium')])
        occasion_encoded = self.encoders['occasion'].transform([occasion])
        
        categorical_features = np.column_stack([gender_encoded, budget_encoded, occasion_encoded])
        
        # Combine features
        X = np.hstack([text_vector, numerical_features, categorical_features])
        
        # Scale features
        X_scaled = self.scalers['standard'].transform(X)
        
        # Ensemble prediction
        predictions = []
        for model in self.models.values():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred)
        
        return np.mean(predictions)

def main():
    """Main training function"""
    print("🎁 Gift Recommendation AI - Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = GiftRecommendationTrainer()
    
    # Load and prepare data
    training_data = trainer.load_training_data()
    
    # Prepare features
    X, y = trainer.prepare_features(training_data)
    
    # Train models
    trainer.train_models(X, y)
    
    # Save models
    trainer.save_models()
    
    print("\n✅ Model training completed successfully!")
    print("Models are ready to be used in the recommendation service.")

if __name__ == "__main__":
    main()

