import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class ScorePredictor:
    """
    Model for predicting team scores in IPL matches.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        """
        Preprocess raw match data for model training or prediction.
        
        Args:
            data (pd.DataFrame): Raw match data
            
        Returns:
            tuple: X (features) and y (target) for training, or X for prediction
        """
        # Extract features
        features = [
            'batting_team_avg_score', 'batting_team_recent_form',
            'bowling_team_avg_economy', 'bowling_team_recent_form',
            'venue_avg_score', 'is_first_innings', 'is_day_night',
            'pitch_type', 'venue_id', 'batting_team_id', 'bowling_team_id'
        ]
        
        # One-hot encode categorical features
        data_encoded = pd.get_dummies(data, columns=['pitch_type', 'venue_id', 'batting_team_id', 'bowling_team_id'])
        
        # Get final feature set
        X = data_encoded.drop(['match_id', 'final_score'], axis=1, errors='ignore')
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        if 'final_score' in data.columns:
            # For training: return features and target
            y = data['final_score']
            return X_scaled, y
        else:
            # For prediction: return features only
            return X_scaled
    
    def train(self, data):
        """
        Train the score prediction model.
        
        Args:
            data (pd.DataFrame): Training data with match statistics
            
        Returns:
            float: Model MAE on test set
        """
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model MAE: {mae:.2f} runs")
        print(f"Model RMSE: {rmse:.2f} runs")
        print(f"Model R²: {r2:.4f}")
        
        return mae
    
    def predict(self, match_data):
        """
        Predict the score for a team in a match.
        
        Args:
            match_data (pd.DataFrame): Match data with team statistics
            
        Returns:
            dict: Prediction results with predicted score and confidence interval
        """
        X = self.preprocess_data(match_data)
        
        # Get prediction
        predicted_score = self.model.predict(X)[0]
        
        # Calculate confidence interval (simple approach)
        # In a real implementation, you might use more sophisticated methods
        confidence_interval = 15  # +/- 15 runs
        
        # Create result dictionary
        result = {
            'predicted_score': int(round(predicted_score)),
            'lower_bound': int(round(predicted_score - confidence_interval)),
            'upper_bound': int(round(predicted_score + confidence_interval))
        }
        
        return result
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    sample_data = pd.DataFrame({
        'match_id': range(100),
        'batting_team_id': np.random.randint(1, 9, 100),
        'bowling_team_id': np.random.randint(1, 9, 100),
        'batting_team_avg_score': np.random.uniform(150, 190, 100),
        'batting_team_recent_form': np.random.uniform(0, 1, 100),
        'bowling_team_avg_economy': np.random.uniform(7, 9, 100),
        'bowling_team_recent_form': np.random.uniform(0, 1, 100),
        'venue_avg_score': np.random.uniform(160, 200, 100),
        'is_first_innings': np.random.choice([0, 1], 100),
        'is_day_night': np.random.choice([0, 1], 100),
        'pitch_type': np.random.choice(['batting', 'bowling', 'balanced'], 100),
        'venue_id': np.random.randint(1, 10, 100),
        'final_score': np.random.randint(120, 220, 100)
    })
    
    # Initialize and train model
    predictor = ScorePredictor()
    predictor.train(sample_data)
    
    # Make a prediction
    new_match = sample_data.iloc[[0]].copy()
    new_match = new_match.drop('final_score', axis=1)
    prediction = predictor.predict(new_match)
    
    print("\nPrediction for new match:")
    print(f"Batting Team ID: {new_match['batting_team_id'].values[0]}")
    print(f"Bowling Team ID: {new_match['bowling_team_id'].values[0]}")
    print(f"Predicted score: {prediction['predicted_score']} runs")
    print(f"Confidence interval: {prediction['lower_bound']} - {prediction['upper_bound']} runs")
    
    # Save model
    predictor.save_model('score_predictor.joblib')
