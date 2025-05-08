import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

class PlayerPredictor:
    """
    Model for predicting player performance in IPL matches.
    """
    
    def __init__(self, performance_type='batting'):
        """
        Initialize the player predictor.
        
        Args:
            performance_type (str): Type of performance to predict ('batting' or 'bowling')
        """
        self.performance_type = performance_type
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self, data):
        """
        Preprocess raw player data for model training or prediction.
        
        Args:
            data (pd.DataFrame): Raw player data
            
        Returns:
            tuple: X (features) and y (target) for training, or X for prediction
        """
        # Extract features based on performance type
        if self.performance_type == 'batting':
            features = [
                'player_batting_avg', 'player_strike_rate', 
                'player_recent_form', 'player_experience',
                'opposition_bowling_avg', 'opposition_economy_rate',
                'venue_batting_avg', 'pitch_type', 'is_home_ground',
                'batting_position', 'is_day_night', 'venue_id',
                'player_id', 'opposition_id'
            ]
            target = 'runs_scored'
        else:  # bowling
            features = [
                'player_bowling_avg', 'player_economy_rate', 
                'player_recent_form', 'player_experience',
                'opposition_batting_avg', 'opposition_strike_rate',
                'venue_bowling_avg', 'pitch_type', 'is_home_ground',
                'bowling_style', 'is_day_night', 'venue_id',
                'player_id', 'opposition_id'
            ]
            target = 'wickets_taken'
        
        # One-hot encode categorical features
        categorical_features = ['pitch_type', 'venue_id', 'player_id', 'opposition_id']
        if self.performance_type == 'batting':
            categorical_features.append('batting_position')
        else:
            categorical_features.append('bowling_style')
            
        data_encoded = pd.get_dummies(data, columns=categorical_features)
        
        # Get final feature set
        X = data_encoded.drop(['match_id', target], axis=1, errors='ignore')
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        if target in data.columns:
            # For training: return features and target
            y = data[target]
            return X_scaled, y
        else:
            # For prediction: return features only
            return X_scaled
    
    def train(self, data):
        """
        Train the player performance prediction model.
        
        Args:
            data (pd.DataFrame): Training data with player statistics
            
        Returns:
            float: Model MAE on test set
        """
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metric = "runs" if self.performance_type == 'batting' else "wickets"
        print(f"Model MAE: {mae:.2f} {metric}")
        print(f"Model RMSE: {rmse:.2f} {metric}")
        print(f"Model R²: {r2:.4f}")
        
        return mae
    
    def predict(self, player_data):
        """
        Predict the performance of a player in a match.
        
        Args:
            player_data (pd.DataFrame): Player data with match context
            
        Returns:
            dict: Prediction results with predicted performance and confidence interval
        """
        X = self.preprocess_data(player_data)
        
        # Get prediction
        predicted_value = self.model.predict(X)[0]
        
        # Calculate confidence interval (simple approach)
        # In a real implementation, you might use more sophisticated methods
        confidence_interval = 10 if self.performance_type == 'batting' else 5
        
        # Create result dictionary
        if self.performance_type == 'batting':
            result = {
                'predicted_runs': int(round(predicted_value)),
                'lower_bound': max(0, int(round(predicted_value - confidence_interval))),
                'upper_bound': int(round(predicted_value + confidence_interval))
            }
        else:  # bowling
            result = {
                'predicted_wickets': int(round(predicted_value)),
                'lower_bound': max(0, int(round(predicted_value - confidence_interval))),
                'upper_bound': min(10, int(round(predicted_value + confidence_interval)))
            }
        
        return result
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'performance_type': self.performance_type
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model from a file."""
        loaded = joblib.load(filepath)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.performance_type = loaded['performance_type']
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration - batting
    batting_data = pd.DataFrame({
        'match_id': range(100),
        'player_id': np.random.randint(1, 50, 100),
        'opposition_id': np.random.randint(1, 9, 100),
        'player_batting_avg': np.random.uniform(20, 50, 100),
        'player_strike_rate': np.random.uniform(120, 160, 100),
        'player_recent_form': np.random.uniform(0, 1, 100),
        'player_experience': np.random.randint(10, 100, 100),
        'opposition_bowling_avg': np.random.uniform(20, 35, 100),
        'opposition_economy_rate': np.random.uniform(7, 10, 100),
        'venue_batting_avg': np.random.uniform(25, 40, 100),
        'pitch_type': np.random.choice(['batting', 'bowling', 'balanced'], 100),
        'is_home_ground': np.random.choice([0, 1], 100),
        'batting_position': np.random.choice(['opener', 'top_order', 'middle_order', 'lower_order'], 100),
        'is_day_night': np.random.choice([0, 1], 100),
        'venue_id': np.random.randint(1, 10, 100),
        'runs_scored': np.random.randint(0, 100, 100)
    })
    
    # Initialize and train batting model
    batting_predictor = PlayerPredictor(performance_type='batting')
    batting_predictor.train(batting_data)
    
    # Make a batting prediction
    new_player = batting_data.iloc[[0]].copy()
    new_player = new_player.drop('runs_scored', axis=1)
    batting_prediction = batting_predictor.predict(new_player)
    
    print("\nBatting Prediction:")
    print(f"Player ID: {new_player['player_id'].values[0]}")
    print(f"Predicted runs: {batting_prediction['predicted_runs']}")
    print(f"Confidence interval: {batting_prediction['lower_bound']} - {batting_prediction['upper_bound']} runs")
    
    # Create sample data for demonstration - bowling
    bowling_data = pd.DataFrame({
        'match_id': range(100),
        'player_id': np.random.randint(1, 50, 100),
        'opposition_id': np.random.randint(1, 9, 100),
        'player_bowling_avg': np.random.uniform(20, 35, 100),
        'player_economy_rate': np.random.uniform(7, 10, 100),
        'player_recent_form': np.random.uniform(0, 1, 100),
        'player_experience': np.random.randint(10, 100, 100),
        'opposition_batting_avg': np.random.uniform(25, 40, 100),
        'opposition_strike_rate': np.random.uniform(120, 160, 100),
        'venue_bowling_avg': np.random.uniform(25, 35, 100),
        'pitch_type': np.random.choice(['batting', 'bowling', 'balanced'], 100),
        'is_home_ground': np.random.choice([0, 1], 100),
        'bowling_style': np.random.choice(['fast', 'medium', 'spin'], 100),
        'is_day_night': np.random.choice([0, 1], 100),
        'venue_id': np.random.randint(1, 10, 100),
        'wickets_taken': np.random.randint(0, 5, 100)
    })
    
    # Initialize and train bowling model
    bowling_predictor = PlayerPredictor(performance_type='bowling')
    bowling_predictor.train(bowling_data)
    
    # Make a bowling prediction
    new_bowler = bowling_data.iloc[[0]].copy()
    new_bowler = new_bowler.drop('wickets_taken', axis=1)
    bowling_prediction = bowling_predictor.predict(new_bowler)
    
    print("\nBowling Prediction:")
    print(f"Player ID: {new_bowler['player_id'].values[0]}")
    print(f"Predicted wickets: {bowling_prediction['predicted_wickets']}")
    print(f"Confidence interval: {bowling_prediction['lower_bound']} - {bowling_prediction['upper_bound']} wickets")
    
    # Save models
    batting_predictor.save_model('batting_predictor.joblib')
    bowling_predictor.save_model('bowling_predictor.joblib')
