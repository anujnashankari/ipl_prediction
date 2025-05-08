import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib

class MatchPredictor:
    """
    Model for predicting IPL match winners based on team statistics and match conditions.
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
            'team1_win_rate', 'team2_win_rate',
            'team1_home_advantage', 'team2_home_advantage',
            'team1_recent_form', 'team2_recent_form',
            'team1_batting_avg', 'team2_batting_avg',
            'team1_bowling_avg', 'team2_bowling_avg',
            'pitch_type', 'venue_id', 'is_day_night',
            'toss_winner', 'toss_decision'
        ]
        
        # One-hot encode categorical features
        data_encoded = pd.get_dummies(data, columns=['pitch_type', 'venue_id', 'toss_decision'])
        
        # Create binary feature for toss winner (1 if team1 won toss, 0 if team2)
        data_encoded['toss_winner_team1'] = data_encoded['toss_winner'].apply(
            lambda x: 1 if x == data_encoded['team1_id'] else 0
        )
        
        # Get final feature set
        X = data_encoded.drop(['match_id', 'team1_id', 'team2_id', 'winner', 'toss_winner'], axis=1)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        
        if 'winner' in data.columns:
            # For training: return features and target
            y = data['winner'].apply(lambda x: 1 if x == data['team1_id'] else 0)
            return X_scaled, y
        else:
            # For prediction: return features only
            return X_scaled
    
    def train(self, data, tune_hyperparams=False):
        """
        Train the match prediction model.
        
        Args:
            data (pd.DataFrame): Training data with match statistics
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
            
        Returns:
            float: Model accuracy on test set
        """
        X, y = self.preprocess_data(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if tune_hyperparams:
            # Hyperparameter tuning with GridSearchCV
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            # Use default model
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict(self, match_data):
        """
        Predict the winner of a match.
        
        Args:
            match_data (pd.DataFrame): Match data with team statistics
            
        Returns:
            dict: Prediction results with winner and probabilities
        """
        X = self.preprocess_data(match_data)
        
        # Get prediction and probabilities
        winner_pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        team1_id = match_data['team1_id'].values[0]
        team2_id = match_data['team2_id'].values[0]
        
        # Determine winner
        winner_id = team1_id if winner_pred == 1 else team2_id
        
        # Create result dictionary
        result = {
            'predicted_winner': winner_id,
            'team1_win_probability': float(proba[1]),
            'team2_win_probability': float(proba[0]),
            'confidence': float(max(proba))
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
        'team1_id': np.random.randint(1, 9, 100),
        'team2_id': np.random.randint(1, 9, 100),
        'team1_win_rate': np.random.uniform(0.4, 0.7, 100),
        'team2_win_rate': np.random.uniform(0.4, 0.7, 100),
        'team1_home_advantage': np.random.uniform(0, 1, 100),
        'team2_home_advantage': np.random.uniform(0, 1, 100),
        'team1_recent_form': np.random.uniform(0, 1, 100),
        'team2_recent_form': np.random.uniform(0, 1, 100),
        'team1_batting_avg': np.random.uniform(130, 180, 100),
        'team2_batting_avg': np.random.uniform(130, 180, 100),
        'team1_bowling_avg': np.random.uniform(7, 9, 100),
        'team2_bowling_avg': np.random.uniform(7, 9, 100),
        'pitch_type': np.random.choice(['batting', 'bowling', 'balanced'], 100),
        'venue_id': np.random.randint(1, 10, 100),
        'is_day_night': np.random.choice([0, 1], 100),
        'toss_winner': np.random.randint(1, 9, 100),
        'toss_decision': np.random.choice(['bat', 'field'], 100),
        'winner': np.random.randint(1, 9, 100)
    })
    
    # Initialize and train model
    predictor = MatchPredictor()
    predictor.train(sample_data)
    
    # Make a prediction
    new_match = sample_data.iloc[[0]].copy()
    new_match = new_match.drop('winner', axis=1)
    prediction = predictor.predict(new_match)
    
    print("\nPrediction for new match:")
    print(f"Team 1 ID: {new_match['team1_id'].values[0]}")
    print(f"Team 2 ID: {new_match['team2_id'].values[0]}")
    print(f"Predicted winner: {prediction['predicted_winner']}")
    print(f"Team 1 win probability: {prediction['team1_win_probability']:.4f}")
    print(f"Team 2 win probability: {prediction['team2_win_probability']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    
    # Save model
    predictor.save_model('match_predictor.joblib')
