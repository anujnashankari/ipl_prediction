import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import sys
import logging

# Add parent directory to path to import from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.match_predictor import MatchPredictor
from models.score_predictor import ScorePredictor
from models.player_predictor import PlayerPredictor
from features.feature_engineering import FeatureEngineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load and prepare data for training.
    In a real implementation, this would load from a database or files.
    Here we generate synthetic data for demonstration.
    
    Returns:
        tuple: DataFrames for matches, teams, players, and performances
    """
    logger.info("Loading data for training...")
    
    # Generate synthetic match data
    matches_df = pd.DataFrame({
        'match_id': range(500),
        'team1_id': np.random.randint(1, 9, 500),
        'team2_id': np.random.randint(1, 9, 500),
        'venue_id': np.random.randint(1, 10, 500),
        'venue_team_id': np.random.randint(1, 9, 500),
        'winner': np.random.randint(1, 9, 500),
        'score': np.random.randint(120, 220, 500),
        'economy': np.random.uniform(7, 10, 500),
        'match_date': pd.date_range(start='2020-01-01', periods=500)
    })
    
    # Ensure winner is either team1 or team2
    for i, row in matches_df.iterrows():
        if row['winner'] not in [row['team1_id'], row['team2_id']]:
            matches_df.at[i, 'winner'] = np.random.choice([row['team1_id'], row['team2_id']])
    
    # Generate team data
    teams_df = pd.DataFrame({
        'team_id': range(1, 9),
        'team_name': [f'Team {i}' for i in range(1, 9)]
    })
    
    # Generate player data
    players_df = pd.DataFrame({
        'player_id': range(1, 101),
        'player_name': [f'Player {i}' for i in range(1, 101)],
        'team_id': np.random.randint(1, 9, 100)
    })
    
    # Generate player performance data
    batting_performances = []
    bowling_performances = []
    
    for match_id in range(500):
        # Get teams for this match
        team1_id = matches_df.loc[match_id, 'team1_id']
        team2_id = matches_df.loc[match_id, 'team2_id']
        
        # Get players from each team
        team1_players = players_df[players_df['team_id'] == team1_id]['player_id'].values
        team2_players = players_df[players_df['team_id'] == team2_id]['player_id'].values
        
        # Generate batting performances for team 1
        for player_id in np.random.choice(team1_players, size=min(11, len(team1_players)), replace=False):
            batting_performances.append({
                'match_id': match_id,
                'player_id': player_id,
                'innings_type': 'batting',
                'runs': np.random.randint(0, 100),
                'balls_faced': np.random.randint(1, 60),
                'strike_rate': np.random.uniform(100, 200),
                'match_date': matches_df.loc[match_id, 'match_date']
            })
        
        # Generate bowling performances for team 2
        for player_id in np.random.choice(team2_players, size=min(6, len(team2_players)), replace=False):
            bowling_performances.append({
                'match_id': match_id,
                'player_id': player_id,
                'innings_type': 'bowling',
                'wickets': np.random.randint(0, 5),
                'overs': np.random.uniform(1, 4),
                'economy_rate': np.random.uniform(6, 12),
                'bowling_average': np.random.uniform(20, 40),
                'match_date': matches_df.loc[match_id, 'match_date']
            })
        
        # Generate batting performances for team 2
        for player_id in np.random.choice(team2_players, size=min(11, len(team2_players)), replace=False):
            batting_performances.append({
                'match_id': match_id,
                'player_id': player_id,
                'innings_type': 'batting',
                'runs': np.random.randint(0, 100),
                'balls_faced': np.random.randint(1, 60),
                'strike_rate': np.random.uniform(100, 200),
                'match_date': matches_df.loc[match_id, 'match_date']
            })
        
        # Generate bowling performances for team 1
        for player_id in np.random.choice(team1_players, size=min(6, len(team1_players)), replace=False):
            bowling_performances.append({
                'match_id': match_id,
                'player_id': player_id,
                'innings_type': 'bowling',
                'wickets': np.random.randint(0, 5),
                'overs': np.random.uniform(1, 4),
                'economy_rate': np.random.uniform(6, 12),
                'bowling_average': np.random.uniform(20, 40),
                'match_date': matches_df.loc[match_id, 'match_date']
            })
    
    # Combine performances
    performances_df = pd.DataFrame(batting_performances + bowling_performances)
    
    logger.info(f"Loaded {len(matches_df)} matches, {len(teams_df)} teams, {len(players_df)} players, and {len(performances_df)} performances")
    
    return matches_df, teams_df, players_df, performances_df

def prepare_training_data(matches_df, teams_df, players_df, performances_df):
    """
    Prepare data for training the prediction models.
    
    Args:
        matches_df (pd.DataFrame): Match data
        teams_df (pd.DataFrame): Team data
        players_df (pd.DataFrame): Player data
        performances_df (pd.DataFrame): Performance data
        
    Returns:
        tuple: DataFrames for match prediction, score prediction, and player prediction
    """
    logger.info("Preparing training data...")
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Calculate team statistics
    team_stats = fe.calculate_team_stats(matches_df, teams_df)
    
    # Calculate player statistics
    player_stats = fe.calculate_player_stats(players_df, performances_df)
    
    # Calculate recent form
    team_form = fe.calculate_recent_form(teams_df, matches_df, 'team')
    player_form = fe.calculate_recent_form(players_df, performances_df, 'player')
    
    # Merge team stats with form
    team_stats = pd.merge(team_stats, team_form, on='team_id')
    
    # Merge player stats with form
    player_stats = pd.merge(player_stats, player_form, on='player_id')
    
    # Create venue statistics
    venue_stats = pd.DataFrame({
        'venue_id': matches_df['venue_id'].unique(),
        'avg_score': [matches_df[matches_df['venue_id'] == v]['score'].mean() for v in matches_df['venue_id'].unique()],
        'avg_wickets': [np.random.uniform(6, 8) for _ in matches_df['venue_id'].unique()]
    })
    
    # Prepare data for match prediction
    match_prediction_data = []
    
    for _, match in matches_df.iterrows():
        match_data = {
            'match_id': match['match_id'],
            'team1_id': match['team1_id'],
            'team2_id': match['team2_id'],
            'venue_id': match['venue_id'],
            'toss_winner': match['team1_id'] if np.random.random() > 0.5 else match['team2_id'],
            'toss_decision': np.random.choice(['bat', 'field']),
            'winner': match['winner']
        }
        
        # Add features
        match_features = fe.prepare_match_features(match_data, team_stats, venue_stats)
        match_prediction_data.append({**match_data, **match_features.iloc[0].to_dict()})
    
    match_prediction_df = pd.DataFrame(match_prediction_data)
    
    # Prepare data for score prediction
    score_prediction_data = []
    
    for _, match in matches_df.iterrows():
        # Create two records - one for each team batting
        for batting_team, bowling_team in [(match['team1_id'], match['team2_id']), (match['team2_id'], match['team1_id'])]:
            score_data = {
                'match_id': match['match_id'],
                'batting_team_id': batting_team,
                'bowling_team_id': bowling_team,
                'venue_id': match['venue_id'],
                'is_first_innings': np.random.choice([0, 1]),
                'is_day_night': np.random.choice([0, 1]),
                'pitch_type': np.random.choice(['batting', 'bowling', 'balanced']),
                'final_score': np.random.randint(120, 220)
            }
            
            # Add features
            score_features = fe.prepare_score_features(score_data, team_stats, venue_stats)
            score_prediction_data.append({**score_data, **score_features.iloc[0].to_dict()})
    
    score_prediction_df = pd.DataFrame(score_prediction_data)
    
    # Prepare data for player prediction
    player_prediction_data = []
    
    # Batting predictions
    batting_performances = performances_df[performances_df['innings_type'] == 'batting']
    for _, perf in batting_performances.iterrows():
        player_id = perf['player_id']
        match_id = perf['match_id']
        
        # Get match details
        match = matches_df[matches_df['match_id'] == match_id].iloc[0]
        
        # Get player's team
        player_team = players_df[players_df['player_id'] == player_id]['team_id'].values[0]
        
        # Get opposition team
        opposition_id = match['team2_id'] if player_team == match['team1_id'] else match['team1_id']
        
        player_data = {
            'match_id': match_id,
            'player_id': player_id,
            'opposition_id': opposition_id,
            'venue_id': match['venue_id'],
            'is_home_ground': 1 if match['venue_team_id'] == player_team else 0,
            'pitch_type': np.random.choice(['batting', 'bowling', 'balanced']),
            'is_day_night': np.random.choice([0, 1]),
            'batting_position': np.random.choice(['opener', 'top_order', 'middle_order', 'lower_order']),
            'runs_scored': perf['runs']
        }
        
        # Add features
        player_features = fe.prepare_player_features(player_data, player_stats, team_stats, 'batting')
        player_prediction_data.append({**player_data, **player_features.iloc[0].to_dict()})
    
    # Bowling predictions
    bowling_performances = performances_df[performances_df['innings_type'] == 'bowling']
    for _, perf in bowling_performances.iterrows():
        player_id = perf['player_id']
        match_id = perf['match_id']
        
        # Get match details
        match = matches_df[matches_df['match_id'] == match_id].iloc[0]
        
        # Get player's team
        player_team = players_df[players_df['player_id'] == player_id]['team_id'].values[0]
        
        # Get opposition team
        opposition_id = match['team2_id'] if player_team == match['team1_id'] else match['team1_id']
        
        player_data = {
            'match_id': match_id,
            'player_id': player_id,
            'opposition_id': opposition_id,
            'venue_id': match['venue_id'],
            'is_home_ground': 1 if match['venue_team_id'] == player_team else 0,
            'pitch_type': np.random.choice(['batting', 'bowling', 'balanced']),
            'is_day_night': np.random.choice([0, 1]),
            'bowling_style': np.random.choice(['fast', 'medium', 'spin']),
            'wickets_taken': perf['wickets']
        }
        
        # Add features
        player_features = fe.prepare_player_features(player_data, player_stats, team_stats, 'bowling')
        player_prediction_data.append({**player_data, **player_features.iloc[0].to_dict()})
    
    player_prediction_df = pd.DataFrame(player_prediction_data)
    
    logger.info(f"Prepared {len(match_prediction_df)} match records, {len(score_prediction_df)} score records, and {len(player_prediction_df)} player records for training")
    
    return match_prediction_df, score_prediction_df, player_prediction_df

def train_models(match_data, score_data, player_data):
    """
    Train and save all prediction models.
    
    Args:
        match_data (pd.DataFrame): Data for match prediction
        score_data (pd.DataFrame): Data for score prediction
        player_data (pd.DataFrame): Data for player performance prediction
    """
    logger.info("Training prediction models...")
    
    # Create output directory if it doesn't exist
    os.makedirs('models/trained', exist_ok=True)
    
    # Train match prediction model
    logger.info("Training match prediction model...")
    match_predictor = MatchPredictor()
    match_predictor.train(match_data)
    match_predictor.save_model('models/trained/match_predictor.joblib')
    
    # Train score prediction model
    logger.info("Training score prediction model...")
    score_predictor = ScorePredictor()
    score_predictor.train(score_data)
    score_predictor.save_model('models/trained/score_predictor.joblib')
    
    # Train batting prediction model
    logger.info("Training batting prediction model...")
    batting_data = player_data[player_data['runs_scored'].notna()]
    batting_predictor = PlayerPredictor(performance_type='batting')
    batting_predictor.train(batting_data)
    batting_predictor.save_model('models/trained/batting_predictor.joblib')
    
    # Train bowling prediction model
    logger.info("Training bowling prediction model...")
    bowling_data = player_data[player_data['wickets_taken'].notna()]
    bowling_predictor = PlayerPredictor(performance_type='bowling')
    bowling_predictor.train(bowling_data)
    bowling_predictor.save_model('models/trained/bowling_predictor.joblib')
    
    logger.info("All models trained and saved successfully")

def main():
    """Main function to train all prediction models."""
    logger.info("Starting model training pipeline...")
    
    # Load data
    matches_df, teams_df, players_df, performances_df = load_data()
    
    # Prepare training data
    match_data, score_data, player_data = prepare_training_data(
        matches_df, teams_df, players_df, performances_df
    )
    
    # Train models
    train_models(match_data, score_data, player_data)
    
    logger.info("Model training pipeline completed successfully")

if __name__ == "__main__":
    main()
