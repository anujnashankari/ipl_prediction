import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta

class FeatureEngineering:
    """
    Class for engineering features from raw IPL data for prediction models.
    """
    
    def __init__(self):
        """Initialize the feature engineering pipeline."""
        pass
    
    def calculate_team_stats(self, matches_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate team statistics from historical match data.
        
        Args:
            matches_df (pd.DataFrame): DataFrame with match data
            teams_df (pd.DataFrame): DataFrame with team information
            
        Returns:
            pd.DataFrame: DataFrame with team statistics
        """
        team_stats = pd.DataFrame()
        team_stats['team_id'] = teams_df['team_id']
        
        # Calculate win rates
        wins = matches_df.groupby('winner').size()
        total_matches = pd.concat([
            matches_df.groupby('team1_id').size(),
            matches_df.groupby('team2_id').size()
        ]).groupby(level=0).sum()
        
        team_stats['win_rate'] = team_stats['team_id'].apply(
            lambda x: wins.get(x, 0) / total_matches.get(x, 1) if x in total_matches else 0
        )
        
        # Calculate home advantage
        home_wins = matches_df[matches_df['venue_team_id'] == matches_df['winner']].groupby('winner').size()
        home_matches = matches_df[matches_df['venue_team_id'] == matches_df['team1_id']].groupby('team1_id').size() + \
                      matches_df[matches_df['venue_team_id'] == matches_df['team2_id']].groupby('team2_id').size()
        
        team_stats['home_advantage'] = team_stats['team_id'].apply(
            lambda x: home_wins.get(x, 0) / home_matches.get(x, 1) if x in home_matches else 0
        )
        
        # Calculate batting statistics
        team_stats['avg_score'] = team_stats['team_id'].apply(
            lambda x: matches_df[(matches_df['team1_id'] == x) | (matches_df['team2_id'] == x)]['score'].mean()
        )
        
        # Calculate bowling statistics
        team_stats['avg_economy'] = team_stats['team_id'].apply(
            lambda x: matches_df[(matches_df['team1_id'] == x) | (matches_df['team2_id'] == x)]['economy'].mean()
        )
        
        return team_stats
    
    def calculate_player_stats(self, players_df: pd.DataFrame, performances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate player statistics from historical performance data.
        
        Args:
            players_df (pd.DataFrame): DataFrame with player information
            performances_df (pd.DataFrame): DataFrame with player performances
            
        Returns:
            pd.DataFrame: DataFrame with player statistics
        """
        player_stats = pd.DataFrame()
        player_stats['player_id'] = players_df['player_id']
        
        # Calculate batting statistics
        batting_performances = performances_df[performances_df['innings_type'] == 'batting']
        
        player_stats['batting_avg'] = player_stats['player_id'].apply(
            lambda x: batting_performances[batting_performances['player_id'] == x]['runs'].mean()
        )
        
        player_stats['strike_rate'] = player_stats['player_id'].apply(
            lambda x: batting_performances[batting_performances['player_id'] == x]['strike_rate'].mean()
        )
        
        # Calculate bowling statistics
        bowling_performances = performances_df[performances_df['innings_type'] == 'bowling']
        
        player_stats['bowling_avg'] = player_stats['player_id'].apply(
            lambda x: bowling_performances[bowling_performances['player_id'] == x]['bowling_average'].mean()
        )
        
        player_stats['economy_rate'] = player_stats['player_id'].apply(
            lambda x: bowling_performances[bowling_performances['player_id'] == x]['economy_rate'].mean()
        )
        
        # Calculate experience (number of matches played)
        player_stats['experience'] = player_stats['player_id'].apply(
            lambda x: len(performances_df[performances_df['player_id'] == x]['match_id'].unique())
        )
        
        return player_stats
    
    def calculate_recent_form(self, entity_df: pd.DataFrame, performances_df: pd.DataFrame, 
                             entity_type: str, n_matches: int = 5) -> pd.DataFrame:
        """
        Calculate recent form for teams or players based on last n matches.
        
        Args:
            entity_df (pd.DataFrame): DataFrame with team or player information
            performances_df (pd.DataFrame): DataFrame with performances
            entity_type (str): Type of entity ('team' or 'player')
            n_matches (int): Number of recent matches to consider
            
        Returns:
            pd.DataFrame: DataFrame with recent form metrics
        """
        recent_form = pd.DataFrame()
        id_col = f"{entity_type}_id"
        recent_form[id_col] = entity_df[id_col]
        
        if entity_type == 'team':
            # For teams, calculate win rate in recent matches
            recent_form['recent_form'] = recent_form[id_col].apply(
                lambda x: self._calculate_team_recent_form(x, performances_df, n_matches)
            )
        else:  # player
            # For batsmen, calculate average runs in recent matches
            recent_form['recent_batting_form'] = recent_form[id_col].apply(
                lambda x: self._calculate_player_recent_batting_form(x, performances_df, n_matches)
            )
            
            # For bowlers, calculate average economy in recent matches
            recent_form['recent_bowling_form'] = recent_form[id_col].apply(
                lambda x: self._calculate_player_recent_bowling_form(x, performances_df, n_matches)
            )
        
        return recent_form
    
    def _calculate_team_recent_form(self, team_id: int, matches_df: pd.DataFrame, n_matches: int) -> float:
        """Calculate team's recent form based on win rate in last n matches."""
        team_matches = matches_df[(matches_df['team1_id'] == team_id) | (matches_df['team2_id'] == team_id)]
        team_matches = team_matches.sort_values('match_date', ascending=False).head(n_matches)
        
        if len(team_matches) == 0:
            return 0.5  # Default value if no recent matches
        
        wins = sum(team_matches['winner'] == team_id)
        return wins / len(team_matches)
    
    def _calculate_player_recent_batting_form(self, player_id: int, performances_df: pd.DataFrame, n_matches: int) -> float:
        """Calculate player's recent batting form based on average runs in last n matches."""
        player_performances = performances_df[
            (performances_df['player_id'] == player_id) & 
            (performances_df['innings_type'] == 'batting')
        ]
        player_performances = player_performances.sort_values('match_date', ascending=False).head(n_matches)
        
        if len(player_performances) == 0:
            return 0  # Default value if no recent performances
        
        avg_runs = player_performances['runs'].mean()
        max_runs = player_performances['runs'].max()
        
        # Normalize to 0-1 scale (assuming 100 runs is excellent performance)
        return min(1.0, avg_runs / 50.0)
    
    def _calculate_player_recent_bowling_form(self, player_id: int, performances_df: pd.DataFrame, n_matches: int) -> float:
        """Calculate player's recent bowling form based on average economy in last n matches."""
        player_performances = performances_df[
            (performances_df['player_id'] == player_id) & 
            (performances_df['innings_type'] == 'bowling')
        ]
        player_performances = player_performances.sort_values('match_date', ascending=False).head(n_matches)
        
        if len(player_performances) == 0:
            return 0.5  # Default value if no recent performances
        
        avg_economy = player_performances['economy_rate'].mean()
        avg_wickets = player_performances['wickets'].mean()
        
        # Normalize to 0-1 scale (lower economy is better, higher wickets is better)
        economy_score = max(0, 1 - (avg_economy / 12.0))
        wicket_score = min(1.0, avg_wickets / 3.0)
        
        # Combine the two metrics (equal weighting)
        return (economy_score + wicket_score) / 2
    
    def prepare_match_features(self, match_data: Dict[str, Any], team_stats: pd.DataFrame, 
                              venue_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for match prediction.
        
        Args:
            match_data (dict): Data for the match to predict
            team_stats (pd.DataFrame): Team statistics
            venue_stats (pd.DataFrame): Venue statistics
            
        Returns:
            pd.DataFrame: DataFrame with features for prediction
        """
        features = pd.DataFrame([match_data])
        
        # Add team statistics
        team1_id = match_data['team1_id']
        team2_id = match_data['team2_id']
        
        team1_stats = team_stats[team_stats['team_id'] == team1_id].iloc[0]
        team2_stats = team_stats[team_stats['team_id'] == team2_id].iloc[0]
        
        features['team1_win_rate'] = team1_stats['win_rate']
        features['team2_win_rate'] = team2_stats['win_rate']
        features['team1_home_advantage'] = team1_stats['home_advantage']
        features['team2_home_advantage'] = team2_stats['home_advantage']
        features['team1_batting_avg'] = team1_stats['avg_score']
        features['team2_batting_avg'] = team2_stats['avg_score']
        features['team1_bowling_avg'] = team1_stats['avg_economy']
        features['team2_bowling_avg'] = team2_stats['avg_economy']
        
        # Add venue statistics
        venue_id = match_data['venue_id']
        venue_row = venue_stats[venue_stats['venue_id'] == venue_id]
        
        if not venue_row.empty:
            features['venue_avg_score'] = venue_row.iloc[0]['avg_score']
            features['venue_avg_wickets'] = venue_row.iloc[0]['avg_wickets']
        else:
            # Default values if venue not found
            features['venue_avg_score'] = 160
            features['venue_avg_wickets'] = 7
        
        return features
    
    def prepare_score_features(self, match_data: Dict[str, Any], team_stats: pd.DataFrame, 
                              venue_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for score prediction.
        
        Args:
            match_data (dict): Data for the match to predict
            team_stats (pd.DataFrame): Team statistics
            venue_stats (pd.DataFrame): Venue statistics
            
        Returns:
            pd.DataFrame: DataFrame with features for prediction
        """
        features = pd.DataFrame([match_data])
        
        # Add team statistics
        batting_team_id = match_data['batting_team_id']
        bowling_team_id = match_data['bowling_team_id']
        
        batting_team_stats = team_stats[team_stats['team_id'] == batting_team_id].iloc[0]
        bowling_team_stats = team_stats[team_stats['team_id'] == bowling_team_id].iloc[0]
        
        features['batting_team_avg_score'] = batting_team_stats['avg_score']
        features['batting_team_recent_form'] = batting_team_stats['recent_form']
        features['bowling_team_avg_economy'] = bowling_team_stats['avg_economy']
        features['bowling_team_recent_form'] = bowling_team_stats['recent_form']
        
        # Add venue statistics
        venue_id = match_data['venue_id']
        venue_row = venue_stats[venue_stats['venue_id'] == venue_id]
        
        if not venue_row.empty:
            features['venue_avg_score'] = venue_row.iloc[0]['avg_score']
        else:
            # Default value if venue not found
            features['venue_avg_score'] = 160
        
        return features
    
    def prepare_player_features(self, player_data: Dict[str, Any], player_stats: pd.DataFrame, 
                               team_stats: pd.DataFrame, performance_type: str) -> pd.DataFrame:
        """
        Prepare features for player performance prediction.
        
        Args:
            player_data (dict): Data for the player and match to predict
            player_stats (pd.DataFrame): Player statistics
            team_stats (pd.DataFrame): Team statistics
            performance_type (str): Type of performance to predict ('batting' or 'bowling')
            
        Returns:
            pd.DataFrame: DataFrame with features for prediction
        """
        features = pd.DataFrame([player_data])
        
        # Add player statistics
        player_id = player_data['player_id']
        player_row = player_stats[player_stats['player_id'] == player_id]
        
        if not player_row.empty:
            if performance_type == 'batting':
                features['player_batting_avg'] = player_row.iloc[0]['batting_avg']
                features['player_strike_rate'] = player_row.iloc[0]['strike_rate']
                features['player_recent_form'] = player_row.iloc[0]['recent_batting_form']
            else:  # bowling
                features['player_bowling_avg'] = player_row.iloc[0]['bowling_avg']
                features['player_economy_rate'] = player_row.iloc[0]['economy_rate']
                features['player_recent_form'] = player_row.iloc[0]['recent_bowling_form']
                
            features['player_experience'] = player_row.iloc[0]['experience']
        else:
            # Default values if player not found
            if performance_type == 'batting':
                features['player_batting_avg'] = 25
                features['player_strike_rate'] = 125
                features['player_recent_form'] = 0.5
            else:  # bowling
                features['player_bowling_avg'] = 30
                features['player_economy_rate'] = 8
                features['player_recent_form'] = 0.5
                
            features['player_experience'] = 20
        
        # Add opposition team statistics
        opposition_id = player_data['opposition_id']
        opposition_row = team_stats[team_stats['team_id'] == opposition_id]
        
        if not opposition_row.empty:
            if performance_type == 'batting':
                features['opposition_bowling_avg'] = opposition_row.iloc[0]['avg_economy']
            else:  # bowling
                features['opposition_batting_avg'] = opposition_row.iloc[0]['avg_score']
        else:
            # Default values if opposition not found
            if performance_type == 'batting':
                features['opposition_bowling_avg'] = 8
            else:  # bowling
                features['opposition_batting_avg'] = 160
        
        return features

# Example usage
if __name__ == "__main__":
    # Create sample data for demonstration
    matches_df = pd.DataFrame({
        'match_id': range(100),
        'team1_id': np.random.randint(1, 9, 100),
        'team2_id': np.random.randint(1, 9, 100),
        'venue_id': np.random.randint(1, 10, 100),
        'venue_team_id': np.random.randint(1, 9, 100),
        'winner': np.random.randint(1, 9, 100),
        'score': np.random.randint(120, 220, 100),
        'economy': np.random.uniform(7, 10, 100),
        'match_date': [datetime.now() - timedelta(days=i) for i in range(100)]
    })
    
    teams_df = pd.DataFrame({
        'team_id': range(1, 9),
        'team_name': [f'Team {i}' for i in range(1, 9)]
    })
    
    # Initialize feature engineering
    fe = FeatureEngineering()
    
    # Calculate team statistics
    team_stats = fe.calculate_team_stats(matches_df, teams_df)
    
    print("Team Statistics:")
    print(team_stats.head())
    
    # Prepare match features for prediction
    match_data = {
        'match_id': 101,
        'team1_id': 1,
        'team2_id': 2,
        'venue_id': 3,
        'toss_winner': 1,
        'toss_decision': 'bat'
    }
    
    # Create simple venue stats for demonstration
    venue_stats = pd.DataFrame({
        'venue_id': range(1, 10),
        'avg_score': np.random.uniform(150, 180, 9),
        'avg_wickets': np.random.uniform(6, 8, 9)
    })
    
    # Prepare features for match prediction
    match_features = fe.prepare_match_features(match_data, team_stats, venue_stats)
    
    print("\nMatch Features:")
    print(match_features.head())
