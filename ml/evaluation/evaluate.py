import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
import joblib
import sys
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to import from sibling modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.match_predictor import MatchPredictor
from models.score_predictor import ScorePredictor
from models.player_predictor import PlayerPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_data():
    """
    Load test data for model evaluation.
    In a real implementation, this would load from a database or files.
    Here we generate synthetic data for demonstration.
    
    Returns:
        tuple: DataFrames for match, score, and player prediction testing
    """
    logger.info("Loading test data for evaluation...")
    
    # Generate synthetic match test data
    match_test_data = pd.DataFrame({
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
    
    # Ensure winner is either team1 or team2
    for i, row in match_test_data.iterrows():
        if row['winner'] not in [row['team1_id'], row['team2_id']]:
            match_test_data.at[i, 'winner'] = np.random.choice([row['team1_id'], row['team2_id']])
    
    # Generate synthetic score test data
    score_test_data = pd.DataFrame({
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
    
    # Generate synthetic batting test data
    batting_test_data = pd.DataFrame({
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
    
    # Generate synthetic bowling test data
    bowling_test_data = pd.DataFrame({
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
    
    logger.info(f"Loaded {len(match_test_data)} match test records, {len(score_test_data)} score test records, {len(batting_test_data)} batting test records, and {len(bowling_test_data)} bowling test records")
    
    return match_test_data, score_test_data, batting_test_data, bowling_test_data

def evaluate_match_predictor(test_data):
    """
    Evaluate the match prediction model.
    
    Args:
        test_data (pd.DataFrame): Test data for match prediction
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating match prediction model...")
    
    # Load model
    match_predictor = MatchPredictor()
    match_predictor.load_model('models/trained/match_predictor.joblib')
    
    # Prepare data
    X_test = test_data.drop(['match_id', 'winner'], axis=1)
    y_test = test_data['winner'].apply(lambda x: 1 if x == test_data['team1_id'].values[0] else 0)
    
    # Make predictions
    y_pred = []
    for i in range(len(test_data)):
        match_data = test_data.iloc[[i]].copy()
        prediction = match_predictor.predict(match_data)
        pred_winner = 1 if prediction['predicted_winner'] == match_data['team1_id'].values[0] else 0
        y_pred.append(pred_winner)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"Match prediction accuracy: {accuracy:.4f}")
    logger.info(f"Match prediction precision: {precision:.4f}")
    logger.info(f"Match prediction recall: {recall:.4f}")
    logger.info(f"Match prediction F1 score: {f1:.4f}")
    
    # Create evaluation report
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics

def evaluate_score_predictor(test_data):
    """
    Evaluate the score prediction model.
    
    Args:
        test_data (pd.DataFrame): Test data for score prediction
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating score prediction model...")
    
    # Load model
    score_predictor = ScorePredictor()
    score_predictor.load_model('models/trained/score_predictor.joblib')
    
    # Prepare data
    X_test = test_data.drop(['match_id', 'final_score'], axis=1)
    y_test = test_data['final_score']
    
    # Make predictions
    y_pred = []
    for i in range(len(test_data)):
        match_data = test_data.iloc[[i]].copy()
        prediction = score_predictor.predict(match_data)
        y_pred.append(prediction['predicted_score'])
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Score prediction MAE: {mae:.2f} runs")
    logger.info(f"Score prediction RMSE: {rmse:.2f} runs")
    logger.info(f"Score prediction R²: {r2:.4f}")
    
    # Create evaluation report
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics

def evaluate_player_predictor(test_data, performance_type):
    """
    Evaluate the player performance prediction model.
    
    Args:
        test_data (pd.DataFrame): Test data for player prediction
        performance_type (str): Type of performance ('batting' or 'bowling')
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info(f"Evaluating {performance_type} prediction model...")
    
    # Load model
    player_predictor = PlayerPredictor(performance_type=performance_type)
    model_path = f'models/trained/{performance_type}_predictor.joblib'
    player_predictor.load_model(model_path)
    
    # Prepare data
    target_col = 'runs_scored' if performance_type == 'batting' else 'wickets_taken'
    X_test = test_data.drop(['match_id', target_col], axis=1)
    y_test = test_data[target_col]
    
    # Make predictions
    y_pred = []
    for i in range(len(test_data)):
        player_data = test_data.iloc[[i]].copy()
        prediction = player_predictor.predict(player_data)
        pred_value = prediction['predicted_runs'] if performance_type == 'batting' else prediction['predicted_wickets']
        y_pred.append(pred_value)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metric = "runs" if performance_type == 'batting' else "wickets"
    logger.info(f"{performance_type.capitalize()} prediction MAE: {mae:.2f} {metric}")
    logger.info(f"{performance_type.capitalize()} prediction RMSE: {rmse:.2f} {metric}")
    logger.info(f"{performance_type.capitalize()} prediction R²: {r2:.4f}")
    
    # Create evaluation report
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics

def generate_evaluation_report(match_metrics, score_metrics, batting_metrics, bowling_metrics):
    """
    Generate a comprehensive evaluation report for all models.
      bowling_metrics):
    """
    Generate a comprehensive evaluation report for all models.
    
    Args:
        match_metrics (dict): Evaluation metrics for match prediction
        score_metrics (dict): Evaluation metrics for score prediction
        batting_metrics (dict): Evaluation metrics for batting prediction
        bowling_metrics (dict): Evaluation metrics for bowling prediction
        
    Returns:
        str: Path to the saved report
    """
    logger.info("Generating evaluation report...")
    
    # Create output directory if it doesn't exist
    os.makedirs('evaluation/reports', exist_ok=True)
    
    # Generate timestamp for report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f'evaluation/reports/model_evaluation_{timestamp}.html'
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>IPL Prediction Models Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #0066cc; }}
            h2 {{ color: #0099cc; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric-good {{ color: green; }}
            .metric-average {{ color: orange; }}
            .metric-poor {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>IPL Prediction Models Evaluation Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Match Prediction Model</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Accuracy</td>
                <td>{match_metrics['accuracy']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if match_metrics['accuracy'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if match_metrics['accuracy'] >= 0.6 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>Precision</td>
                <td>{match_metrics['precision']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if match_metrics['precision'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if match_metrics['precision'] >= 0.6 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>Recall</td>
                <td>{match_metrics['recall']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if match_metrics['recall'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if match_metrics['recall'] >= 0.6 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>F1 Score</td>
                <td>{match_metrics['f1']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if match_metrics['f1'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if match_metrics['f1'] >= 0.6 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
        </table>
        
        <h2>Score Prediction Model</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Mean Absolute Error (MAE)</td>
                <td>{score_metrics['mae']:.2f} runs</td>
                <td>
                    {'<span class="metric-good">Good</span>' if score_metrics['mae'] <= 10 else
                     '<span class="metric-average">Average</span>' if score_metrics['mae'] <= 15 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>Root Mean Squared Error (RMSE)</td>
                <td>{score_metrics['rmse']:.2f} runs</td>
                <td>
                    {'<span class="metric-good">Good</span>' if score_metrics['rmse'] <= 15 else
                     '<span class="metric-average">Average</span>' if score_metrics['rmse'] <= 20 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>R² Score</td>
                <td>{score_metrics['r2']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if score_metrics['r2'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if score_metrics['r2'] >= 0.5 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
        </table>
        
        <h2>Batting Performance Prediction Model</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Mean Absolute Error (MAE)</td>
                <td>{batting_metrics['mae']:.2f} runs</td>
                <td>
                    {'<span class="metric-good">Good</span>' if batting_metrics['mae'] <= 10 else
                     '<span class="metric-average">Average</span>' if batting_metrics['mae'] <= 15 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>Root Mean Squared Error (RMSE)</td>
                <td>{batting_metrics['rmse']:.2f} runs</td>
                <td>
                    {'<span class="metric-good">Good</span>' if batting_metrics['rmse'] <= 15 else
                     '<span class="metric-average">Average</span>' if batting_metrics['rmse'] <= 20 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>R² Score</td>
                <td>{batting_metrics['r2']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if batting_metrics['r2'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if batting_metrics['r2'] >= 0.5 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
        </table>
        
        <h2>Bowling Performance Prediction Model</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td>Mean Absolute Error (MAE)</td>
                <td>{bowling_metrics['mae']:.2f} wickets</td>
                <td>
                    {'<span class="metric-good">Good</span>' if bowling_metrics['mae'] <= 0.8 else
                     '<span class="metric-average">Average</span>' if bowling_metrics['mae'] <= 1.2 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>Root Mean Squared Error (RMSE)</td>
                <td>{bowling_metrics['rmse']:.2f} wickets</td>
                <td>
                    {'<span class="metric-good">Good</span>' if bowling_metrics['rmse'] <= 1.0 else
                     '<span class="metric-average">Average</span>' if bowling_metrics['rmse'] <= 1.5 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
            <tr>
                <td>R² Score</td>
                <td>{bowling_metrics['r2']:.4f}</td>
                <td>
                    {'<span class="metric-good">Good</span>' if bowling_metrics['r2'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if bowling_metrics['r2'] >= 0.5 else
                     '<span class="metric-poor">Poor</span>'}
                </td>
            </tr>
        </table>
        
        <h2>Summary</h2>
        <p>
            Overall model performance assessment:
            <ul>
                <li>Match Prediction: 
                    {'<span class="metric-good">Good</span>' if match_metrics['accuracy'] >= 0.7 else
                     '<span class="metric-average">Average</span>' if match_metrics['accuracy'] >= 0.6 else
                     '<span class="metric-poor">Poor</span>'}
                </li>
                <li>Score Prediction: 
                    {'<span class="metric-good">Good</span>' if score_metrics['mae'] <= 10 else
                     '<span class="metric-average">Average</span>' if score_metrics['mae'] <= 15 else
                     '<span class="metric-poor">Poor</span>'}
                </li>
                <li>Batting Performance Prediction: 
                    {'<span class="metric-good">Good</span>' if batting_metrics['mae'] <= 10 else
                     '<span class="metric-average">Average</span>' if batting_metrics['mae'] <= 15 else
                     '<span class="metric-poor">Poor</span>'}
                </li>
                <li>Bowling Performance Prediction: 
                    {'<span class="metric-good">Good</span>' if bowling_metrics['mae'] <= 0.8 else
                     '<span class="metric-average">Average</span>' if bowling_metrics['mae'] <= 1.2 else
                     '<span class="metric-poor">Poor</span>'}
                </li>
            </ul>
        </p>
        
        <p>
            Recommendations for improvement:
            <ul>
                <li>Consider collecting more historical data for better training</li>
                <li>Experiment with different feature engineering approaches</li>
                <li>Try ensemble methods to combine multiple models</li>
                <li>Incorporate more contextual features like player form and head-to-head records</li>
            </ul>
        </p>
    </body>
    </html>
    """
    
    # Save report to file
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Evaluation report saved to {report_path}")
    
    return report_path

def main():
    """Main function to evaluate all prediction models."""
    logger.info("Starting model evaluation pipeline...")
    
    # Load test data
    match_test_data, score_test_data, batting_test_data, bowling_test_data = load_test_data()
    
    # Evaluate models
    match_metrics = evaluate_match_predictor(match_test_data)
    score_metrics = evaluate_score_predictor(score_test_data)
    batting_metrics = evaluate_player_predictor(batting_test_data, 'batting')
    bowling_metrics = evaluate_player_predictor(bowling_test_data, 'bowling')
    
    # Generate evaluation report
    report_path = generate_evaluation_report(
        match_metrics, score_metrics, batting_metrics, bowling_metrics
    )
    
    logger.info(f"Model evaluation pipeline completed successfully. Report saved to {report_path}")

if __name__ == "__main__":
    main()
