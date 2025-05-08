import os
import sys
import joblib
import pandas as pd
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from models import (
    Team, Player, Venue, Match, PlayerPerformance, Prediction, PlayerPrediction,
    TeamCreate, PlayerCreate, VenueCreate, MatchCreate, PlayerPerformanceCreate,
    TeamResponse, PlayerResponse, VenueResponse, MatchResponse, PlayerPerformanceResponse,
    PredictionResponse, PlayerPredictionResponse, MatchPredictionRequest, ScorePredictionRequest,
    PlayerPredictionRequest, get_db, engine, Base
)

# Create database tables
Base.metadata.create_all(bind=engine)

# Add ML module to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml'))

# Import ML models
from models.match_predictor import MatchPredictor
from models.score_predictor import ScorePredictor
from models.player_predictor import PlayerPredictor
from llm.reasoning import LLMReasoning

app = FastAPI(
    title="IPL Prediction API",
    description="API for IPL cricket match prediction system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Team endpoints
@app.post("/teams/", response_model=TeamResponse)
def create_team(team: TeamCreate, db: Session = Depends(get_db)):
    db_team = Team(**team.dict())
    db.add(db_team)
    db.commit()
    db.refresh(db_team)
    return db_team

@app.get("/teams/", response_model=List[TeamResponse])
def read_teams(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    teams = db.query(Team).offset(skip).limit(limit).all()
    return teams

@app.get("/teams/{team_id}", response_model=TeamResponse)
def read_team(team_id: int, db: Session = Depends(get_db)):
    team = db.query(Team).filter(Team.id == team_id).first()
    if team is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return team

# Player endpoints
@app.post("/players/", response_model=PlayerResponse)
def create_player(player: PlayerCreate, db: Session = Depends(get_db)):
    db_player = Player(**player.dict())
    db.add(db_player)
    db.commit()
    db.refresh(db_player)
    return db_player

@app.get("/players/", response_model=List[PlayerResponse])
def read_players(
    skip: int = 0, 
    limit: int = 100, 
    team_id: Optional[int] = None,
    role: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Player)
    if team_id:
        query = query.filter(Player.team_id == team_id)
    if role:
        query = query.filter(Player.role == role)
    players = query.offset(skip).limit(limit).all()
    return players

@app.get("/players/{player_id}", response_model=PlayerResponse)
def read_player(player_id: int, db: Session = Depends(get_db)):
    player = db.query(Player).filter(Player.id == player_id).first()
    if player is None:
        raise HTTPException(status_code=404, detail="Player not found")
    return player

# Venue endpoints
@app.post("/venues/", response_model=VenueResponse)
def create_venue(venue: VenueCreate, db: Session = Depends(get_db)):
    db_venue = Venue(**venue.dict())
    db.add(db_venue)
    db.commit()
    db.refresh(db_venue)
    return db_venue

@app.get("/venues/", response_model=List[VenueResponse])
def read_venues(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    venues = db.query(Venue).offset(skip).limit(limit).all()
    return venues

@app.get("/venues/{venue_id}", response_model=VenueResponse)
def read_venue(venue_id: int, db: Session = Depends(get_db)):
    venue = db.query(Venue).filter(Venue.id == venue_id).first()
    if venue is None:
        raise HTTPException(status_code=404, detail="Venue not found")
    return venue

# Match endpoints
@app.post("/matches/", response_model=MatchResponse)
def create_match(match: MatchCreate, db: Session = Depends(get_db)):
    db_match = Match(**match.dict())
    db.add(db_match)
    db.commit()
    db.refresh(db_match)
    return db_match

@app.get("/matches/", response_model=List[MatchResponse])
def read_matches(
    skip: int = 0, 
    limit: int = 100, 
    team_id: Optional[int] = None,
    venue_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    query = db.query(Match)
    if team_id:
        query = query.filter((Match.team1_id == team_id) | (Match.team2_id == team_id))
    if venue_id:
        query = query.filter(Match.venue_id == venue_id)
    matches = query.offset(skip).limit(limit).all()
    return matches

@app.get("/matches/{match_id}", response_model=MatchResponse)
def read_match(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter(Match.id == match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    return match

# Player Performance endpoints
@app.post("/performances/", response_model=PlayerPerformanceResponse)
def create_performance(performance: PlayerPerformanceCreate, db: Session = Depends(get_db)):
    db_performance = PlayerPerformance(**performance.dict())
    db.add(db_performance)
    db.commit()
    db.refresh(db_performance)
    return db_performance

@app.get("/performances/", response_model=List[PlayerPerformanceResponse])
def read_performances(
    skip: int = 0, 
    limit: int = 100, 
    player_id: Optional[int] = None,
    match_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    query = db.query(PlayerPerformance)
    if player_id:
        query = query.filter(PlayerPerformance.player_id == player_id)
    if match_id:
        query = query.filter(PlayerPerformance.match_id == match_id)
    performances = query.offset(skip).limit(limit).all()
    return performances

# Prediction endpoints
@app.post("/predictions/match/", response_model=PredictionResponse)
def predict_match(request: MatchPredictionRequest, db: Session = Depends(get_db)):
    # Check if match exists
    match = db.query(Match).filter(Match.id == request.match_id).first()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found")
    
    # Check if prediction already exists
    existing_prediction = db.query(Prediction).filter(Prediction.match_id == request.match_id).first()
    if existing_prediction:
        return existing_prediction
    
    # Prepare data for prediction
    match_data = {
        'match_id': match.id,
        'team1_id': match.team1_id,
        'team2_id': match.team2_id,
        'venue_id': match.venue_id,
        'is_day_night': match.is_day_night,
        'pitch_type': match.pitch_type
    }
    
    # Get team statistics
    team1 = db.query(Team).filter(Team.id == match.team1_id).first()
    team2 = db.query(Team).filter(Team.id == match.team2_id).first()
    
    match_data.update({
        'team1_win_rate': team1.win_rate,
        'team2_win_rate': team2.win_rate,
        'team1_home_advantage': team1.home_advantage,
        'team2_home_advantage': team2.home_advantage,
        'team1_recent_form': team1.recent_form,
        'team2_recent_form': team2.recent_form,
        'team1_batting_avg': team1.avg_score,
        'team2_batting_avg': team2.avg_score,
        'team1_bowling_avg': team1.avg_economy,
        'team2_bowling_avg': team2.avg_economy
    })
    
    if match.toss_winner_id:
        match_data['toss_winner'] = match.toss_winner_id
        match_data['toss_decision'] = match.toss_decision
    else:
        # If toss hasn't happened, assume team1 wins and chooses to bat
        match_data['toss_winner'] = match.team1_id
        match_data['toss_decision'] = 'bat'
    
    # Load match predictor model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'models', 'trained', 'match_predictor.joblib')
    match_predictor = MatchPredictor()
    match_predictor.load_model(model_path)
    
    # Make prediction
    match_df = pd.DataFrame([match_data])
    prediction_result = match_predictor.predict(match_df)
    
    # Load score predictor model
    score_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'models', 'trained', 'score_predictor.joblib')
    score_predictor = ScorePredictor()
    score_predictor.load_model(score_model_path)
    
    # Predict scores for both teams
    venue = db.query(Venue).filter(Venue.id == match.venue_id).first()
    
    team1_batting_data = {
        'match_id': match.id,
        'batting_team_id': match.team1_id,
        'bowling_team_id': match.team2_id,
        'venue_id': match.venue_id,
        'is_first_innings': True,
        'is_day_night': match.is_day_night,
        'pitch_type': match.pitch_type,
        'batting_team_avg_score': team1.avg_score,
        'batting_team_recent_form': team1.recent_form,
        'bowling_team_avg_economy': team2.avg_economy,
        'bowling_team_recent_form': team2.recent_form,
        'venue_avg_score': venue.avg_score if venue else 160
    }
    
    team2_batting_data = {
        'match_id': match.id,
        'batting_team_id': match.team2_id,
        'bowling_team_id': match.team1_id,
        'venue_id': match.venue_id,
        'is_first_innings': False,
        'is_day_night': match.is_day_night,
        'pitch_type': match.pitch_type,
        'batting_team_avg_score': team2.avg_score,
        'batting_team_recent_form': team2.recent_form,
        'bowling_team_avg_economy': team1.avg_economy,
        'bowling_team_recent_form': team1.recent_form,
        'venue_avg_score': venue.avg_score if venue else 160
    }
    
    team1_score_prediction = score_predictor.predict(pd.DataFrame([team1_batting_data]))
    team2_score_prediction = score_predictor.predict(pd.DataFrame([team2_batting_data]))
    
    # Get LLM reasoning
    try:
        llm = LLMReasoning()
        reasoning_data = {
            'team1_id': match.team1_id,
            'team2_id': match.team2_id,
            'predicted_winner': prediction_result['predicted_winner'],
            'team1_win_probability': prediction_result['team1_win_probability'],
            'team2_win_probability': prediction_result['team2_win_probability']
        }
        reasoning = llm.get_reasoning(reasoning_data, 'match')
    except Exception as e:
        reasoning = f"Unable to generate reasoning: {str(e)}"
    
    # Create prediction object
    db_prediction = Prediction(
        match_id=match.id,
        predicted_winner_id=prediction_result['predicted_winner'],
        team1_win_probability=prediction_result['team1_win_probability'],
        team2_win_probability=prediction_result['team2_win_probability'],
        team1_predicted_score=team1_score_prediction['predicted_score'],
        team2_predicted_score=team2_score_prediction['predicted_score'],
        confidence=prediction_result['confidence'],
        reasoning=reasoning
    )
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    
    return db_prediction

@app.post("/predictions/player/", response_model=PlayerPredictionResponse)
def predict_player_performance(request: PlayerPredictionRequest, db: Session = Depends(get_db)):
    # Check if match and player exist
    match = db.query(Match).filter(Match.id == request.match_id).first()
    player = db.query(Player).filter(Player.id == request.player_id).first()
    
    if match is None or player is None:
        raise HTTPException(status_code=404, detail="Match or player not found")
    
    # Check if prediction already exists
    existing_prediction = db.query(PlayerPrediction).filter(
        PlayerPrediction.match_id == request.match_id,
        PlayerPrediction.player_id == request.player_id
    ).first()
    
    if existing_prediction:
        return existing_prediction
    
    # Determine opposition team
    opposition_team_id = match.team2_id if player.team_id == match.team1_id else match.team1_id
    opposition_team = db.query(Team).filter(Team.id == opposition_team_id).first()
    
    # Prepare data for prediction
    if request.performance_type == 'batting':
        player_data = {
            'match_id': match.id,
            'player_id': player.id,
            'opposition_id': opposition_team_id,
            'venue_id': match.venue_id,
            'is_home_ground': 1 if match.venue_id == player.team.home_venue_id else 0,
            'pitch_type': match.pitch_type,
            'is_day_night': match.is_day_night,
            'batting_position': 'top_order',  # Simplified assumption
            'player_batting_avg': player.batting_avg,
            'player_strike_rate': player.strike_rate,
            'player_recent_form': player.recent_batting_form,
            'player_experience': player.experience,
            'opposition_bowling_avg': opposition_team.avg_economy
        }
        
        # Load batting predictor model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'models', 'trained', 'batting_predictor.joblib')
        player_predictor = PlayerPredictor(performance_type='batting')
        player_predictor.load_model(model_path)
        
    else:  # bowling
        player_data = {
            'match_id': match.id,
            'player_id': player.id,
            'opposition_id': opposition_team_id,
            'venue_id': match.venue_id,
            'is_home_ground': 1 if match.venue_id == player.team.home_venue_id else 0,
            'pitch_type': match.pitch_type,
            'is_day_night': match.is_day_night,
            'bowling_style': 'medium',  # Simplified assumption
            'player_bowling_avg': player.bowling_avg,
            'player_economy_rate': player.economy_rate,
            'player_recent_form': player.recent_bowling_form,
            'player_experience': player.experience,
            'opposition_batting_avg': opposition_team.avg_score
        }
        
        # Load bowling predictor model
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'models', 'trained', 'bowling_predictor.joblib')
        player_predictor = PlayerPredictor(performance_type='bowling')
        player_predictor.load_model(model_path)
    
    # Make prediction
    player_df = pd.DataFrame([player_data])
    prediction_result = player_predictor.predict(player_df)
    
    # Get LLM reasoning
    try:
        llm = LLMReasoning()
        reasoning_data = {
            'player_id': player.id,
            'opposition_id': opposition_team_id,
            **prediction_result
        }
        reasoning = llm.get_reasoning(reasoning_data, 'player')
    except Exception as e:
        reasoning = f"Unable to generate reasoning: {str(e)}"
    
    # Create prediction object
    if request.performance_type == 'batting':
        db_prediction = PlayerPrediction(
            player_id=player.id,
