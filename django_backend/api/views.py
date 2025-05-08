import os
import sys
import joblib
import pandas as pd
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.conf import settings

from .models import Team, Player, Venue, Match, PlayerPerformance, Prediction, PlayerPrediction
from .serializers import (
    TeamSerializer, PlayerSerializer, VenueSerializer, MatchSerializer,
    PlayerPerformanceSerializer, PredictionSerializer, PlayerPredictionSerializer,
    MatchPredictionRequestSerializer, ScorePredictionRequestSerializer,
    PlayerPredictionRequestSerializer
)

# Add ML module to path
sys.path.append(os.path.join(settings.BASE_DIR, '..', 'ml'))

# Import ML models
from models.match_predictor import MatchPredictor
from models.score_predictor import ScorePredictor
from models.player_predictor import PlayerPredictor
from llm.reasoning import LLMReasoning

class TeamViewSet(viewsets.ModelViewSet):
    queryset = Team.objects.all()
    serializer_class = TeamSerializer

class PlayerViewSet(viewsets.ModelViewSet):
    queryset = Player.objects.all()
    serializer_class = PlayerSerializer
    
    def get_queryset(self):
        queryset = Player.objects.all()
        team_id = self.request.query_params.get('team_id', None)
        role = self.request.query_params.get('role', None)
        
        if team_id:
            queryset = queryset.filter(team_id=team_id)
        if role:
            queryset = queryset.filter(role=role)
            
        return queryset

class VenueViewSet(viewsets.ModelViewSet):
    queryset = Venue.objects.all()
    serializer_class = VenueSerializer

class MatchViewSet(viewsets.ModelViewSet):
    queryset = Match.objects.all()
    serializer_class = MatchSerializer
    
    def get_queryset(self):
        queryset = Match.objects.all()
        team_id = self.request.query_params.get('team_id', None)
        venue_id = self.request.query_params.get('venue_id', None)
        
        if team_id:
            queryset = queryset.filter(team1_id=team_id) | queryset.filter(team2_id=team_id)
        if venue_id:
            queryset = queryset.filter(venue_id=venue_id)
            
        return queryset

class PlayerPerformanceViewSet(viewsets.ModelViewSet):
    queryset = PlayerPerformance.objects.all()
    serializer_class = PlayerPerformanceSerializer
    
    def get_queryset(self):
        queryset = PlayerPerformance.objects.all()
        player_id = self.request.query_params.get('player_id', None)
        match_id = self.request.query_params.get('match_id', None)
        
        if player_id:
            queryset = queryset.filter(player_id=player_id)
        if match_id:
            queryset = queryset.filter(match_id=match_id)
            
        return queryset

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    
    def get_queryset(self):
        queryset = Prediction.objects.all()
        match_id = self.request.query_params.get('match_id', None)
        
        if match_id:
            queryset = queryset.filter(match_id=match_id)
            
        return queryset
    
    @action(detail=False, methods=['post'])
    def predict_match(self, request):
        """
        Predict the outcome of a match.
        """
        serializer = MatchPredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        match_id = serializer.validated_data['match_id']
        
        try:
            match = Match.objects.get(id=match_id)
        except Match.DoesNotExist:
            return Response({'error': 'Match not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Check if prediction already exists
        existing_prediction = Prediction.objects.filter(match=match).first()
        if existing_prediction:
            serializer = PredictionSerializer(existing_prediction)
            return Response(serializer.data)
        
        # Prepare data for prediction
        match_data = {
            'match_id': match.id,
            'team1_id': match.team1.id,
            'team2_id': match.team2.id,
            'venue_id': match.venue.id,
            'is_day_night': match.is_day_night,
            'pitch_type': match.pitch_type,
            'team1_win_rate': match.team1.win_rate,
            'team2_win_rate': match.team2.win_rate,
            'team1_home_advantage': match.team1.home_advantage,
            'team2_home_advantage': match.team2.home_advantage,
            'team1_recent_form': match.team1.recent_form,
            'team2_recent_form': match.team2.recent_form,
            'team1_batting_avg': match.team1.avg_score,
            'team2_batting_avg': match.team2.avg_score,
            'team1_bowling_avg': match.team1.avg_economy,
            'team2_bowling_avg': match.team2.avg_economy
        }
        
        if match.toss_winner:
            match_data['toss_winner'] = match.toss_winner.id
            match_data['toss_decision'] = match.toss_decision
        else:
            # If toss hasn't happened, assume team1 wins and chooses to bat
            match_data['toss_winner'] = match.team1.id
            match_data['toss_decision'] = 'bat'
        
        # Load match predictor model
        model_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'models', 'trained', 'match_predictor.joblib')
        match_predictor = MatchPredictor()
        match_predictor.load_model(model_path)
        
        # Make prediction
        match_df = pd.DataFrame([match_data])
        prediction_result = match_predictor.predict(match_df)
        
        # Load score predictor model
        score_model_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'models', 'trained', 'score_predictor.joblib')
        score_predictor = ScorePredictor()
        score_predictor.load_model(score_model_path)
        
        # Predict scores for both teams
        team1_batting_data = {
            'match_id': match.id,
            'batting_team_id': match.team1.id,
            'bowling_team_id': match.team2.id,
            'venue_id': match.venue.id,
            'is_first_innings': True,
            'is_day_night': match.is_day_night,
            'pitch_type': match.pitch_type,
            'batting_team_avg_score': match.team1.avg_score,
            'batting_team_recent_form': match.team1.recent_form,
            'bowling_team_avg_economy': match.team2.avg_economy,
            'bowling_team_recent_form': match.team2.recent_form,
            'venue_avg_score': match.venue.avg_score
        }
        
        team2_batting_data = {
            'match_id': match.id,
            'batting_team_id': match.team2.id,
            'bowling_team_id': match.team1.id,
            'venue_id': match.venue.id,
            'is_first_innings': False,
            'is_day_night': match.is_day_night,
            'pitch_type': match.pitch_type,
            'batting_team_avg_score': match.team2.avg_score,
            'batting_team_recent_form': match.team2.recent_form,
            'bowling_team_avg_economy': match.team1.avg_economy,
            'bowling_team_recent_form': match.team1.recent_form,
            'venue_avg_score': match.venue.avg_score
        }
        
        team1_score_prediction = score_predictor.predict(pd.DataFrame([team1_batting_data]))
        team2_score_prediction = score_predictor.predict(pd.DataFrame([team2_batting_data]))
        
        # Get LLM reasoning
        try:
            llm = LLMReasoning()
            reasoning_data = {
                'team1_id': match.team1.id,
                'team2_id': match.team2.id,
                'predicted_winner': prediction_result['predicted_winner'],
                'team1_win_probability': prediction_result['team1_win_probability'],
                'team2_win_probability': prediction_result['team2_win_probability']
            }
            reasoning = llm.get_reasoning(reasoning_data, 'match')
        except Exception as e:
            reasoning = f"Unable to generate reasoning: {str(e)}"
        
        # Create prediction object
        prediction = Prediction.objects.create(
            match=match,
            predicted_winner=Team.objects.get(id=prediction_result['predicted_winner']),
            team1_win_probability=prediction_result['team1_win_probability'],
            team2_win_probability=prediction_result['team2_win_probability'],
            team1_predicted_score=team1_score_prediction['predicted_score'],
            team2_predicted_score=team2_score_prediction['predicted_score'],
            confidence=prediction_result['confidence'],
            reasoning=reasoning
        )
        
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)

class PlayerPredictionViewSet(viewsets.ModelViewSet):
    queryset = PlayerPrediction.objects.all()
    serializer_class = PlayerPredictionSerializer
    
    def get_queryset(self):
        queryset = PlayerPrediction.objects.all()
        player_id = self.request.query_params.get('player_id', None)
        match_id = self.request.query_params.get('match_id', None)
        
        if player_id:
            queryset = queryset.filter(player_id=player_id)
        if match_id:
            queryset = queryset.filter(match_id=match_id)
            
        return queryset
    
    @action(detail=False, methods=['post'])
    def predict_player_performance(self, request):
        """
        Predict a player's performance in a match.
        """
        serializer = PlayerPredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        match_id = serializer.validated_data['match_id']
        player_id = serializer.validated_data['player_id']
        performance_type = serializer.validated_data['performance_type']
        
        try:
            match = Match.objects.get(id=match_id)
            player = Player.objects.get(id=player_id)
        except (Match.DoesNotExist, Player.DoesNotExist):
            return Response({'error': 'Match or player not found'}, status=status.HTTP_404_NOT_FOUND)
        
        # Check if prediction already exists
        existing_prediction = PlayerPrediction.objects.filter(match=match, player=player).first()
        if existing_prediction:
            serializer = PlayerPredictionSerializer(existing_prediction)
            return Response(serializer.data)
        
        # Determine opposition team
        opposition_team = match.team2 if player.team == match.team1 else match.team1
        
        # Prepare data for prediction
        if performance_type == 'batting':
            player_data = {
                'match_id': match.id,
                'player_id': player.id,
                'opposition_id': opposition_team.id,
                'venue_id': match.venue.id,
                'is_home_ground': 1 if match.venue == player.team.home_venue else 0,
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
            model_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'models', 'trained', 'batting_predictor.joblib')
            player_predictor = PlayerPredictor(performance_type='batting')
            player_predictor.load_model(model_path)
            
        else:  # bowling
            player_data = {
                'match_id': match.id,
                'player_id': player.id,
                'opposition_id': opposition_team.id,
                'venue_id': match.venue.id,
                'is_home_ground': 1 if match.venue == player.team.home_venue else 0,
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
            model_path = os.path.join(settings.BASE_DIR, '..', 'ml', 'models', 'trained', 'bowling_predictor.joblib')
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
                'opposition_id': opposition_team.id,
                **prediction_result
            }
            reasoning = llm.get_reasoning(reasoning_data, 'player')
        except Exception as e:
            reasoning = f"Unable to generate reasoning: {str(e)}"
        
        # Create prediction object
        if performance_type == 'batting':
            player_prediction = PlayerPrediction.objects.create(
                player=player,
                match=match,
                predicted_runs=prediction_result['predicted_runs'],
                predicted_wickets=None,
                confidence=prediction_result['upper_bound'] - prediction_result['lower_bound'],
                reasoning=reasoning
            )
        else:  # bowling
            player_prediction = PlayerPrediction.objects.create(
                player=player,
                match=match,
                predicted_runs=None,
                predicted_wickets=prediction_result['predicted_wickets'],
                confidence=prediction_result['upper_bound'] - prediction_result['lower_bound'],
                reasoning=reasoning
            )
        
        serializer = PlayerPredictionSerializer(player_prediction)
        return Response(serializer.data)
