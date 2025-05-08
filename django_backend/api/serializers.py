from rest_framework import serializers
from .models import Team, Player, Venue, Match, PlayerPerformance, Prediction, PlayerPrediction

class TeamSerializer(serializers.ModelSerializer):
    class Meta:
        model = Team
        fields = '__all__'

class PlayerSerializer(serializers.ModelSerializer):
    team_name = serializers.ReadOnlyField(source='team.name')
    
    class Meta:
        model = Player
        fields = '__all__'

class VenueSerializer(serializers.ModelSerializer):
    class Meta:
        model = Venue
        fields = '__all__'

class MatchSerializer(serializers.ModelSerializer):
    team1_name = serializers.ReadOnlyField(source='team1.name')
    team2_name = serializers.ReadOnlyField(source='team2.name')
    venue_name = serializers.ReadOnlyField(source='venue.name')
    
    class Meta:
        model = Match
        fields = '__all__'

class PlayerPerformanceSerializer(serializers.ModelSerializer):
    player_name = serializers.ReadOnlyField(source='player.name')
    match_details = serializers.ReadOnlyField(source='match.__str__')
    
    class Meta:
        model = PlayerPerformance
        fields = '__all__'

class PredictionSerializer(serializers.ModelSerializer):
    match_details = serializers.ReadOnlyField(source='match.__str__')
    predicted_winner_name = serializers.ReadOnlyField(source='predicted_winner.name')
    
    class Meta:
        model = Prediction
        fields = '__all__'

class PlayerPredictionSerializer(serializers.ModelSerializer):
    player_name = serializers.ReadOnlyField(source='player.name')
    match_details = serializers.ReadOnlyField(source='match.__str__')
    
    class Meta:
        model = PlayerPrediction
        fields = '__all__'

class MatchPredictionRequestSerializer(serializers.Serializer):
    match_id = serializers.IntegerField(required=True)

class ScorePredictionRequestSerializer(serializers.Serializer):
    match_id = serializers.IntegerField(required=True)
    batting_team_id = serializers.IntegerField(required=True)
    bowling_team_id = serializers.IntegerField(required=True)
    is_first_innings = serializers.BooleanField(required=True)

class PlayerPredictionRequestSerializer(serializers.Serializer):
    match_id = serializers.IntegerField(required=True)
    player_id = serializers.IntegerField(required=True)
    performance_type = serializers.ChoiceField(choices=['batting', 'bowling'], required=True)
