from django.db import models

class Team(models.Model):
    """Model representing an IPL team."""
    name = models.CharField(max_length=100)
    short_name = models.CharField(max_length=10)
    logo = models.URLField(blank=True, null=True)
    home_venue = models.ForeignKey('Venue', on_delete=models.SET_NULL, null=True, related_name='home_teams')
    
    # Team statistics
    win_rate = models.FloatField(default=0.0)
    home_advantage = models.FloatField(default=0.0)
    avg_score = models.FloatField(default=0.0)
    avg_economy = models.FloatField(default=0.0)
    recent_form = models.FloatField(default=0.0)
    
    def __str__(self):
        return self.name

class Player(models.Model):
    """Model representing an IPL player."""
    name = models.CharField(max_length=100)
    team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='players')
    role = models.CharField(max_length=20, choices=[
        ('batsman', 'Batsman'),
        ('bowler', 'Bowler'),
        ('all_rounder', 'All-Rounder'),
        ('wicket_keeper', 'Wicket Keeper')
    ])
    batting_style = models.CharField(max_length=20, blank=True, null=True)
    bowling_style = models.CharField(max_length=20, blank=True, null=True)
    nationality = models.CharField(max_length=50, blank=True, null=True)
    
    # Player statistics
    batting_avg = models.FloatField(default=0.0)
    strike_rate = models.FloatField(default=0.0)
    bowling_avg = models.FloatField(default=0.0)
    economy_rate = models.FloatField(default=0.0)
    experience = models.IntegerField(default=0)
    recent_batting_form = models.FloatField(default=0.0)
    recent_bowling_form = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"{self.name} ({self.team.short_name})"

class Venue(models.Model):
    """Model representing an IPL match venue."""
    name = models.CharField(max_length=100)
    city = models.CharField(max_length=50)
    country = models.CharField(max_length=50, default='India')
    capacity = models.IntegerField(blank=True, null=True)
    
    # Venue statistics
    avg_score = models.FloatField(default=0.0)
    avg_wickets = models.FloatField(default=0.0)
    
    def __str__(self):
        return f"{self.name}, {self.city}"

class Match(models.Model):
    """Model representing an IPL match."""
    match_date = models.DateTimeField()
    venue = models.ForeignKey(Venue, on_delete=models.CASCADE, related_name='matches')
    team1 = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='home_matches')
    team2 = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='away_matches')
    is_day_night = models.BooleanField(default=False)
    pitch_type = models.CharField(max_length=20, choices=[
        ('batting', 'Batting Friendly'),
        ('bowling', 'Bowling Friendly'),
        ('balanced', 'Balanced')
    ], default='balanced')
    
    # Match result (null if not played yet)
    toss_winner = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='toss_wins', null=True, blank=True)
    toss_decision = models.CharField(max_length=10, choices=[
        ('bat', 'Bat'),
        ('field', 'Field')
    ], null=True, blank=True)
    winner = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='match_wins', null=True, blank=True)
    team1_score = models.IntegerField(null=True, blank=True)
    team2_score = models.IntegerField(null=True, blank=True)
    
    def __str__(self):
        return f"{self.team1.short_name} vs {self.team2.short_name} at {self.venue.name} on {self.match_date.strftime('%Y-%m-%d')}"

class PlayerPerformance(models.Model):
    """Model representing a player's performance in a match."""
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='performances')
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name='player_performances')
    
    # Batting performance
    runs_scored = models.IntegerField(default=0)
    balls_faced = models.IntegerField(default=0)
    fours = models.IntegerField(default=0)
    sixes = models.IntegerField(default=0)
    
    # Bowling performance
    overs_bowled = models.FloatField(default=0.0)
    wickets_taken = models.IntegerField(default=0)
    runs_conceded = models.IntegerField(default=0)
    
    def __str__(self):
        return f"{self.player.name} in {self.match}"

class Prediction(models.Model):
    """Model representing a prediction for a match."""
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name='predictions')
    predicted_winner = models.ForeignKey(Team, on_delete=models.CASCADE, related_name='predicted_wins')
    team1_win_probability = models.FloatField()
    team2_win_probability = models.FloatField()
    team1_predicted_score = models.IntegerField()
    team2_predicted_score = models.IntegerField()
    confidence = models.FloatField()
    reasoning = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.match}"

class PlayerPrediction(models.Model):
    """Model representing a prediction for a player's performance."""
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name='performance_predictions')
    match = models.ForeignKey(Match, on_delete=models.CASCADE, related_name='player_predictions')
    predicted_runs = models.IntegerField(null=True, blank=True)
    predicted_wickets = models.IntegerField(null=True, blank=True)
    confidence = models.FloatField()
    reasoning = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Prediction for {self.player.name} in {self.match}"
