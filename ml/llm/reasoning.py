import requests
import json
import os
from typing import Dict, List, Any, Optional

class LLMReasoning:
    """
    Class for integrating LLM reasoning capabilities with IPL predictions.
    Uses Ollama for local LLM inference.
    """
    
    def __init__(self, model_name="llama2"):
        """
        Initialize the LLM reasoning module.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434/api/generate"
        
    def _generate_prompt(self, prediction_data: Dict[str, Any], prediction_type: str) -> str:
        """
        Generate a prompt for the LLM based on prediction data.
        
        Args:
            prediction_data (dict): Prediction data from ML models
            prediction_type (str): Type of prediction ('match', 'score', or 'player')
            
        Returns:
            str: Formatted prompt for the LLM
        """
        if prediction_type == 'match':
            team1_id = prediction_data.get('team1_id')
            team2_id = prediction_data.get('team2_id')
            winner_id = prediction_data.get('predicted_winner')
            team1_prob = prediction_data.get('team1_win_probability', 0)
            team2_prob = prediction_data.get('team2_win_probability', 0)
            
            prompt = f"""
            You are a cricket analyst specializing in IPL matches. Analyze the following match prediction:
            
            Team 1 (ID: {team1_id}) vs Team 2 (ID: {team2_id})
            Predicted winner: Team with ID {winner_id}
            Team 1 win probability: {team1_prob:.2f}
            Team 2 win probability: {team2_prob:.2f}
            
            Based on this prediction, provide a detailed analysis of:
            1. The key factors that might influence this outcome
            2. What strengths the predicted winning team likely has
            3. What weaknesses the predicted losing team might have
            4. Any specific match conditions that could affect the result
            5. How close the match is likely to be based on the probabilities
            
            Your analysis should be concise, insightful, and focused on cricket-specific factors.
            """
            
        elif prediction_type == 'score':
            batting_team_id = prediction_data.get('batting_team_id')
            bowling_team_id = prediction_data.get('bowling_team_id')
            predicted_score = prediction_data.get('predicted_score')
            lower_bound = prediction_data.get('lower_bound')
            upper_bound = prediction_data.get('upper_bound')
            
            prompt = f"""
            You are a cricket analyst specializing in IPL matches. Analyze the following score prediction:
            
            Batting Team (ID: {batting_team_id}) vs Bowling Team (ID: {bowling_team_id})
            Predicted score: {predicted_score} runs
            Confidence interval: {lower_bound} - {upper_bound} runs
            
            Based on this prediction, provide a detailed analysis of:
            1. What factors might contribute to this projected score
            2. How the batting team's strengths match up against the bowling team
            3. What strategies the bowling team might employ to restrict the score
            4. What would be a good target or chase strategy given this prediction
            5. How this score compares to typical IPL scores
            
            Your analysis should be concise, insightful, and focused on cricket-specific factors.
            """
            
        elif prediction_type == 'player':
            player_id = prediction_data.get('player_id')
            opposition_id = prediction_data.get('opposition_id')
            
            if 'predicted_runs' in prediction_data:
                # Batting prediction
                predicted_value = prediction_data.get('predicted_runs')
                lower_bound = prediction_data.get('lower_bound')
                upper_bound = prediction_data.get('upper_bound')
                performance_type = "batting"
                metric = "runs"
            else:
                # Bowling prediction
                predicted_value = prediction_data.get('predicted_wickets')
                lower_bound = prediction_data.get('lower_bound')
                upper_bound = prediction_data.get('upper_bound')
                performance_type = "bowling"
                metric = "wickets"
            
            prompt = f"""
            You are a cricket analyst specializing in IPL player performance. Analyze the following player prediction:
            
            Player (ID: {player_id}) against Team (ID: {opposition_id})
            Predicted {performance_type} performance: {predicted_value} {metric}
            Confidence interval: {lower_bound} - {upper_bound} {metric}
            
            Based on this prediction, provide a detailed analysis of:
            1. What factors might contribute to this projected performance
            2. How the player's strengths match up against this specific opposition
            3. What strategies the player might employ for success
            4. What would be considered a good performance in this context
            5. How this performance might impact the match outcome
            
            Your analysis should be concise, insightful, and focused on cricket-specific factors.
            """
        
        else:
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        
        return prompt.strip()
    
    def get_reasoning(self, prediction_data: Dict[str, Any], prediction_type: str) -> str:
        """
        Get LLM reasoning for a prediction.
        
        Args:
            prediction_data (dict): Prediction data from ML models
            prediction_type (str): Type of prediction ('match', 'score', or 'player')
            
        Returns:
            str: LLM reasoning text
        """
        prompt = self._generate_prompt(prediction_data, prediction_type)
        
        try:
            # Call Ollama API
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"Error calling Ollama API: {response.status_code}")
                print(response.text)
                return "Unable to generate reasoning at this time."
                
        except Exception as e:
            print(f"Exception when calling Ollama API: {str(e)}")
            return "Unable to generate reasoning due to an error."
    
    def format_reasoning(self, reasoning_text: str) -> Dict[str, str]:
        """
        Format the raw reasoning text into structured sections.
        
        Args:
            reasoning_text (str): Raw reasoning text from LLM
            
        Returns:
            dict: Structured reasoning with sections
        """
        # Simple formatting - in a real implementation, you might use more
        # sophisticated parsing to extract specific sections
        lines = reasoning_text.strip().split('\n')
        sections = []
        current_section = ""
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_section:
                    sections.append(current_section)
                current_section = line
            else:
                current_section += " " + line
        
        if current_section:
            sections.append(current_section)
        
        # Create a structured format
        result = {
            "summary": sections[0] if sections else reasoning_text,
            "details": sections[1:] if len(sections) > 1 else []
        }
        
        return result

# Example usage
if __name__ == "__main__":
    # Initialize LLM reasoning
    llm = LLMReasoning(model_name="llama2")
    
    # Example match prediction data
    match_prediction = {
        'team1_id': 1,  # Mumbai Indians
        'team2_id': 2,  # Chennai Super Kings
        'predicted_winner': 1,
        'team1_win_probability': 0.65,
        'team2_win_probability': 0.35
    }
    
    # Get reasoning
    reasoning = llm.get_reasoning(match_prediction, 'match')
    
    # Format reasoning
    formatted_reasoning = llm.format_reasoning(reasoning)
    
    print("LLM Reasoning:")
    print(formatted_reasoning['summary'])
    print("\nDetails:")
    for detail in formatted_reasoning['details']:
        print(f"- {detail}")
