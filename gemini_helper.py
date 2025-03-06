import google as genai
from typing import Dict, Any

class GeminiHelper:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def get_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Get response from Gemini model"""
        chat = self.model.start_chat(history=[])
        
        # Add context if provided
        if context:
            context_str = "\nContext: " + str(context)
            prompt += context_str
            
        response = chat.send_message(prompt)
        return response.text
        
    def format_inventory_response(self, prediction_data: Dict[str, Any]) -> str:
        """Format inventory prediction data into natural language"""
        prompt = f"""
        Based on the following inventory prediction data:
        - Product: {prediction_data['description']}
        - Current Average Daily Demand: {prediction_data['metrics']['avg_daily_demand']:.2f}
        - Recommended Restock Amount: {prediction_data['recommended_restock']}
        - Forecast Period: {prediction_data['days_to_forecast']} days
        
        Provide a natural language summary of the inventory recommendation.
        Include insights about the demand patterns and suggested actions.
        """
        return self.get_response(prompt, prediction_data)
