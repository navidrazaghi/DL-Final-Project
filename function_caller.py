# function_caller.py 

import os
import ollama
from exa_py import Exa
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Exa API client
exa = Exa(api_key=os.getenv("EXA_API_KEY"))

def get_intent(user_prompt: str) -> str:
    """
    Determines the user's intent by asking the LLM to classify the prompt.
    """
    system_prompt = """
    You are an expert intent detection model. Your task is to classify a user's prompt into one of three categories:
    1. 'web_search': The user is asking for real-time, recent, or up-to-date information (e.g., news, weather, stock prices, latest events, "what is the latest on...").
    2. 'play_game': The user explicitly wants to play the '20 Questions' game.
    3. 'chat': The user is having a general conversation, asking for information from a provided context, or anything that doesn't fit the other two categories.

    You must respond with ONLY one of the category names: 'web_search', 'play_game', or 'chat'. Do not provide any other text or explanation.
    """
    
    try:
        response = ollama.chat(
            model='mistral',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            options={'temperature': 0}
        )
        intent = response['message']['content'].strip().lower()

        if intent in ['web_search', 'play_game', 'chat']:
            return intent
        else:
            return 'chat'
            
    except Exception as e:
        print(f"Error in intent detection: {e}")
        return 'chat'

def search_web(query: str) -> str:
    """
    Performs a web search using the Exa API and returns a formatted string of results.
    This version provides a cleaner format for the LLM to process.
    """
    try:
        print(f"Performing web search for: '{query}'")
        search_response = exa.search_and_contents(
            query,
            num_results=3,  # Get top 3 results as per project requirements
            text={"max_characters": 1000}, # Get a decent amount of text for summarizing
        )
        
        # Format the results into a clean, readable string for the summarization prompt
        formatted_results = []
        for i, result in enumerate(search_response.results, 1):
            formatted_results.append(f"Source {i}:\nTitle: {result.title}\nContent: {result.text}...")
        
        print("Web search successful.")
        return "\n\n".join(formatted_results)

    except Exception as e:
        print(f"An error occurred during web search: {e}")
        return "Sorry, I couldn't perform the web search due to an error."