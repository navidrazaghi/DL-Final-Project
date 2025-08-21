# test.py

import random
import ollama

class ValidatorModel:
    """
    Simulates a player who has chosen a secret word.
    This model is used to validate the questions and guesses of the AI player.
    """
    def __init__(self):
        """Initializes the validator with a dataset of words and selects one."""
        self.dataset = [
            # Animals
            "elephant", "giraffe", "penguin", "dolphin", "kangaroo",
            "octopus", "butterfly", "peacock", "scorpion", "wolf",
            # Objects
            "computer", "telescope", "microscope", "guitar", "piano",
            "camera", "drone", "bicycle", "motorcycle", "helicopter",
            # Nature & Places
            "mountain", "ocean", "volcano", "desert", "rainforest",
            "river", "waterfall", "pyramid", "castle", "bridge",
            # Food
            "pizza", "sushi", "chocolate", "coffee", "strawberry",
            "avocado", "pineapple", "popcorn", "honey", "cheese",
            # Concepts & Abstract
            "dream", "music", "gravity", "internet", "history",
            "love", "time", "shadow", "echo", "rainbow"
        ]
        self.secret_word = ""
        # self.reset()

    def reset(self):
        """Selects a new random secret word from the dataset for a new game."""
        self.secret_word = random.choice(self.dataset)
        # print(f"Validator has selected a new secret word.")
        print(f"\n--- New Game --- Validator's Secret Word: [ {self.secret_word} ] ---")

    def validate_question(self, question: str) -> str:
        """
        Answers a yes/no question about the secret word using an LLM for logical consistency.
        """
        prompt = f"""
        You are an assistant playing '20 Questions'.
        The secret word is: '{self.secret_word}'.
        The user asked this question: '{question}'
        You MUST answer with only 'yes' or 'no'.
        """
        try:
            response = ollama.chat(
                model='mistral',
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0}
            )
            answer = response['message']['content'].strip().lower()
            # Ensure the response is strictly 'yes' or 'no'
            if 'yes' in answer:
                return 'yes'
            return 'no'
        except Exception as e:
            print(f"Error validating question: {e}")
            return 'no' # Default to 'no' on error

    def validate_guess(self, guess: str) -> str:
        """
        Validates the final guess against the secret word.
        """
        if guess.strip().lower() == self.secret_word:
            return "yes"
        else:
            return "no"