"""
Evaluation script for 20 Questions Game
Usage: python evaluate_20Q.py -N 100
"""

import argparse
import sys
import os
from typing import Dict, Any, List
import random

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# This would be provided by TAs - creating a mock for demonstration
class ValidatorModel:
    """Mock validator model for testing (TAs will provide the real implementation)"""
   
    def __init__(self):
        self.target_words = [
            "dog", "cat", "car", "book", "phone", "apple", "tree", "house",
            "chair", "table", "computer", "ball", "water", "fire", "sun",
            "moon", "flower", "bird", "fish", "door", "window", "pen", "paper"
        ]
        self.reset_word()
   
    def reset_word(self):
        """Select a new random word"""
        self.current_word = random.choice(self.target_words)
        self.word_attributes = self._get_word_attributes(self.current_word)
   
    def _get_word_attributes(self, word: str) -> Dict[str, bool]:
        """Define attributes for each word (simplified logic)"""
        attributes = {
            "dog": {"alive": True, "animal": True, "big": True, "indoor": False, "electronic": False},
            "cat": {"alive": True, "animal": True, "big": False, "indoor": True, "electronic": False},
            "car": {"alive": False, "animal": False, "big": True, "indoor": False, "electronic": True},
            "book": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": False},
            "phone": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": True},
            "apple": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": False},
            "tree": {"alive": True, "animal": False, "big": True, "indoor": False, "electronic": False},
            "house": {"alive": False, "animal": False, "big": True, "indoor": False, "electronic": False},
            "chair": {"alive": False, "animal": False, "big": True, "indoor": True, "electronic": False},
            "table": {"alive": False, "animal": False, "big": True, "indoor": True, "electronic": False},
            "computer": {"alive": False, "animal": False, "big": True, "indoor": True, "electronic": True},
            "ball": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": False},
            "water": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": False},
            "fire": {"alive": False, "animal": False, "big": False, "indoor": False, "electronic": False},
            "sun": {"alive": False, "animal": False, "big": True, "indoor": False, "electronic": False},
            "moon": {"alive": False, "animal": False, "big": True, "indoor": False, "electronic": False},
            "flower": {"alive": True, "animal": False, "big": False, "indoor": False, "electronic": False},
            "bird": {"alive": True, "animal": True, "big": False, "indoor": False, "electronic": False},
            "fish": {"alive": True, "animal": True, "big": False, "indoor": False, "electronic": False},
            "door": {"alive": False, "animal": False, "big": True, "indoor": True, "electronic": False},
            "window": {"alive": False, "animal": False, "big": True, "indoor": True, "electronic": False},
            "pen": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": False},
            "paper": {"alive": False, "animal": False, "big": False, "indoor": True, "electronic": False}
        }
       
        return attributes.get(word, {
            "alive": False, "animal": False, "big": False,
            "indoor": True, "electronic": False
        })
   
    def validate_question(self, question: str) -> str:
        """Answer yes/no questions about the current word"""
        question = question.lower().strip().rstrip('?')
       
        # Simple question parsing (in real implementation this would be more sophisticated)
        if "alive" in question or "living" in question:
            return "yes" if self.word_attributes.get("alive", False) else "no"
        elif "animal" in question:
            return "yes" if self.word_attributes.get("animal", False) else "no"
        elif "big" in question or "large" in question or "bigger than" in question:
            return "yes" if self.word_attributes.get("big", False) else "no"
        elif "indoor" in question or "inside" in question or "house" in question:
            return "yes" if self.word_attributes.get("indoor", False) else "no"
        elif "electronic" in question or "electric" in question or "technology" in question:
            return "yes" if self.word_attributes.get("electronic", False) else "no"
        elif "eat" in question or "food" in question:
            return "yes" if self.current_word in ["apple", "water"] else "no"
        elif "move" in question or "moving" in question:
            return "yes" if self.word_attributes.get("alive", False) or self.current_word == "car" else "no"
        elif "metal" in question:
            return "yes" if self.current_word in ["car", "computer", "phone"] else "no"
        elif "soft" in question:
            return "yes" if self.current_word in ["cat", "dog", "paper"] else "no"
        elif "round" in question or "circular" in question:
            return "yes" if self.current_word in ["ball", "apple", "sun", "moon"] else "no"
        else:
            # Random answer for unknown questions
            return random.choice(["yes", "no"])
   
    def validate_guess(self, guess: str) -> str:
        """Check if the guess is correct"""
        return "yes" if guess.lower().strip() == self.current_word.lower() else "no"

class GameEvaluator:
    """Evaluator for the 20 Questions game"""
   
    def __init__(self, ai_agent):
        self.ai_agent = ai_agent
        self.validator = ValidatorModel()
   
    def run_single_game(self) -> bool:
        """Run a single game and return True if AI wins"""
        # Reset both the validator and the game
        self.validator.reset_word()
        self.ai_agent.game.reset_game()
        self.ai_agent.game.start_game(self.validator.current_word)
       
        question_count = 0
        max_questions = 20
       
        while question_count < max_questions:
            # AI generates a question
            question = self.ai_agent.game.generate_question()
            question_count += 1
           
            # Validator answers the question
            answer = self.validator.validate_question(question)
           
            # AI processes the answer, which includes making a guess
            response = self.ai_agent.game.process_answer(answer) 

            # The response will be the guess, e.g., "My guess: Is it a car? (Yes/No)"
            # Extract the guess from the response
            guess = self.ai_agent.game.last_guess 

            if guess:
                result = self.validator.validate_guess(guess)
                if result == "yes":
                    # The user (validator) would say 'yes' to the guess
                    # Simulate this by calling process_answer again
                    final_response = self.ai_agent.game.process_answer("yes")
                    if "won" in final_response:
                        return True # AI wins
       
        return False  # AI loses (didn't guess correctly within 20 questions)
   
    def evaluate(self, num_games: int) -> Dict[str, Any]:
        """Run evaluation for specified number of games"""
        wins = 0
        total_games = num_games
       
        print(f"Starting evaluation with {num_games} games...")
       
        for game_num in range(num_games):
            if game_num % 10 == 0:
                print(f"Progress: {game_num}/{num_games} games completed")
           
            if self.run_single_game():
                wins += 1
       
        win_rate = wins / total_games if total_games > 0 else 0
       
        results = {
            "total_games": total_games,
            "wins": wins,
            "losses": total_games - wins,
            "win_rate": win_rate
        }
       
        return results

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate 20 Questions Game AI')
    parser.add_argument('-N', '--num-games', type=int, required=True,
                       help='Number of games to play for evaluation')
   
    args = parser.parse_args()
   
    if args.num_games <= 0:
        print("Error: Number of games must be positive")
        sys.exit(1)
   
    try:
        # Import and initialize the AI agent
        from ai_chat_agent import AIChatAgent
       
        # Create AI agent
        ai_agent = AIChatAgent()
       
        # Create evaluator
        evaluator = GameEvaluator(ai_agent)
       
        # Run evaluation
        results = evaluator.evaluate(args.num_games)
       
        # Print results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Total games played: {results['total_games']}")
        print(f"Games won: {results['wins']}")
        print(f"Games lost: {results['losses']}")
        print(f"Win rate: {results['win_rate']:.2%}")
        print("="*50)
       
        return results['wins']
       
    except ImportError as e:
        print(f"Error importing AI agent: {e}")
        print("Make sure ai_chat_agent.py is in the same directory")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()