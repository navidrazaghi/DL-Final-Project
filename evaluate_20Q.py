# evaluate_20Q.py

import argparse
from tqdm import tqdm  # A nice progress bar

# Import the necessary classes from your project files
from game_logic import Game20Q
from test import ValidatorModel

def run_single_game(my_model, validator_model):
    """
    Runs a single instance of the 20 Questions game from start to finish.
    """
    # The game loop continues until the game_over flag is True
    while not my_model.game_over:
        # 1. AI generates a question
        question = my_model.generate_question()
        if question is None: # Should not happen unless max_questions is 0
            break

        # 2. Validator answers the question
        answer = validator_model.validate_question(question)

        # 3. AI generates a guess based on the new information
        guess = my_model.generate_guess(question, answer)
        if guess is None: # Fallback if guess generation fails
            guess = "thing" 

        # 4. Validator checks the guess
        guess_correctness = validator_model.validate_guess(guess)
        
        # 5. Record the full turn in the game's history
        my_model.record_turn(
            question=question,
            user_answer=answer,
            guess=guess,
            guess_correctness=guess_correctness
        )

        # Optional: Print turn-by-turn details for debugging
        print(50*"-")
        print(f"  Q{my_model.questions_asked}: {question} -> Ans: {answer}")
        print(f"  Guess: {guess} -> Correct: {guess_correctness}")
        print(50*"-")
    # After the loop, check if the AI won
    if my_model.winner == 'ai':
        return True
    
    # print(f"AI failed. The word was: {validator_model.secret_word}")
    return False

def main():
    """
    Main function to parse arguments and run the evaluation loop.
    """
    parser = argparse.ArgumentParser(description="Evaluate the 20 Questions game model.")
    parser.add_argument("-N", "--num_games", type=int, required=True,
                        help="The number of game rounds to run for evaluation.")
    args = parser.parse_args()

    num_games = args.num_games
    successful_guesses = 0

    print(f"Starting evaluation for {num_games} games...")

    # Initialize the models once
    my_game_model = Game20Q()
    validator = ValidatorModel()
    
    # Use tqdm for a nice progress bar
    for i in tqdm(range(num_games), desc="Evaluating Games"):
        # Reset both models for a new game
        my_game_model.reset()
        validator.reset()
        
        # Run the game and check for success
        if run_single_game(my_game_model, validator):
            successful_guesses += 1
            
    # --- Final Report ---
    print("\n--------- Evaluation Complete ---------")
    success_rate = (successful_guesses / num_games) * 100
    print(f"Total Games Played: {num_games}")
    print(f"Successful Guesses (AI Wins): {successful_guesses}")
    print(f"Success Rate: {success_rate:.2f}%")
    print("---------------------------------------")

if __name__ == "__main__":
    main()