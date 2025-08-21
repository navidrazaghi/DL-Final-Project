# game_logic.py

import ollama
import re

class Game20Q:
    """
    Manages the state and logic for the '20 Questions' game.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the game to its initial state."""
        self.questions_asked = 0
        self.max_questions = 20
        self.game_over = False
        self.winner = None
        self.history = []  # Stores dictionaries with all turn info, including guess feedback

    def get_initial_message(self):
        """Returns the welcome message to start the game."""
        return (
            "Let's play 20 Questions! Please think of a word, and I will try to guess it in 20 questions or less. "
            "You can answer my questions with 'yes' or 'no'. Let's begin!"
        )

    # ----------------- Helpers -----------------

    def _clean_text(self, s: str) -> str:
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s\u0600-\u06FF]', ' ', s.lower())).strip()

    def _extract_prev_qs_guesses(self):
        """Extracts previous questions and guesses (cleaned) from history."""
        qs, ans, gs, gsfb = set(), set(), set(), set()
        for turn in self.history:
            if turn.get("question"):
                qs.add(self._clean_text(turn["question"]))
            if turn.get("guess"):
                gs.add(self._clean_text(turn["guess"]))
            if turn.get("answer"):
                ans.add(self._clean_text(turn["answer"]))
            if turn.get("guess_feedback"):
                gsfb.add(self._clean_text(turn["guess_feedback"]))
        return qs, ans, gs, gsfb

    def _format_history(self):
        """
        Formats the game history for the prompt, now including feedback on guesses.
        """
        if not self.history:
            return "No questions asked yet. This is your first question."

        formatted = ""
        for i, turn in enumerate(self.history):
            formatted += (
                f"--- Turn {i+1} ---\n"
                f"- My Question: {turn['question']}\n"
                f"- User's Answer: {turn['answer']}\n"
                f"- My Guess: {turn['guess']}\n"
                f"- Guess Correctness: {turn['guess_feedback']}\n"
            )
        return formatted

    # ----------------- Core Methods -----------------

    def generate_question(self):
        """
        Generates the next yes/no question based on history (no guess here).
        """
        if self.questions_asked >= self.max_questions:
            return None

        history_text = self._format_history()
        prev_qs, _, prev_gs, _ = self._extract_prev_qs_guesses()

        system_prompt = f"""
You are the Question-Asker in a 20 Questions game.
Your task: generate exactly ONE yes/no question.

Rules:
1. Maximum 16 words
2. Do not use commas or parentheses
3. Do not repeat any previous questions: {prev_qs or '[none]'}
4. Do not reuse wording from: {history_text}
5. Each question must have a clear yes/no answer
6. The question must NOT contain the word "or" in any form
7. Do not ask questions comparing two things or giving options
8. The question must be simple, logical, and related to previous answers

Final Output: ONE yes/no question only.
"""
        response = ollama.chat(
            model="mistral",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": history_text}],
            options={"temperature": 0.2}
        )
        content = response["message"]["content"]
        # print("question generated " , content)
        # m = re.search(r'Question:\s*(.+)', content, flags=re.IGNORECASE)

        # if not m:
        #     return "Is it man-made?"  # fallback
        return content.strip()

    def generate_guess(self, current_question, current_answer):
        """
        Generates the next logical single-word guess (no question).
        """
        history_text = self._format_history()
        _, _, prev_gs, _ = self._extract_prev_qs_guesses()
        history_text += ( 
                f"--- Turn {len(self.history)+2} ---\n"
                f"- My Question: {current_question}\n"
                f"- User's Answer: {current_answer}\n")

        system_prompt = f"""
            You are the ONE-WORD Guesser in a 20 Questions game.
            Your task: reply with exactly one lowercase English noun.

            Rules:
            1. Only one word, no spaces, no punctuation
            2. Output must be only the word itself, nothing more
            3. Do not explain, justify, or add comments
            4. Do not use parentheses, brackets, or extra text
            5. Never guess generic words like 'thing' or 'based'
            6. Consider these questions, guesses, and answers: {history_text or '-'}
            7. These words are already guessed and wrong: {list(prev_gs) or '-'}
            8. Do not repeat those words, but related nouns are allowed

            Examples of correct output: dog, salt, computer  
            Examples of incorrect output: salt (a compound), dog - animal, computer.  

            Final Output: exactly one lowercase noun only.
            """

        response = ollama.chat(
            model="mistral",
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": history_text or "[start]"}],
            options={"temperature": 0.25}
        )
        content = response["message"]["content"]
        # print("guess generated " , content)
        # m = re.search(r'Guess:\s*(.+)', content, flags=re.IGNORECASE)
        # if not m:
        #     return None  # fallback
        return content.strip()

    def record_turn(self, question, user_answer, guess, guess_correctness):
        """
        Records the events of a single turn, including the user's feedback on the guess.
        """
        self.history.append({
            'question': question,
            'answer': user_answer,
            'guess': guess,
            'guess_feedback': guess_correctness.lower()
        })
        self.questions_asked += 1

        if guess_correctness.lower() == 'yes':
            self.game_over = True
            self.winner = 'ai'
        elif self.questions_asked >= self.max_questions:
            self.game_over = True
            self.winner = 'user'
