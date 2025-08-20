# AI Chat Agent - Deep Learning Project (Fixed 20-Questions flow)
# Complete implementation file

import os
import json
import pickle
import faiss
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import requests
from datetime import datetime
import PyPDF2
from io import BytesIO
import torch
from sentence_transformers import SentenceTransformer
import re
import random

import ollama 
MODEL = "mistral"
# ---- Model generation options

GEN_OPTIONS = {
    "temperature": 0.3,     # lower = more precise, less random
    "num_predict": 512,     # max tokens to generate
    "num_ctx": 4096,        # context window (depends on model)
    "repeat_penalty": 1.1,
}

class ConversationManager:
    """Manages conversation history with context window management"""
   
    def __init__(self, max_context_length: int = 4000):
        self.conversation_history = []
        self.max_context_length = max_context_length
   
    def add_message(self, role: str, content: str):
        """Add a message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._manage_context_window()
   
    def _manage_context_window(self):
        """Manage context window by removing old messages if necessary"""
        total_length = sum(len(msg["content"]) for msg in self.conversation_history)
       
        while total_length > self.max_context_length and len(self.conversation_history) > 2:
            # Remove oldest messages but keep the first system message
            if len(self.conversation_history) > 1 and self.conversation_history[1]["role"] != "system":
                removed = self.conversation_history.pop(1)
                total_length -= len(removed["content"])
            else:
                break
   
    def get_conversation_context(self) -> str:
        """Get formatted conversation context for the model"""
        context = ""
        for msg in self.conversation_history[-10:]:  # Keep last 10 messages
            context += f"{msg['role'].capitalize()}: {msg['content']}\n"
        return context
   
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

class RAGSystem:
    """Retrieval-Augmented Generation that works with both QA-style PDFs and normal text PDFs."""
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.mode = None  # "qa" یا "text"

    def process_pdf(self, pdf_file) -> bool:
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            texts = []
            for p in reader.pages:
                t = p.extract_text() or ""
                texts.append(t.replace("\r", "\n"))
            full_text = "\n".join(texts)

            # 1) تلاش برای Q/A detection
            qa_units = self._extract_qa_units(full_text)
            if qa_units:
                self.mode = "qa"
                self.documents = [f"Q: {u['q']}\nA: {u['a']}" for u in qa_units]
                self.metadatas = qa_units
            else:
                self.mode = "text"
                chunks = self._sliding_chunks(full_text, tokens_per_chunk=180, overlap=40)
                self.documents = chunks
                self.metadatas = [{"text": c} for c in chunks]

            if not self.documents:
                return False

            # 2) امبدینگ + نرمال‌سازی
            self.embeddings = self.embedding_model.encode(
                self.documents,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')

            # 3) FAISS بر اساس cosine similarity
            dim = self.embeddings.shape[1]
            # self.index = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(self.embeddings)
            return True

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return False

    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[str]:
        """Retrieve relevant context (answers if QA, chunks if normal text)."""
        if self.index is None or not self.documents:
            return []

        try:
            q_emb = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
            scores, idxs = self.index.search(q_emb, k)
            idxs = idxs[0].tolist()

            contexts = []
            for i in idxs:
                if 0 <= i < len(self.documents):
                    text = self.documents[i]
                    if self.mode == "qa":
                        ans = self.extract_answer(text)
                        contexts.append(ans or text)
                    else:
                        contexts.append(text)
            return contexts
        except Exception as e:
            print(f"Error in RAG retrieval: {e}")
            return []

    # ------------------ helpers ------------------

    def extract_answer(self, text: str) -> Optional[str]:
        """If it's QA, return only the answer part."""
        m = re.search(r'(?:^|\n)A\s*[:：]\s*(.+)', text, flags=re.IGNORECASE)
        if m: return m.group(1).strip()
        m2 = re.search(r'(?:^|\n)(?:پاسخ|جواب)\s*[:：]\s*(.+)', text, flags=re.IGNORECASE)
        if m2: return m2.group(1).strip()
        return None

    def _extract_qa_units(self, text: str) -> List[Dict[str, Any]]:
        patterns = [
            r'(?:^|\n)\s*(?:Question|Q)\s*[:：]\s*(?P<q>.+?)\n\s*(?:Answer|A)\s*[:：]\s*(?P<a>.+?)(?=\n\s*(?:Question|Q)\s*[:：]|\Z)',
            r'(?:^|\n)\s*(?:سوال|پرسش)\s*[:：]\s*(?P<q>.+?)\n\s*(?:پاسخ|جواب)\s*[:：]\s*(?P<a>.+?)(?=\n\s*(?:سوال|پرسش)\s*[:：]|\Z)',
        ]
        units = []
        for pat in patterns:
            for m in re.finditer(pat, text, flags=re.IGNORECASE | re.DOTALL):
                q = m.group('q').strip()
                a = m.group('a').strip()
                units.append({"q": q, "a": a, "text": f"Q: {q}\nA: {a}"})
        return units

    def _sliding_chunks(self, text: str, tokens_per_chunk=180, overlap=40) -> List[str]:
        words = text.split()
        if not words: return []
        chunks = []
        step = max(1, tokens_per_chunk - overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + tokens_per_chunk]).strip()
            if chunk: chunks.append(chunk)
        return chunks
        
class WebSearchAgent:
    """Web search functionality using Exa API"""
    def __init__(self, exa_api_key: Optional[str] = None):
        # از env هم پشتیبانی کن؛ کلید رو هاردکد نکن
        self.exa_api_key = exa_api_key or os.getenv("EXA_API_KEY")
        self.base_url = "https://api.exa.ai"

    def search_web(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        if not self.exa_api_key:
            return [{
                "title": "Search unavailable",
                "content": "EXA_API_KEY not set. Configure your key.",
                "url": ""
            }]
        try:
            from datetime import datetime, timezone
            now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

            headers = {
                "x-api-key": self.exa_api_key,
                "Content-Type": "application/json"
            }
            data = {
                "query": query,
                "numResults": num_results,
                # فیلتر تاریخ اختیاریه؛ اگر لازم داری نگه دار
                "startCrawlDate": "2024-01-01T00:00:00Z",
                "endCrawlDate": now_utc,
                "startPublishedDate": "2024-01-01T00:00:00Z",
                "endPublishedDate": now_utc,
                # این‌ها خروجی محتوا می‌ده
                "text": True,
                "summary": True
            }

            resp = requests.post(f"{self.base_url}/search", headers=headers, json=data, timeout=15)
            if resp.ok:
                payload = resp.json()
                return payload.get("results", [])[:num_results]
            else:
                return [{
                    "title": "Search failed",
                    "content": f"API {resp.status_code}: {resp.text}",
                    "url": ""
                }]
        except Exception as e:
            return [{"title": "Search error", "content": f"Error: {e}", "url": ""}]

class TwentyQuestionsGame:
    """Implementation of 20 Questions game (phase-based + smarter guessing)
       - Asks ONE yes/no question, then immediately makes a guess.
       - Avoids duplicate/overlapping questions (e.g., indoors vs outdoors).
       - Avoids repeated guesses using a guessed_set.
       - Uses a small knowledge base + constraints to filter candidates and pick high-info questions.
    """
   
    def __init__(self):
        self.reset_game()
        self.common_objects = [
            "dog", "cat", "car", "apple", "book", "chair", "computer", "phone", "tree",
            "house", "ball", "table", "window", "door", "pen", "paper", "water", "fire", 
            "sun", "moon", "star", "flower", "bird", "fish", "guitar", "piano", "bread",
            "coffee", "television", "lamp", "shoe", "clock", "mirror", "knife", "spoon"
        ]

        # Canonical yes/no attributes and their question strings
        self.attr_questions = {
            'alive': "Is it alive?",
            'animal': "Is it an animal?",
            'plant': "Is it a plant?",
            'man_made': "Is it made by humans?",
            'electronic': "Is it electronic?",
            'metal': "Is it made of metal?",
            'edible': "Can you eat it?",
            'tool': "Is it a tool?",
            'furniture': "Is it furniture?",
            'transport': "Is it used for transportation?",
            'kitchen': "Is it found in a kitchen?",
            'work_study': "Is it used for work or study?",
            'moving_parts': "Does it have moving parts?",
            'soft': "Is it soft?",
            'colorful': "Is it colorful?",
            'noisy': "Does it make noise?",
            'expensive': "Is it expensive?",
            'indoors': "Is it found indoors?",
            'outdoors': "Is it found outdoors?",
            'office': "Is it found in offices?",
            'nature': "Is it found in nature?",
            'bigger_breadbox': "Is it bigger than a breadbox?",
            'hand_hold': "Can you hold it in your hand?",
        }

        # Pairs/groups we treat as overlapping → don't ask both
        self.mutual_exclusive = [
            {'indoors', 'outdoors'},  # user complaint: asked both; we avoid asking the pair
        ]

        # Build a tiny knowledge base of objects → attribute booleans
        self.kb = self._build_kb()

    def reset_game(self):
        """Reset game state"""
        self.game_active = False
        self.target_word = None
        self.question_count = 0           # counts only yes/no QUESTIONS (max 20)
        self.max_questions = 20
        self.game_history = []
        self.asked_questions = []         # user-facing strings
        self.learned_info = {}            # map(question_text → 'yes'/'no')
        self.constraints = {}             # map(attr_key → True/False)
        self.asked_keys = set()           # canonical attr keys already asked
        self.forbidden_keys = set()       # blocked by mutual exclusivity
        self.phase = "ask"                # 'ask' or 'judge'
        self.last_guess = None
        self.last_attr = None
        self.guessed_set = set()          # to avoid repeated guesses
        self.candidates = set()           # current viable objects

    def start_game(self, target_word: str = None):
        """Start a new game"""
        self.reset_game()
        self.game_active = True
        self.target_word = target_word or random.choice(self.common_objects)
        # initialize candidate pool
        self.candidates = set(self.kb.keys())
        return ("I'm ready to play 20 Questions! Please think of a word, and I'll try to guess it with up to 20 yes/no questions. "
                "After each answer, I'll make a guess. Ready?")

    def is_game_request(self, message: str) -> bool:
        game_keywords = [
            "20 questions", "twenty questions", "guessing game", "play game",
            "start game", "game", "guess my word", "think of a word", "play 20"
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in game_keywords)

    # ----------------------------- Question selection -----------------------------
    def generate_question(self) -> str:
        """Pick the next best yes/no question using an info-gain heuristic over remaining candidates."""
        # Choose the attribute with split closest to 50/50 among candidates
        attr = self._choose_best_attribute()
        if not attr:
            # ultimate fallback (rare): ask a non-attr question (won't be used for filtering)
            q = f"Does it start with the letter {random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}?"
            self.last_attr = None
            self.asked_questions.append(q)
            return q

        q = self.attr_questions[attr]
        self.last_attr = attr
        self.asked_keys.add(attr)
        # apply mutual exclusivity → forbid sibling attrs
        for group in self.mutual_exclusive:
            if attr in group:
                self.forbidden_keys.update(group - {attr})
        self.asked_questions.append(q)
        return q

    def _choose_best_attribute(self) -> Optional[str]:
        # candidate set fallback
        if not self.candidates:
            self.candidates = set(self.kb.keys()) - self.guessed_set

        best_attr = None
        best_score = 1e9  # we minimize |p-0.5|
        best_known = -1

        for attr, q in self.attr_questions.items():
            if attr in self.asked_keys or attr in self.forbidden_keys:
                continue
            # compute distribution among candidates
            t = f = u = 0
            for obj in self.candidates:
                val = self.kb.get(obj, {}).get(attr, None)
                if val is True: t += 1
                elif val is False: f += 1
                else: u += 1
            known = t + f
            if known == 0:
                continue
            p = t / known
            score = abs(p - 0.5)
            # prefer most balanced; tie-break: more known coverage
            if score < best_score or (score == best_score and known > best_known):
                best_score = score
                best_known = known
                best_attr = attr
        return best_attr

    # ----------------------------- Answer handling -----------------------------
    def process_answer(self, answer: str) -> str:
        """Phase-aware processing.
        - 'ask': user answered our QUESTION → update constraints, filter candidates, then GUESS
        - 'judge': user judged our GUESS → win or ask next question
        """
        ans = answer.lower().strip()
        if ans in ['y', 'yeah', 'yep', 'true', 'correct']:
            ans = 'yes'
        if ans in ['n', 'nope', 'false']:
            ans = 'no'
        if ans not in ['yes', 'no']:
            return "Please answer with 'yes' or 'no'."

        # -------------------- phase: ask --------------------
        if self.phase == 'ask':
            # record question + answer
            self.question_count += 1
            last_q = self.asked_questions[-1] if self.asked_questions else ""
            self.game_history.append(f"Q{self.question_count}: {last_q} - {ans}")
            self.learned_info[last_q] = ans

            # map to attribute & update constraints/candidates
            if self.last_attr is not None:
                self.constraints[self.last_attr] = (ans == 'yes')
                self._refilter_candidates()

            # immediately make a (non-repeated) guess
            self.last_guess = self.make_guess()
            self.phase = 'judge'
            return f"My guess: Is it a {self.last_guess}? (Yes/No)"

        # -------------------- phase: judge --------------------
        if self.phase == 'judge':
            if ans == 'yes':
                self.game_active = False
                return "Great! I guessed it correctly! 🎉 Game won! Say 'play game' to start again."

            # wrong guess → remember so we don't repeat
            if self.last_guess:
                self.guessed_set.add(self.last_guess)
                if self.last_guess in self.candidates:
                    self.candidates.remove(self.last_guess)

            # out of questions?
            if self.question_count >= self.max_questions:
                self.game_active = False
                return "I've used all 20 questions and still missed it. You win! What was your word?"

            # continue with next question
            next_q = self.generate_question()
            self.phase = 'ask'
            return f"Question {self.question_count + 1}: {next_q} (Yes/No)"

        return "Let's continue. Please answer with yes or no."

    # ----------------------------- Guessing logic -----------------------------
    def make_guess(self) -> str:
        """Pick the best candidate consistent with constraints, avoiding repeats."""
        # Build a list of viable candidates not yet guessed
        viable = [o for o in (self.candidates or self.kb.keys()) if o not in self.guessed_set]
        if not viable:
            # fall back to anything unguessed in KB
            viable = [o for o in self.kb.keys() if o not in self.guessed_set]
        if not viable:
            return "idea"

        # score by how many constraints they satisfy (hard-filter first)
        strict = []
        for o in viable:
            attrs = self.kb.get(o, {})
            ok = True
            for a, v in self.constraints.items():
                val = attrs.get(a, None)
                if val is not None and val != v:
                    ok = False
                    break
            if ok:
                strict.append(o)
        pool = strict if strict else viable

        def score(o: str) -> Tuple[int, int]:
            attrs = self.kb.get(o, {})
            matches = 0
            known = 0
            for a, v in self.constraints.items():
                val = attrs.get(a, None)
                if val is not None:
                    known += 1
                    if val == v:
                        matches += 1
            # prefer more matches, then more known coverage
            return (matches, known)

        best = sorted(pool, key=lambda o: (score(o)[0], score(o)[1], -len(o)), reverse=True)
        return best[0]

    def end_game(self, result: str = "ended"):
        self.game_active = False
        return f"Game {result}! Thanks for playing. Say 'play game' or 'start game' to start a new round."

    # ----------------------------- Helpers -----------------------------
    def _refilter_candidates(self):
        if not self.candidates:
            self.candidates = set(self.kb.keys())
        kept = set()
        for o in self.candidates:
            attrs = self.kb.get(o, {})
            consistent = True
            for a, v in self.constraints.items():
                val = attrs.get(a, None)
                if val is not None and val != v:
                    consistent = False
                    break
            if consistent:
                kept.add(o)
        self.candidates = kept if kept else set(self.kb.keys()) - self.guessed_set

    def _build_kb(self) -> Dict[str, Dict[str, Optional[bool]]]:
        T, F = True, False
        kb = {
            # animals / plants
            'dog':   {'alive': T, 'animal': T, 'plant': F, 'man_made': F, 'electronic': F, 'edible': F, 'tool': F, 'furniture': F,
                      'moving_parts': T, 'soft': T, 'colorful': F, 'noisy': T, 'expensive': T,
                      'indoors': T, 'outdoors': T, 'office': F, 'nature': F,
                      'bigger_breadbox': T, 'hand_hold': F},
            'cat':   {'alive': T, 'animal': T, 'plant': F, 'man_made': F, 'electronic': F, 'edible': F, 'tool': F, 'furniture': F,
                      'moving_parts': T, 'soft': T, 'colorful': F, 'noisy': F, 'expensive': T,
                      'indoors': T, 'outdoors': T, 'office': F, 'nature': F,
                      'bigger_breadbox': T, 'hand_hold': F},
            'bird':  {'alive': T, 'animal': T, 'plant': F, 'man_made': F, 'electronic': F, 'edible': F, 'tool': F, 'furniture': F,
                      'moving_parts': T, 'soft': F, 'colorful': T, 'noisy': T, 'expensive': F,
                      'indoors': F, 'outdoors': T, 'office': F, 'nature': T,
                      'bigger_breadbox': F, 'hand_hold': F},
            'fish':  {'alive': T, 'animal': T, 'plant': F, 'man_made': F, 'electronic': F, 'edible': F,
                      'indoors': T, 'outdoors': T, 'nature': T,
                      'bigger_breadbox': F, 'hand_hold': F},
            'tree':  {'alive': T, 'animal': F, 'plant': T, 'man_made': F, 'electronic': F, 'edible': F,
                      'indoors': F, 'outdoors': T, 'nature': T,
                      'bigger_breadbox': T, 'hand_hold': F},
            'flower':{'alive': T, 'plant': T, 'animal': F, 'man_made': F, 'electronic': F, 'edible': F,
                      'indoors': T, 'outdoors': T, 'nature': T,
                      'bigger_breadbox': F, 'hand_hold': T, 'colorful': T},

            # man-made objects
            'car':   {'alive': F, 'man_made': T, 'electronic': T, 'metal': T, 'transport': T, 'moving_parts': T,
                      'indoors': F, 'outdoors': T, 'office': F, 'nature': F,
                      'bigger_breadbox': T, 'hand_hold': F, 'noisy': T, 'expensive': T},
            'chair': {'man_made': T, 'furniture': T, 'indoors': T, 'outdoors': T, 'office': T,
                      'bigger_breadbox': T, 'hand_hold': F},
            'table': {'man_made': T, 'furniture': T, 'indoors': T, 'outdoors': T, 'office': T,
                      'bigger_breadbox': T, 'hand_hold': F},
            'computer': {'man_made': T, 'electronic': T, 'work_study': T, 'indoors': T, 'office': T,
                        'bigger_breadbox': T, 'hand_hold': F, 'expensive': T},
            'phone': {'man_made': T, 'electronic': T, 'work_study': T, 'indoors': T, 'outdoors': T,
                      'hand_hold': T, 'bigger_breadbox': F, 'noisy': T, 'expensive': T},
            'guitar':{'man_made': T, 'electronic': F, 'indoors': T, 'outdoors': T,
                      'hand_hold': T, 'bigger_breadbox': T, 'noisy': T},
            'piano': {'man_made': T, 'electronic': F, 'indoors': T, 'outdoors': F,
                      'hand_hold': F, 'bigger_breadbox': T, 'noisy': T, 'expensive': T},
            'television': {'man_made': T, 'electronic': T, 'indoors': T, 'office': T,
                           'bigger_breadbox': T, 'hand_hold': F, 'noisy': F, 'expensive': T},
            'lamp':  {'man_made': T, 'electronic': T, 'indoors': T, 'office': T,
                      'hand_hold': F, 'bigger_breadbox': F},
            'clock': {'man_made': T, 'electronic': F, 'indoors': T, 'office': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'mirror':{'man_made': T, 'indoors': T, 'office': T,
                      'hand_hold': F, 'bigger_breadbox': F},
            'knife': {'man_made': T, 'metal': T, 'kitchen': T, 'indoors': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'spoon': {'man_made': T, 'metal': T, 'kitchen': T, 'indoors': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'pen':   {'man_made': T, 'work_study': T, 'indoors': T, 'office': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'paper': {'man_made': T, 'work_study': T, 'indoors': T, 'office': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'book':  {'man_made': T, 'work_study': T, 'indoors': T, 'office': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'shoe':  {'man_made': T, 'indoors': T, 'outdoors': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'window':{'man_made': T, 'indoors': T, 'outdoors': T,
                      'hand_hold': F, 'bigger_breadbox': T},
            'door':  {'man_made': T, 'indoors': T, 'outdoors': T,
                      'hand_hold': F, 'bigger_breadbox': T},

            # food/drink
            'apple': {'plant': T, 'alive': F, 'edible': T, 'man_made': F, 'indoors': T, 'outdoors': T, 'kitchen': T,
                      'hand_hold': T, 'bigger_breadbox': F, 'colorful': T},
            'bread': {'edible': T, 'man_made': T, 'kitchen': T, 'indoors': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'coffee':{'edible': T, 'man_made': T, 'kitchen': T, 'indoors': T,
                      'hand_hold': T, 'bigger_breadbox': F},

            # elements / nature
            'water': {'alive': F, 'man_made': F, 'nature': T, 'indoors': T, 'outdoors': T,
                      'hand_hold': T, 'bigger_breadbox': F},
            'fire':  {'alive': F, 'man_made': F, 'nature': T, 'indoors': T, 'outdoors': T,
                      'hand_hold': F, 'bigger_breadbox': T},
            'sun':   {'alive': F, 'nature': T, 'outdoors': T, 'bigger_breadbox': T, 'hand_hold': F},
            'moon':  {'alive': F, 'nature': T, 'outdoors': T, 'bigger_breadbox': T, 'hand_hold': F},
            'star':  {'alive': F, 'nature': T, 'outdoors': T, 'bigger_breadbox': T, 'hand_hold': F},
            'house': {'man_made': T, 'outdoors': T, 'bigger_breadbox': T, 'indoors': T},
            'ball':  {'man_made': T, 'indoors': T, 'outdoors': T, 'hand_hold': T, 'bigger_breadbox': F},
        }
        return kb

class LLMModel:
    """Wrapper for language model using Ollama"""
   
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name
        self.ollama_available = self._check_ollama()
        
        if self.ollama_available:
            print(f"Using Ollama with model: {self.model_name}")
        else:
            print("Ollama not available, using fallback responses")
   
    def _check_ollama(self) -> bool:
        """Check if Ollama is available and the model exists"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"].split(":")[0] for model in models]
                if self.model_name in available_models:
                    return True
                else:
                    print(f"Model {self.model_name} not found. Available models: {available_models}")
                    return False
            return False
        except Exception as e:
            print(f"Ollama not available: {e}")
            return False
   
    def generate_response(self, prompt: str, max_length: int = 150) -> str:
        """Generate response using Ollama"""
        if not self.ollama_available:
            return self._fallback_response(prompt)
       
        try:
            import requests
            
            # Clean and format the prompt for better results
            cleaned_prompt = self._clean_prompt(prompt)
            
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": max_length,
                    "stop": ["User:", "Human:", "\n\nUser:", "\n\nHuman:"]
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()
                # Clean up the response
                cleaned_response = self._clean_response(generated_text)
                return cleaned_response if cleaned_response else self._fallback_response(prompt)
            else:
                print(f"Ollama API error: {response.status_code}")
                return self._fallback_response(prompt)
                
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return self._fallback_response(prompt)
           
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and optimize prompt for Mistral"""
        # Create a cleaner prompt structure for Mistral
        lines = prompt.strip().split('\n')
        
        # Extract the actual conversation
        conversation_lines = []
        current_context = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Current information from web search:"):
                current_context.append(line)
            elif line.startswith("Relevant information from uploaded document:"):
                current_context.append(line)
            elif line.startswith("User:") or line.startswith("Assistant:"):
                conversation_lines.append(line)
        
        # Build the final prompt
        final_prompt = ""
        
        # Add context if available
        if current_context:
            final_prompt += "\n".join(current_context) + "\n\n"
        
        # Add conversation
        if conversation_lines:
            final_prompt += "\n".join(conversation_lines)
        else:
            final_prompt += prompt
            
        return final_prompt
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove common artifacts
        response = response.replace("Assistant:", "").strip()
        response = response.replace("AI:", "").strip()
        
        # Remove any text that appears to be continuing a conversation incorrectly
        unwanted_starts = [
            "navigated", "User :", "my name is", "hello navigated",
            "Thanks! I'll take a look", "Let me analyze", "Looking at"
        ]
        
        response_lower = response.lower()
        for unwanted in unwanted_starts:
            if response_lower.startswith(unwanted.lower()):
                return ""
        
        # Split on user indicators and take first part
        for delimiter in ["\nUser:", "\nHuman:", "User:", "Human:"]:
            if delimiter in response:
                response = response.split(delimiter)[0]
        
        # Remove extra whitespace and newlines
        response = response.strip()
        
        # If response is too short, return empty to trigger fallback
        if len(response) < 2:
            return ""
            
        return response
   
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response system when Ollama is unavailable"""
        # Extract the actual user message from the prompt
        user_message = ""
        if "User:" in prompt:
            user_message = prompt.split("User:")[-1].replace("Assistant:", "").strip()
        
        user_message_lower = user_message.lower()
       
        if any(greet in user_message_lower for greet in ["hello", "hi", "hey", "good morning", "good afternoon"]):
            return "Hello! How can I help you today?"
        elif "how are you" in user_message_lower:
            return "I'm doing well, thank you for asking! How can I assist you?"
        elif "my name is" in user_message_lower:
            # Extract name
            name_part = user_message_lower.split("my name is")[-1].strip()
            name = name_part.split()[0] if name_part.split() else "there"
            return f"Nice to meet you, {name.capitalize()}! How can I help you today?"
        elif any(word in user_message_lower for word in ["what is", "explain", "tell me about"]):
            return "I'd be happy to help explain that topic. Could you be more specific about what you'd like to know?"
        elif any(word in user_message_lower for word in ["thank you", "thanks"]):
            return "You're welcome! Is there anything else I can help you with?"
        elif "game" in user_message_lower:
            return "Would you like to play a game? I can play 20 Questions with you!"
        elif "help" in user_message_lower:
            return "I'm here to help! You can ask me questions, upload a PDF for document analysis, request current information from the web, or play 20 Questions with me."
        else:
            return "I understand. How can I help you with that?"

class AIChatAgent:
    """Main AI Chat Agent class integrating all components"""
   
    def __init__(self):
        self.conversation_manager = ConversationManager()
        self.rag_system = RAGSystem()
        self.web_search = WebSearchAgent()
        self.game = TwentyQuestionsGame()
        self.llm = LLMModel("mistral")  # Using Mistral with Ollama
        
        # Add initial system message
        self.conversation_manager.add_message(
            "system", 
            "You are a helpful AI assistant."
        )
         
    
    def should_search_web(self, query: str) -> bool:
        instr = ("You are a binary classifier. If answering the following query "
                "needs up-to-date external info from the web, reply exactly 'yes'; "
                "otherwise reply exactly 'no'.")
        out = self.llm.generate_response(f"{instr}\n\nQuery: {query}\nAnswer:", max_length=3).strip().lower()
        return out.startswith("y")
    

    def process_message(self, user_message: str) -> str:
        """Process user message and generate appropriate response"""
        # Add user message to conversation history
        self.conversation_manager.add_message("user", user_message)
       
        # Check if user wants to start/continue the game
        if self.game.is_game_request(user_message) and not self.game.game_active:
            intro = self.game.start_game()
            first_q = self.game.generate_question()
            response = intro + "\n\n" + f"Question 1: {first_q} (Yes/No)"
            self.conversation_manager.add_message("assistant", response)
            return response
       
        # Handle game interactions
        if self.game.game_active:
            # stop/quit
            if any(word in user_message.lower() for word in ["quit", "stop", "exit", "end game"]):
                response = self.game.end_game("stopped")
                self.conversation_manager.add_message("assistant", response)
                return response

            # explicit 'correct' shortcut works during judge phase
            if "correct" in user_message.lower() and self.game.phase == 'judge':
                response = "Great! I guessed it correctly! 🎉 " + self.game.end_game("won")
                self.conversation_manager.add_message("assistant", response)
                return response

            # route all other inputs to phase-aware processor
            response = self.game.process_answer(user_message)
            self.conversation_manager.add_message("assistant", response)
            return response
       
        # Check if web search is needed
        search_context = ""
        if self.should_search_web(user_message):
            search_results = self.web_search.search_web(user_message)
            search_context = self._format_search_results(search_results)
       
        # Get RAG context if available
        rag_context = ""
        if self.rag_system.index is not None:
            results = self.rag_system.retrieve_relevant_context(user_message, k=3)
            if results:
                rag_context = "Relevant information from uploaded document:\n" + "\n---\n".join(results)

        # Build prompt for LLM
        prompt = self._build_prompt(user_message, search_context, rag_context)
       
        # Generate response
        response = self.llm.generate_response(prompt)
       
        # Add response to conversation history
        self.conversation_manager.add_message("assistant", response)
       
        return response
   
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return ""
        context = "Current information from web search:\n"
        for i, r in enumerate(results[:3], 1):
            title = r.get("title") or r.get("url", "No title")
            content = r.get("summary") or r.get("text")
            if not content:
                highs = r.get("highlights") or []
                content = highs[0] if highs else ""
            if content:
                content = (content[:200] + "...") if len(content) > 200 else content
            context += f"{i}. {title}: {content}\n"
        return context

    def _build_prompt(self, user_message: str, search_context: str, rag_context: str) -> str:
        """Build comprehensive prompt for the language model"""
        
        # Build a cleaner prompt for Mistral
        prompt_parts = []
        
        # Add search context if available
        if search_context:
            prompt_parts.append(search_context)
        
        # Add RAG context if available
        if rag_context:
            prompt_parts.append(rag_context)
        
        # Get only recent conversation history (last 3 exchanges)
        recent_history = []
        history = self.conversation_manager.conversation_history
        
        # Skip system message and get user-assistant pairs
        for i in range(len(history) - 6, len(history) - 1):  # Last 3 exchanges
            if i >= 1:  # Skip system message
                msg = history[i]
                if msg["role"] in ["user", "assistant"]:
                    recent_history.append(msg)
        
        # Build conversation context
        conversation_parts = []
        for msg in recent_history:
            if msg["role"] == "user":
                conversation_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                conversation_parts.append(f"Assistant: {msg['content']}")
        
        # Add current user message
        conversation_parts.append(f"User: {user_message}")
        conversation_parts.append("Assistant:")
        
        # Combine all parts
        if prompt_parts:
            context = "\n\n".join(prompt_parts)
            full_prompt = f"{context}\n\n" + "\n".join(conversation_parts)
        else:
            full_prompt = "\n".join(conversation_parts)
        return full_prompt
