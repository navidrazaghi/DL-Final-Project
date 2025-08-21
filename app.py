# app.py 

import streamlit as st
import ollama
import os
from rag_handler import create_or_load_rag_index, get_relevant_context
from function_caller import get_intent, search_web
from game_logic import Game20Q

# --- Page Configuration ---
st.set_page_config(page_title="AI Chat Agent", page_icon="🤖", layout="wide")
st.title("🤖 AI Chat Agent Project")
st.caption("A Deep Learning Project from Sharif University of Technology")

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful and conversational AI assistant. Remember and use information from previous turns in the conversation."}]
if "rag_db" not in st.session_state:
    st.session_state.rag_db = None
if "game_mode" not in st.session_state:
    st.session_state.game_mode = False
if "game" not in st.session_state:
    st.session_state.game = Game20Q()
# New state to manage the turn-based logic of the 20 Questions game
if "game_turn_state" not in st.session_state:
    st.session_state.game_turn_state = "START" # States: START, AWAITING_QUESTION_ANSWER, AWAITING_GUESS_ANSWER

# --- Helper Functions ---
def reset_conversation():
    """Resets the chat history and game state."""
    st.session_state.messages = [{"role": "system", "content": "You are a helpful and conversational AI assistant."}]
    st.session_state.game_mode = False
    st.session_state.game.reset()
    st.session_state.game_turn_state = "START"
    st.success("Conversation and game state have been reset!")

def generate_chat_response(history, latest_prompt, context=""):
    """Generates a response for standard chat and RAG, using conversation history."""
    messages_for_api = list(history)
    full_prompt = f"Use the following context if relevant:\n---\n{context}\n---\n\nUser Query: {latest_prompt}" if context else latest_prompt
    messages_for_api.append({"role": "user", "content": full_prompt})
    
    response = ollama.chat(model='mistral', messages=messages_for_api)
    return response['message']['content']

# --- Sidebar ---
with st.sidebar:
    st.header("Controls & Settings")
    st.subheader("📄 RAG Document")
    uploaded_file = st.file_uploader("Upload a PDF to chat with it", type="pdf")
    if uploaded_file is not None:
        # Simple caching mechanism
        if "last_uploaded_filename" not in st.session_state or st.session_state.last_uploaded_filename != uploaded_file.name:
            st.session_state.last_uploaded_filename = uploaded_file.name
            with open("temp_doc.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Processing PDF... This may take a moment."):
                st.session_state.rag_db = create_or_load_rag_index("temp_doc.pdf")
            st.success("PDF processed successfully!")
        elif st.session_state.rag_db is None:
             st.session_state.rag_db = create_or_load_rag_index("temp_doc.pdf")

    st.divider()
    st.button("Clear Chat History", on_click=reset_conversation)

# --- Main Chat Interface ---
# Display chat messages from history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

import re
# Handle user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        # ------------------------------------------------------------------
        # --- 20 Questions Game Logic ---
        # ------------------------------------------------------------------
        if st.session_state.game_mode:
            game = st.session_state.game
            if(re.search(r"\b(quit|exit|cancel|stop|end game)\b",
                              prompt, flags = re.I)):
                # Reset game state for the next time
                    st.session_state.game_mode = False
                    st.session_state.game_turn_state = "START"
                    game.reset()

            # State 1: User has just agreed to play. Bot asks the first question.
            if st.session_state.game_turn_state == "START":
                with st.spinner("I'm thinking of my first question..."):

                    question = game.generate_question()
                    st.session_state.current_question = question
                    response = f"Great! Let's begin.\n\n**Question {game.questions_asked + 1}:** {question}"
                    st.session_state.game_turn_state = "AWAITING_QUESTION_ANSWER"

            # State 2: Bot has asked a question, user's prompt is the answer.
            elif st.session_state.game_turn_state == "AWAITING_QUESTION_ANSWER":
                st.session_state.last_answer = prompt  # Save the answer to the question
                current_guess =  game.generate_guess(st.session_state.current_question, st.session_state.last_answer)
                st.session_state.current_guess = current_guess
                response = f"Got it. Now, is the word you're thinking of **'{current_guess}'**? (yes/no)"
                st.session_state.game_turn_state = "AWAITING_GUESS_ANSWER"

            # State 3: Bot is waiting for guess validation.
            elif st.session_state.game_turn_state == "AWAITING_GUESS_ANSWER":
                guess_correctness = prompt
                
                # Record the full turn with all the info
                game.record_turn(
                    question=st.session_state.current_question,
                    user_answer=st.session_state.last_answer,
                    guess=st.session_state.current_guess,
                    guess_correctness=guess_correctness
                )

                # Check if the game has ended
                if game.game_over:
                    if game.winner == 'ai':
                        response = f"I win! The word was **'{st.session_state.current_guess}'**. Great game!"
                    else:  # User wins
                        response = "You've stumped me after 20 questions! You win!"
                    
                    # Reset game state for the next time
                    st.session_state.game_mode = False
                    st.session_state.game_turn_state = "START"
                    game.reset()
                else:
                    # If game continues, generate the next question
                    with st.spinner("Thinking of my next question..."):

                        question = game.generate_question()
                        st.session_state.current_question = question
                        response = f"Okay, next round.\n\n**Question {game.questions_asked + 1}:** {question}"
                        st.session_state.game_turn_state = "AWAITING_QUESTION_ANSWER"

        # ------------------------------------------------------------------
        # --- Standard Chat / RAG / Web Search Logic ---
        # ------------------------------------------------------------------
        else:
            with st.spinner("Analyzing intent..."):
                intent = get_intent(prompt)
                st.info(f"Detected Intent: **{intent}**")

            if intent == 'web_search':
                with st.spinner("Searching the web and summarizing..."):
                    search_results_context = search_web(prompt)
                    summarization_prompt = f"Based ONLY on the following search results, write a concise summary to answer the user's request about '{prompt}'. Synthesize the information into a coherent paragraph.\n\nSearch Results:\n---\n{search_results_context}\n---"
                    synthesis_response = ollama.chat(
                        model='mistral',
                        messages=[{'role': 'user', 'content': summarization_prompt}]
                    )
                    response = synthesis_response['message']['content']
            
            elif intent == 'play_game':
                st.session_state.game_mode = True
                st.session_state.game.reset()
                st.session_state.game_turn_state = "START"  # Set the initial state for the game
                response = st.session_state.game.get_initial_message()

            else: # 'chat' intent (includes RAG)
                with st.spinner("Generating response..."):
                    context = ""
                    if st.session_state.rag_db:
                        context = get_relevant_context(prompt, st.session_state.rag_db)
                    response = generate_chat_response(st.session_state.messages, prompt, context=context)
        
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})