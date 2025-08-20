
import streamlit as st
import os
import sys
from datetime import datetime

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_chat_agent import AIChatAgent
import requests

def notify_slack(text: str):
    webhook = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook: return
    try:
        requests.post(webhook, json={"text": text}, timeout=5)
    except Exception:
        pass




def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'chat_agent' not in st.session_state:
        st.session_state.chat_agent = AIChatAgent()
        st.session_state.chat_agent.web_search.exa_api_key = "14f915e9-2fcb-4b59-b839-c027be34d922"
    if 'messages' not in st.session_state:
        st.session_state.messages = []
   
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
def main():
    st.set_page_config(
        page_title="AI Chat Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
   
    # Initialize session state
    initialize_session_state()
   
    # Sidebar for configuration and file upload
    with st.sidebar:

       
        # PDF Upload Section
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF document for RAG",
            type="pdf",
            help="Upload a PDF to enable document-based question answering"
        )
       
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    success = st.session_state.chat_agent.rag_system.process_pdf(uploaded_file)
                    if success:
                        st.session_state.pdf_processed = True
                        st.success("PDF processed successfully!")
                    else:
                        st.error("Failed to process PDF")
       
        if st.session_state.pdf_processed:
            st.success("✅ PDF is loaded and ready for questions")
       
        st.divider()
       
        # Game Status
        st.subheader("🎮 Game Status")
        if st.session_state.chat_agent.game.game_active:
            st.info(f"🎯 20 Questions Game Active")
            st.write(f"Question: {st.session_state.chat_agent.game.question_count}/20")
            if st.button("End Game"):
                response = st.session_state.chat_agent.game.end_game("ended")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
                st.rerun()
        else:
            st.write("No active game")
            if st.button("Start 20 Questions"):
                response = st.session_state.chat_agent.game.start_game()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now()
                })
                st.rerun()
       
        st.divider()
       
        # Clear Chat
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.session_state.chat_agent.conversation_manager.clear_history()
            st.session_state.chat_agent.game.reset_game()
            st.rerun()
   
    # Main chat interface
    st.title("🤖 AI Chat Agent")
    st.write("A comprehensive AI assistant with conversation memory, RAG, web search, and 20 Questions game!")
   
    # Display feature status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💬 Conversation", "Active", delta="Memory enabled")
    with col2:
        status = "Ready" if st.session_state.pdf_processed else "No PDF"
        st.metric("📚 RAG System", status)
    with col3:
        web_status = "Ready" 
        st.metric("🌐 Web Search", web_status)
    with col4:
        game_status = "Playing" if st.session_state.chat_agent.game.game_active else "Ready"
        st.metric("🎮 20 Questions", game_status)
   
    st.divider()
   
    # Chat messages display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "timestamp" in message:
                    st.caption(f"Time: {message['timestamp'].strftime('%H:%M:%S')}")
   
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to display
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
       
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")
       
        # Process message and get response
        with st.spinner("Thinking..."):
            response = st.session_state.chat_agent.process_message(prompt)
            notify_slack(f"✅ Chatbot replied at {datetime.now().isoformat()}:\n{response[:400]}")

        # Add assistant response to display
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
       
        # Display assistant response
        with st.chat_message("assistant"):
            st.write(response)
            st.caption(f"Time: {datetime.now().strftime('%H:%M:%S')}")
       
        # Rerun to update the display
        st.rerun()
   
    # Instructions and examples
    with st.expander("📝 How to Use"):
        st.write("""
        **Available Features:**
       
        1. **💬 Natural Conversation**: Just chat naturally - the agent remembers our conversation!
       
        2. **📚 Document Q&A**: Upload a PDF in the sidebar and ask questions about its content.
       
        3. **🌐 Web Search**: Ask about current events, news, weather, prices, etc.
           - Examples: "What's the latest news?", "Current weather", "Stock prices"
       
        4. **🎮 20 Questions Game**: Say "play game" or "20 questions" to start.
           - Think of a word and answer yes/no to my questions!
       
        **Example Queries:**
        - "Hello, how are you?"
        - "What's in the uploaded document about machine learning?"
        - "What's the latest news about AI?"
        - "Let's play 20 questions!"
        - "What's the weather like today?"
        """)
   
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.write("""
        **System Components:**
        - **LLM**: Mistral 7B via Ollama (can be configured for other models up to 7B parameters)
        - **RAG**: FAISS + SentenceTransformers for document retrieval
        - **Web Search**: EXA API integration
        - **Embeddings**: all-MiniLM-L6-v2 for document processing
        - **Context Management**: Automatic conversation history management
       
        **Architecture:**
        - Modular design with separate classes for each functionality
        - Streamlit-based user interface
        - Memory-efficient context window management
        - Real-time response generation
        """)

if __name__ == "__main__":
    main()