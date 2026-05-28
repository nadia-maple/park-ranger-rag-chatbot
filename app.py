import logging
from typing import Dict, List
import streamlit as st
from src.chatbot import ChatbotConfig, HybridRAGChatbot

# ==========================================
# Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================
# Page Configuration & UI Setup
# ==========================================
st.set_page_config(page_title="Park Ranger AI Assistant", page_icon="🌲", layout="centered")
st.title("🌲 Park Ranger AI Assistant")
st.markdown("Ask me anything about National Parks in the US and Canada!")

# ==========================================
# Sidebar: API Key Configuration
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your Google Gemini API Key:", type="password")
    st.markdown("[Get a Gemini API Key](https://aistudio.google.com/app/apikey)")
    clear_clicked = st.button("Clear Chat History")
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This is an unofficial, educational project. "
        "The information provided is sourced from Parks Canada and the US National Park Service "
        "but is not endorsed by or affiliated with either government."
    )

# ==========================================
# Application State & Initialization
# ==========================================
# Initialize chat history in Streamlit session state

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your virtual Park Ranger. How can I help you plan your next adventure?",
            "sources": [],
        }
    ]

if "bot" not in st.session_state:
    st.session_state.bot = None


@st.cache_resource(show_spinner=False)
def load_chatbot(api_key_value: str) -> HybridRAGChatbot:
    config = ChatbotConfig(memory_path=None)
    return HybridRAGChatbot(api_key=api_key_value, config=config)


if clear_clicked:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your virtual Park Ranger. How can I help you plan your next adventure?",
            "sources": [],
        }
    ]
    if st.session_state.bot is not None:
        st.session_state.bot.clear_memory()
    st.success("History cleared!")


if not api_key:
    st.warning("👈 Please enter your Gemini API Key in the sidebar to start.")
    st.stop()


try:
    if st.session_state.bot is None:
        with st.spinner("Waking up the Park Ranger (Loading Database)..."):
            st.session_state.bot = load_chatbot(api_key)
except Exception as exc:
    logger.exception("Failed to initialize chatbot")
    st.error(f"Error loading the chatbot: {exc}")
    st.stop()

# ==========================================
# Chat Interface
# ==========================================
# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View Sources"):
                for index, source in enumerate(msg["sources"], start=1):
                    source_text = source.get("source_page") or source.get("title") or "Unknown Source"
                    url = source.get("source_url") or source.get("url")
                    score = source.get("score", "N/A")
                    if url:
                        st.markdown(f"{index}. [{source_text}]({url}) (Score: {score})")
                    else:
                        st.markdown(f"{index}. {source_text} (Score: {score})")

# User input processing
prompt = st.chat_input("e.g., Do I need a pass for Banff National Park?")
if prompt:
    user_message: Dict[str, List[Dict[str, str]] | str] = {"role": "user", "content": prompt, "sources": []}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    session_memory = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages[-12:]
    ]

# Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching park records..."):
            try:
                result = st.session_state.bot.ask(prompt, memory=session_memory)
                answer = result["answer"]
                sources = result.get("sources", [])
                st.markdown(answer)

                if sources:
                    with st.expander("View Sources"):
                        for index, source in enumerate(sources, start=1):
                            source_text = source.get("source_page") or source.get("title") or "Unknown Source"
                            url = source.get("source_url") or source.get("url")
                            score = source.get("score", "N/A")
                            if url:
                                st.markdown(f"{index}. [{source_text}]({url}) (Score: {score})")
                            else:
                                st.markdown(f"{index}. {source_text} (Score: {score})")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                logger.exception("Chat request failed")
                error_msg = f"Sorry, I encountered an error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )
