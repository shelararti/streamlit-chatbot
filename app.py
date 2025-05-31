import streamlit as st
from chat import get_response
from datetime import datetime

st.set_page_config(page_title="Chatbot", page_icon="🤖")

# ✅ Initialize session state variables
if "sessions" not in st.session_state:
    st.session_state.sessions = []
    st.session_state.chat_history = []
    st.session_state.current_session = None

# 🔄 Start first session if none exists
if st.session_state.current_session is None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_session = {"title": f"Chat @ {timestamp}", "history": []}
    st.session_state.sessions.append(new_session)
    st.session_state.current_session = 0
    st.session_state.chat_history = []

# 📚 Sidebar for managing sessions
st.sidebar.title("🕘 Previous Conversations")

# ➕ Start a new conversation
if st.sidebar.button("➕ Start New Conversation"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_session = {"title": f"Chat @ {timestamp}", "history": []}
    st.session_state.sessions.append(new_session)
    st.session_state.chat_history = []
    st.session_state.current_session = len(st.session_state.sessions) - 1

# 📋 List previous sessions
for i, session in enumerate(st.session_state.sessions):
    label = session["title"]
    if i == st.session_state.current_session:
        label = f"✅ {label}"
    if st.sidebar.button(label, key=f"session_{i}"):
        st.session_state.chat_history = session["history"]
        st.session_state.current_session = i

# 🧠 Main Chat UI
st.title("🤖 Simple Chatbot")
st.markdown("Ask anything!")

# Display previous messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box (auto-clearing)
user_input = st.chat_input("Ask something...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get bot response
    response = get_response(user_input)

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Save to session history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Update stored session
    st.session_state.sessions[st.session_state.current_session]["history"] = st.session_state.chat_history
