import streamlit as st
from medai_bot import SessionState, bot_reply, menu

st.set_page_config(
    page_title="MedAI Assistant",
    page_icon="🏥",
    layout="centered"
)

# Page title
st.title("🏥 MedAI Virtual Health Assistant")

st.write(
    "An AI-powered patient support chatbot that helps users book appointments, "
    "get symptom guidance, receive medication reminders, and access healthcare information."
)

# Sidebar
with st.sidebar:
    st.header("About MedAI")

    st.write(
        "MedAI is an AI-driven healthcare chatbot designed to assist patients "
        "with appointment booking, symptom guidance, medication reminders, "
        "and healthcare FAQs."
    )

    st.write("Technology Stack:")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- NLP")
    st.write("- TF-IDF")
    st.write("- Scikit-learn")

# Initialize session state
if "state" not in st.session_state:
    st.session_state.state = SessionState()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "🏥 Hello! Welcome to MedAI.\n\n" + menu()}
    ]

# Chat launcher toggle
show_chat = st.toggle("💬 Open MedAI Chat", value=True)

if show_chat:

    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.state = SessionState()
        st.session_state.messages = [
            {"role": "assistant", "content": "🏥 Hello! Welcome to MedAI.\n\n" + menu()}
        ]
        st.rerun()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Type your message")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Bot thinking animation
        with st.spinner("MedAI is thinking..."):
            reply = bot_reply(prompt, st.session_state.state)

        st.session_state.messages.append({"role": "assistant", "content": reply})

        with st.chat_message("assistant"):
            st.markdown(reply)

# Footer
st.markdown("---")

st.markdown(
    "**MedAI Virtual Health Assistant**  \n"
    "Developed using Python, Streamlit, and Natural Language Processing (NLP)."
)