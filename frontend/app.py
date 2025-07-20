# frontend/app.py

import streamlit as st
import base64
from pathlib import Path
import sys
import uuid

# --- PATH SETUP ---
try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from backend.core.rag.agent import app_graph
    from backend.core.schemas import AskResponse, SourceDocument
except ImportError:
    st.error("Could not import backend modules. Ensure the structure is correct and dependencies are installed.")
    st.stop()

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(page_title="Maths AI Tutor", page_icon="🧠", layout="wide")

st.title("Maths AI TA 🧠🤖")
st.caption("Your personal AI assistant for the probability course. Ask a question or upload a picture of your work for a hint!")

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# --- RENDER CHAT HISTORY FIRST ---
# This loop now just displays what's already in the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in the assistant's message
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                sources_list = [SourceDocument(**s) for s in message["sources"]]
                for source in sources_list:
                    st.write(f"**{source.source_type.capitalize()}:** {source.location}")
                    if source.url:
                        st.write(f"🔗 [Link]({source.url})")

# --- RENDER INPUT WIDGETS AT THE BOTTOM ---
# st.chat_input will automatically stick to the bottom of the screen.

uploaded_file = st.file_uploader(
    "Upload an image of your work to ask a question about it:",
    type=["png", "jpg", "jpeg"]
)
prompt = st.chat_input("Ask a question...")

if prompt:
    # --- 1. HANDLE USER INPUT ---
    # Add user's message to history and display it immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 2. GENERATE AND DISPLAY AI RESPONSE ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            image_data = None
            if uploaded_file is not None:
                # Display the image the user uploaded
                st.image(uploaded_file, caption="Image submitted with your question.", width=200)
                image_data = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

            # Prepare inputs for the agent
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            inputs = {"query": prompt, "image_data": image_data, "chat_history": []}

            try:
                # Get the final state from the agent
                final_state = app_graph.invoke(inputs, config=config)
                full_conversation_state = app_graph.get_state(config)
                
                # Extract data from the final state
                response_data = AskResponse(
                    answer=full_conversation_state.values.get("response", "Sorry, I encountered an issue."),
                    sources=full_conversation_state.values.get("sources", [])
                )
                
                # Display the main answer
                st.markdown(response_data.answer)
                
                # Display the sources in an expander
                if response_data.sources:
                    with st.expander("View Sources"):
                        for source in response_data.sources:
                            st.write(f"**{source.source_type.capitalize()}:** {source.location}")
                            if source.url:
                                st.write(f"🔗 [Link]({source.url})")
                
                # --- 3. ADD AI RESPONSE TO SESSION STATE ---
                # Convert Pydantic models to dicts for JSON serialization in session state
                sources_as_dicts = [s.dict() for s in response_data.sources]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_data.answer,
                    "sources": sources_as_dicts
                })

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
                import traceback
                st.exception(traceback.format_exc())

    # Rerun the script to clear the input box and show the new message
    st.rerun()