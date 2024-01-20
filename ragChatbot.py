import streamlit as st
from getResponse import chain

# Define the color palette
COLORS = {
    "background": "#161A30",
    "text": "#B6BBC4",
    "chat_background": "#31304D",
    "input_background": "#0E0C15",
    "button_background": "#31304D",
    "button_text": "#B6BBC4",
}

# Set page configuration
st.set_page_config(
    page_title="COHERE-POWERED EXAM RAG BOT",
    page_icon=":robot_face:",
    layout="centered",
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS to inject into the Streamlit app
def local_css():
    return f"""
        <style>
            .stApp {{
                background-color: {COLORS["background"]};
            }}
            .css-1d391kg {{
                background-color: {COLORS["chat_background"]};
            }}
            .css-1cpxqw2 {{
                background-color: {COLORS["input_background"]};
                color: {COLORS["text"]};
            }}
            .css-2trqyj {{
                background-color: {COLORS["button_background"]};
                color: {COLORS["button_text"]};
            }}
            .css-hi6a2p {{
                color: {COLORS["text"]};
            }}
        </style>
    """

# Custom CSS for message bubbles
def message_bubble(sender, text):
    bubble_color = COLORS["button_background"] if sender == "user" else COLORS["chat_background"]
    text_color = COLORS["button_text"] if sender == "user" else COLORS["text"]
    return f"""
    <div style="display: flex; justify-content: {'flex-end' if sender == 'user' else 'flex-start'};">
        <div style="max-width: 60%; margin: 10px; padding: 10px; border-radius: 20px; background-color: {bubble_color}; color: {text_color};">
            {text}
        </div>
    </div>
    """


# Inject custom CSS
st.markdown(local_css(), unsafe_allow_html=True)

# Display the title of the app
st.title("COHERE-POWERED EXAM RAG BOT")

# Brief description
description = """
This app utilizes Cohere's powerful language models to provide efficient and accurate answers to O Level Physics questions. 
By leveraging Cohere Embed to create vector embeddings of course notes, Cohere Generate to produce responses, and Cohere ReRank to refine document retrieval, 
we ensure high-quality and prompt assistance. The innovative use of RAG (Retrieval-Augmented Generation) significantly enhances the accuracy and relevance of the information provided.
"""

st.markdown(description, unsafe_allow_html=True)

# Chat area and input
chat_history = []

# Chat area
st.markdown("## Chat with the Physics Expert Bot")
chat_area = st.empty()  # Placeholder for the chat display area

# Chat input
user_input = st.text_input("What's your physics query?", key="user_input")

if st.button("Send"):
    if user_input:  # Check if the input is not empty
        # Add user input to chat history
        st.session_state.chat_history.append({"sender": "user", "text": user_input})
        with st.spinner('Getting your answer...'):
            result = chain(user_input)
            source_contents = '\n\n'.join([f"- {doc.page_content}" for doc in result['source_documents']])
            relevant_sources = ', '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))
            answer = result['answer']
            # Add bot response to chat history
            st.session_state.chat_history.append({"sender": "bot", "text": answer})
            # Show sources in a dropdown if needed
            st.expander("See source texts").write(source_contents + "\n\n" + relevant_sources)
        # Clear the input field
        #st.session_state.user_input = ""

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(message_bubble(chat["sender"], chat["text"]), unsafe_allow_html=True)