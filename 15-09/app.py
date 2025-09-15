import streamlit as st
import os
from openai import AzureOpenAI

model_name = "gpt-4o"
deployment = "gpt-4o"
api_version = "2024-12-01-preview"

def test_azure_connection(endpoint, api_key):
    """Test the Azure OpenAI connection"""
    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        # Test with a simple request
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=10,
            model=deployment
        )
        return True, client
    except Exception as e:
        return False, str(e)

def get_ai_response_stream(history, client):
    """Get AI response with streaming"""
    try:
        # Prepare messages: add system message only if not already present
        messages = []
        
        # Check if the first message is already a system message
        if not history or history[0].get("role") != "system":
            messages.append({"role": "system", "content": "You are a helpful assistant."})

        # Add all messages from history
        messages.extend(history)
        
        stream = client.chat.completions.create(
            messages=messages,
            max_tokens=4096,
            temperature=0.7,
            top_p=1.0,
            model=deployment,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error communicating with AI: {str(e)}"

def configuration_page():
    """Configuration page for Azure credentials"""
    st.title("🔧 Azure OpenAI Configuration")
    st.markdown("Please provide your Azure OpenAI credentials to start chatting.")
    
    with st.form("azure_config"):
        st.subheader("Azure OpenAI Settings")
        
        endpoint = st.text_input(
            "Azure Endpoint", 
            value="",
            help="Your Azure OpenAI endpoint URL"
        )
        
        api_key = st.text_input(
            "API Key", 
            type="password",
            help="Your Azure OpenAI API key"
        )
        
        submitted = st.form_submit_button("Connect to Azure OpenAI")
        
        if submitted:
            if not endpoint or not api_key:
                st.error("Please provide both endpoint and API key.")
                return False, None, None
            
            with st.spinner("Testing connection..."):
                success, result = test_azure_connection(endpoint, api_key)
                
                if success:
                    st.success("✅ Connection successful! You can now start chatting.")
                    # Store credentials in session state
                    st.session_state.azure_endpoint = endpoint
                    st.session_state.azure_api_key = api_key
                    st.session_state.azure_client = result
                    st.session_state.config_complete = True
                    st.rerun()
                else:
                    st.error(f"❌ Connection failed: {result}")
                    return False, None, None
    

def chat_page():
    """Chat page"""
    st.title("🤖 Chatbot AI with Azure OpenAI")
    
    # Show current configuration in sidebar
    with st.sidebar:
        st.header("⚙️ Current Configuration")
        st.info(f"**Endpoint**: {st.session_state.azure_endpoint}")
        st.info(f"**Model**: {model_name}")
        
        if st.button("🔧 Change Configuration"):
            # Reset configuration
            for key in ['azure_endpoint', 'azure_api_key', 'azure_client', 'config_complete', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Chat Information")

    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Show message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Write your message..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.write_stream(
                get_ai_response_stream(st.session_state.messages, st.session_state.azure_client)
            )
        
        # Add AI response to history
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Add counting of messages and conversations in sidebar
    with st.sidebar:
        if "messages" in st.session_state:
            st.info(f"**Messages**: {len(st.session_state.messages)}")
            st.info(f"**Conversations**: {len(st.session_state.messages) // 2}")

def main():
    st.set_page_config(
        page_title="Chatbot AI Azure", 
        page_icon="🤖",
        layout="wide"
    )
    
    # Check if configuration is complete
    if not st.session_state.get('config_complete', False):
        configuration_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()

    #creare un interfaccia streamlit per un chatbot che usa gpt con azure
    #impostare uno stream della risposta, risposta caricata parola per parola mano a mano che viene generata
    #impostare due schermate, una fatta e una dove l'utente inserisce endopoint, 
    
    #Deploy di un modello di embedding su azure
    #Fare uno script dove gli passiamo una frase e viene trasformato in vettore
    #Capire come implementare Tenacity in streamlit, per fare richiami all'API nel caso non partisse in modo automatico
