import streamlit as st
from openai import AzureOpenAI

model_name = "gpt-4o"
embedding_model = "text-embedding-ada-002"
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
            model=model_name
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
            model=model_name,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Error communicating with AI: {str(e)}"

def get_embedding(client, input_text):
    """Get embedding for the input text"""
    try:
        response = client.embeddings.create(
            input=input_text,
            model=embedding_model
        )
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("No embedding data returned")
    except Exception as e:
        raise RuntimeError(f"Error obtaining embedding: {str(e)}")

def main_menu():
    """Main menu to choose between Chat and Embeddings"""
    st.title("🚀 Azure OpenAI Assistant")
    st.markdown("Choose your preferred mode:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💬 Chat Mode")
        st.markdown("Interactive conversation with GPT-4o")
        if st.button("🗨️ Start Chat", use_container_width=True):
            st.session_state.app_mode = "chat"
            st.session_state.config_complete = False
            st.rerun()
    
    with col2:
        st.markdown("### 🔢 Embeddings Mode")
        st.markdown("Convert text to vector embeddings")
        if st.button("📊 Start Embeddings", use_container_width=True):
            st.session_state.app_mode = "embeddings"
            st.session_state.config_complete = False
            st.rerun()
    
    # Show current mode if one is selected
    if "app_mode" in st.session_state:
        st.info(f"Current mode: **{st.session_state.app_mode.title()}**")
        if st.button("🔄 Change Mode"):
            for key in ['app_mode', 'config_complete', 'azure_endpoint', 'azure_api_key', 'azure_client', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

def configuration_page():
    """Configuration page for Azure credentials"""
    st.title("🔧 Azure OpenAI Configuration")
    st.markdown("Please provide your Azure OpenAI credentials to start")
    
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
                    st.success("✅ Connection successful! You can now start")
                    # Store credentials in session state
                    st.session_state.azure_endpoint = endpoint
                    st.session_state.azure_api_key = api_key
                    st.session_state.azure_client = result
                    st.session_state.config_complete = True
                    st.rerun()
                else:
                    st.error(f"❌ Connection failed: {result}")
                    return False, None, None
    

def embeddings_page():
    """Embeddings page for text to vector conversion"""
    st.title("🔢 Text Embeddings with Azure OpenAI")
    
    # Show current configuration in sidebar
    with st.sidebar:
        st.header("⚙️ Current Configuration")
        st.info(f"**Endpoint**: {st.session_state.azure_endpoint}")
        st.info(f"**Model**: {embedding_model}")
        
        if st.button("🔧 Change Configuration"):
            st.session_state.config_complete = False
            st.rerun()
        
        if st.button("🏠 Back to Menu"):
            for key in ['app_mode', 'config_complete', 'azure_endpoint', 'azure_api_key', 'azure_client']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Main embeddings interface
    st.markdown("### Enter text to convert to embeddings:")
    
    # Text input methods
    input_method = st.radio("Choose input method:", ["Single Text", "Multiple Texts"])
    
    if input_method == "Single Text":
        input_text = st.text_area("Enter your text:", height=100, placeholder="Type your text here...")
        
        if st.button("🔄 Generate Embedding") and input_text:
            with st.spinner("Generating embedding..."):
                try:
                    embedding = get_embedding(st.session_state.azure_client, input_text)
                    
                    # Display results
                    st.success("✅ Embedding generated successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Vector Dimensions", len(embedding))
                    with col2:
                        st.metric("Input Length", len(input_text))
                    
                    # Show first few values
                    st.subheader("Embedding Preview (first 10 values):")
                    st.code(str(embedding[:10]))
                    
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    else:  # Multiple Texts
        st.markdown("**Upload a text file or enter multiple texts:**")
        
        uploaded_file = st.file_uploader("Choose a text file", type=['txt'])
        
        if uploaded_file is not None:
            content = str(uploaded_file.read(), "utf-8")
            texts = [line.strip() for line in content.split('\n') if line.strip()]
            
            st.write(f"Found {len(texts)} lines to process")
            
            if st.button("🔄 Generate All Embeddings"):
                embeddings_results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(texts):
                    try:
                        embedding = get_embedding(st.session_state.azure_client, text)
                        embeddings_results.append({
                            'text': text,
                            'embedding': embedding,
                            'dimensions': len(embedding)
                        })
                        progress_bar.progress((i + 1) / len(texts))
                    except Exception as e:
                        st.warning(f"Failed to process line {i+1}: {str(e)}")
                
                st.success(f"✅ Generated {len(embeddings_results)} embeddings!")
                
                # Show summary
                if embeddings_results:
                    st.subheader("Results Summary:")
                    for i, result in enumerate(embeddings_results[:5]):  # Show first 5
                        with st.expander(f"Text {i+1}: {result['text'][:50]}..."):
                            st.write(f"Dimensions: {result['dimensions']}")
                            st.code(str(result['embedding'][:10]))

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
        
        if st.button("🏠 Back to Menu"):
            for key in ['app_mode', 'config_complete', 'azure_endpoint', 'azure_api_key', 'azure_client', 'messages']:
                if key in st.session_state:
                    del st.session_state[key]
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
        page_title="Azure OpenAI Assistant", 
        page_icon="🚀",
        layout="wide"
    )
    
    # Check if app mode is selected
    if not st.session_state.get('app_mode'):
        main_menu()
    else:
        # Check if configuration is complete for the selected mode
        if not st.session_state.get('config_complete', False):
            configuration_page()
        else:
            # Route to the appropriate page based on selected mode
            if st.session_state.app_mode == "chat":
                chat_page()
            elif st.session_state.app_mode == "embeddings":
                embeddings_page()

if __name__ == "__main__":
    main()

    #creare un interfaccia streamlit per un chatbot che usa gpt con azure
    #impostare uno stream della risposta, risposta caricata parola per parola mano a mano che viene generata
    #impostare due schermate, una fatta e una dove l'utente inserisce endopoint, 
    
    #Deploy di un modello di embedding su azure
    #Fare uno script dove gli passiamo una frase e viene trasformato in vettore
    #Capire come implementare Tenacity in streamlit, per fare richiami all'API nel caso non partisse in modo automatico
