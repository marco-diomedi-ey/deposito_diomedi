from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from a .env file if present

model_name = "text-embedding-ada-002"
api_version = "2024-12-01-preview"
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key: str = os.getenv("AZURE_OPENAI_API_KEY")

def get_embedding(client, input_text):
    """Get embedding for the input text"""
    try:
        response = client.embeddings.create(
            input=input_text,
            model=model_name
        )
        if response.data and len(response.data) > 0:
            return response.data[0].embedding
        else:
            raise ValueError("No embedding data returned")
    except Exception as e:
        raise RuntimeError(f"Error obtaining embedding: {str(e)}")

if __name__ == "__main__":
    input_text = input("Enter text to embed: ")
    embedding_client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key
    )
    embedding = get_embedding(embedding_client, input_text)
    print("Embedding:", embedding)