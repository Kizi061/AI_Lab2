# Create an embedding vector for the text

from openai import OpenAI
import os
from dotenv import load_dotenv

# Clear old value (optional but safe)
os.environ.pop("OPENAI_API_KEY", None)

load_dotenv(override=True)  # Make sure it replaces existing values

api_key = os.getenv("OPENAI_API_KEY")
print("Using API key:", api_key)

#Check if the API value is loaded
if not api_key:
    raise ValueError("OPENAI_API_KEY is not loaded. Check your .env file and dotenv setup.")

client = OpenAI(api_key=api_key)

try:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        encoding_format="float",
        input="Hello world"
    )
    
    # Print the embedding
    embedding = response.data[0].embedding
    print("Embeddig vector : ", embedding)
    print("Embeddig vector length: ", len(embedding))

except Exception as e:
    print(" Error occurred:", e)

