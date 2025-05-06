# Create an image embedding

from openai import OpenAI
import os
from dotenv import load_dotenv
import base64


# Clear old value (optional but safe)
os.environ.pop("OPENAI_API_KEY", None)

load_dotenv(override=True)  # Make sure it replaces existing values

api_key = os.getenv("OPENAI_API_KEY")
print("Using API key:", api_key)

#Check if the API value is loaded
if not api_key:
    raise ValueError("OPENAI_API_KEY is not loaded. Check your .env file and dotenv setup.")

client = OpenAI(api_key=api_key)

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Image path 
image_path = "2.jpg"
base64_image = encode_image(image_path)

# Call the OpenA
# I API for image embedding
response = client.embeddings.create(
    model="text-embedding-3-small", #"openai/image-embedding-001",
    input=base64_image, #{"image": base64_image}
    encoding_format="float"
)

# Get the embedding vector
# embedding_vector = response.data[0].embedding #response['data'][0]['embedding']
print("Embeddig vector : ", response)
# print("Embeddig vector length: ", len(embedding_vector))


# try:
#     response = client.embeddings.create(
#         model="text-embedding-3-small",
#         encoding_format="float",
#         input="Hello world"
#     )
    
#     # Print the embedding
#     embedding = response.data[0].embedding
#     print("Embeddig vector : ", embedding)
#     print("Embeddig vector length: ", len(embedding))

# except Exception as e:
#     print(" Error occurred:", e)

