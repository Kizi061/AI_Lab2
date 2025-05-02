import os
import openai
from dotenv import load_dotenv #for secret key

load_dotenv()
key = os.getenv("OPEN_API_KEY")

# Create a client instance
# Set your API key
client = openai.OpenAI(api_key=key)


# Writing AI agent
messages = [
    {
        "role": "system",
        "content": " Convert the given paragraph from English to Spanish language. The paragrah describes the key features of advanced renewable energy system. Ensure to use the correct terminology related to the renewable energy sector. It should use the correct technical terms and translate the concept "
    }
]

# Call the GPT-4o model
response = client.chat.completions.create(
    model = "gpt-4o",
    messages=messages,
    temperature=0
)

#Extract and print the result
# print(response)
result = response.choices[0].message.content
print(f"Result:{result}")