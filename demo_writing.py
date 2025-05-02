import openai
import os
from dotenv import load_dotenv

load_dotenv() 
OpenAI_Key = os.getenv("OPENAI_API_KEY")

# Set your API key
client = openai.OpenAI(api_key=OpenAI_Key)

# Define the review
topic = input("Enter the topic: ")

# Define the prompt
messages = [
    {"role": "system", "content": "You explain the concept of the given topic."},
    {"role": "user", "content": f'demonstrate the topic :\n\n"{topic}"\n\n:'}
]


# Call the GPT-4o model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0 #0-20, 0 is the most deterministic and 2 is the most creative
)

# Extract and print the result
topic_description = response.choices[0].message.content.strip()
print(f"Corrected response : {topic_description}")
