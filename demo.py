import os
import openai
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPEN_API_KEY")

# Create a client instance
# Set your API key
client = openai.OpenAI(api_key=key)

#Define the review
review = input("Enter a string:")

# Define the prompt
# # Reading AI agent
# messages = [
#     {
#         "role": "system",
#         "content": "You classify resturant review sentiment Positive, Negative or Neutral"
#     },
#      {
#         "role": "user",
#         "content": f'"{review}"'
#         # "content": f'Classify the sentimentof the review:\n\n"{review}"\n\n sentiment:'
#     }
# ]

# Writing AI agent
# messages = [
#     {
#         "role": "system",
#         "content": "You identify the language of the given text."
#     },
#      {
#         "role": "user",
#         "content": f'Detect the language of this sentence:\n\n"{review}"\n\n language:'
#     }
# ]

# Writing AI agent
messages = [
    {
        "role": "system",
        "content": "You correct grammer and spelling in English text"
    },
     {
        "role": "user",
        "content": f'Correct the following text:\n\n"{review}"\n\n Corrected:'
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
corrected = response.choices[0].message.content.strip() 
# print(f"Sentiment:{sentiment}")
# print(f"language:{language}")
print(f"Corrected:{corrected}")