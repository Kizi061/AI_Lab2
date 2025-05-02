# Write asking an LLM to summarize a news article. Allow user to input multiple lines and then type END to indicate user is done uploaded the news
# Test prompt on the same article using GPT-4
# Analyze & summarize the news about the tone & detail level.

import openai
import os
from dotenv import load_dotenv

load_dotenv() 
OpenAI_Key = os.getenv("OPENAI_API_KEY")

# Set your API key
client = openai.OpenAI(api_key=OpenAI_Key)

def get_article():
    data = []
    print("Enter your input (type 'END' on a new line to finish):.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "end":
            break  
        data.append(user_input)
    merged_data = "\n".join(data)
    return merged_data

# Get user input
paragraph  = get_article()  
print(f"User data: {paragraph}")

# Define the prompt 3
messages = [
    {"role": "system", "content": "Summarize the news in few sentences. Highlight the important points like events, issues. Avoid any speculation or opinions. Tailor the summary for students"},
    {"role": "user", "content": f'Article summary :\n\n"{paragraph}"\n\n:'}
]

# Call the GPT-4o model
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    temperature=0 #0-20, 0 is the most deterministic and 2 is the most creative
)

# Extract and print the result
summary_response = response.choices[0].message.content.strip()
print(f"Summary : {summary_response}")
