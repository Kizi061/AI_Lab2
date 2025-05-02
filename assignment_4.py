# Write 3 different prompts asking an LLM to summarize a news article.
# Test each prompt on the same article using GPT-4
# Analyze & summarize the news about the tone & detail level.

import openai
import os
from dotenv import load_dotenv

load_dotenv() 
OpenAI_Key = os.getenv("OPENAI_API_KEY")

# Set your API key
client = openai.OpenAI(api_key=OpenAI_Key)

# Define the review
article = input("Enter the article: ")

# Define the prompt 1
# messages = [
#     {"role": "system", "content": "Read the paragraph. Give the summary of the paragraph"},
#     {"role": "user", "content": f'Article summary :\n\n"{article}"\n\n:'}
# ]

# Define the prompt 2
# messages = [
#     {"role": "system", "content": "Summarize the given news article. Highlight the important facts like events, issues. The user for this summary are students"},
#     {"role": "user", "content": f'Article summary :\n\n"{article}"\n\n:'}
# ]

# Define the prompt 3
messages = [
    {"role": "system", "content": "Summarize the news in few sentences. Highlight the important points like events, issues. Avoid any speculation or opinions. Tailor the summary for students"},
    {"role": "user", "content": f'Article summary :\n\n"{article}"\n\n:'}
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
