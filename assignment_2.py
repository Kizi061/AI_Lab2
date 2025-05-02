import os
import openai
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPEN_API_KEY")

# Create a client instance
# Set your API key
client = openai.OpenAI(api_key=key)

#Define the review
input_array_int = input("Enter a list of integer:")
input_target = input("Enter a target:")


# Writing AI agent
messages = [
    {
        "role": "system",
        "content": " write an Python function that finds all the possible combination of target number provided by the user. All outputs should be a unique pairs of numbers whose sum equals the target.(e.g., avoid both (2, 3) and (3, 2)). Optimize your solution for time and space complexity  "
    },
     {
        "role": "user",
        "content": f'The integar array :\n\n"{input_array_int}"\n\n The targer :\n\n"{input_target}"\n\n:'
    }
]

# Call the GPT-4o model
response = client.chat.completions.create(
    model = "gpt-4o",#"gpt-4-turbo",
    messages=messages,
    temperature=0
)

#Extract and print the result
# print(response)
result = response.choices[0].message.content
print(f"Result:{result}")


# def find_unique_pairs(nums, target):
#     # First, sort the array
#     nums.sort()

#     # Initialize two pointers
#     left, right = 0, len(nums) - 1

#     # To store the unique pairs
#     unique_pairs = []

#     while left < right:
#         current_sum = nums[left] + nums[right]

#         if current_sum == target:
#             # Add the pair to the result
#             unique_pairs.append((nums[left], nums[right]))

#             # Move both pointers to find the next unique pair
#             left += 1
#             right -= 1

#             # Skip duplicates
#             while left < right and nums[left] == nums[left - 1]:
#                 left += 1
#             while left < right and nums[right] == nums[right + 1]:
#                 right -= 1

#         elif current_sum < target:
#             # Move the left pointer to increase the sum
#             left += 1
#         else:
#             # Move the right pointer to decrease the sum
#             right -= 1

#     return unique_pairs

# # Example usage
# nums = [2, 3, 4, 5, 6, 0]
# target = 8
# print(find_unique_pairs(nums, target))

