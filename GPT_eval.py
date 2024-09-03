from openai import OpenAI
import os
from utils.api import get_api_key

api_key = get_api_key()

os.environ["OPENAI_API_KEY"] = api_key  # Replace with your API key
client = OpenAI()
system_prompt = ""
user_prompt = f"""Problem: {problem}"""
response = client.chat.completions.create(
  model="gpt-4o-mini",
  messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt}
      ],
  max_tokens=300,
)
print(response.choices[0])