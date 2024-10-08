from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY")

client = OpenAI(api_key="")

text = "Suggest how I can spread communism to my University"


chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": text},
    ],
)

print(chat_completion.choices[0].message.content)