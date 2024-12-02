from pydantic import BaseModel, validator
import re
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
client = OpenAI(api_key=API_KEY)

class LatexSafePromptTemplate(BaseModel):
    template: str

    @validator("template")
    def validate_template(cls, v):
        if not isinstance(v, str):
            raise ValueError("Template must be a string")
        return v

    def format(self, **kwargs) -> str:
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing key in template formatting: {e}")

class ConversationLayer:
    def __init__(self, system_message: str):
        self.messages = [{"role": "system", "content": system_message}]
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_response(self, model: str = "gpt-3.5-turbo") -> str:
        # Placeholder for the actual API call to OpenAI, assuming client is globally accessible
        chat_completion = client.chat.completions.create(
            model=model,
            messages=self.messages
        )
        content = chat_completion.choices[0].message.content
        self.add_message("assistant", content)
        return content
