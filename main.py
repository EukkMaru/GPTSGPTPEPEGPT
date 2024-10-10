import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import time

load_dotenv()

API_KEY = os.getenv("API_KEY")

client = OpenAI(api_key=API_KEY)

#TODO Add time comparison

class ConversationLayer:
    def __init__(self, system_message: str):
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
    
    def get_response(self, model: str = "gpt-3.5-turbo") -> str:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=self.messages
        )
        content = chat_completion.choices[0].message.content
        self.add_message("assistant", content)
        return content

class GPTSGPTPEPEGPT:
    def __init__(self, config_file: str = "system_messages.json", use_prompt_generator: bool = True, use_evaluator: bool = True):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.use_prompt_generator = use_prompt_generator
        if self.use_prompt_generator:
            self.prompt_generator = ConversationLayer(self.config["prompt_generator"])
        self.prompt_executor = ConversationLayer(self.config["prompt_executor"])
        self.use_evaluator = use_evaluator
        if self.use_evaluator:
            self.performance_evaluator = ConversationLayer(self.config["performance_evaluator"])
    
    def evaluate_output(self, original_input: str, task_output: str) -> str:
        evaluation_request = f"""Original Input: {original_input}
Task Output: {task_output}"""
        self.performance_evaluator.add_message("user", evaluation_request)
        return self.performance_evaluator.get_response()
    
    def process_request(self, user_input: str) -> Dict[str, str]:
        if self.use_prompt_generator:
            self.prompt_generator.add_message("user", user_input)
            optimized_prompt = self.prompt_generator.get_response()
            self.prompt_executor.add_message("user", optimized_prompt)
        else:
            self.prompt_executor.add_message("user", user_input)
        
        task_output = self.prompt_executor.get_response()
        
        result = {
            "original_input": user_input,
            "task_output": task_output
        }
        
        if self.use_prompt_generator:
            result["optimized_prompt"] = optimized_prompt
        
        if self.use_evaluator:
            evaluation = self.evaluate_output(user_input, task_output)
            result["evaluation"] = evaluation
        
        return result

    def process_request_single_layer(self, user_input: str) -> Dict[str, str]:
        self.prompt_executor.add_message("user", user_input)
        task_output = self.prompt_executor.get_response()
        
        result = {
            "original_input": user_input,
            "task_output": task_output
        }
        
        if self.use_evaluator:
            evaluation = self.evaluate_output(user_input, task_output)
            result["evaluation"] = evaluation
        
        return result

def compare_approaches(gpt_system: GPTSGPTPEPEGPT, user_input: str):
    print("Multi-layer approach result:")
    start_multi = time.time()
    multi_layer_result = gpt_system.process_request(user_input)
    end_multi = time.time()
    multi_elapsed = end_multi - start_multi
    print(json.dumps(multi_layer_result, indent=2))
    
    print("\nSingle-layer approach result:")
    start_single = time.time()
    single_layer_result = gpt_system.process_request_single_layer(user_input)
    end_single = time.time()
    single_elapsed = end_single - start_single
    print(json.dumps(single_layer_result, indent=2))
    print(f"\n\nMulti-layer Time: {multi_elapsed} seconds\nSingle-layer Time: {single_elapsed} seconds")

gpt_system = GPTSGPTPEPEGPT(config_file="system_messages.json", use_prompt_generator=True, use_evaluator=True)
compare_approaches(gpt_system, "Write a python script to solve a quadratic equation using newton's method")