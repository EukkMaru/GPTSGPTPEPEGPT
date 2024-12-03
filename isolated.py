import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict
import time
from helpers import LatexSafePromptTemplate, ConversationLayer, client

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Please set the API_KEY environment variable.")


class GPTSGPTPEPEGPT:
    def __init__(self, config_file: str = "system_messages.json", use_prompt_generator: bool = True, use_evaluator: bool = True):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.use_prompt_generator = use_prompt_generator
        if self.use_prompt_generator:
            prompt_template = LatexSafePromptTemplate(template=self.config["prompt_generator"])
            self.prompt_generator = ConversationLayer(prompt_template.format())
        self.prompt_executor = ConversationLayer(self.config["prompt_executor"])
        self.use_evaluator = use_evaluator
        if self.use_evaluator:
            evaluator_template = LatexSafePromptTemplate(template=self.config["performance_evaluator"])
            self.performance_evaluator = ConversationLayer(evaluator_template.format())
            
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

def run_gptsgptpepegpt(user_input: str, config_file: str = "system_messages.json", use_prompt_generator: bool = True, use_evaluator: bool = True) -> Dict[str, str]:
    """
    G!P!T!S!G!P!T!P!E!P!E!G!P!T!실!행!

    Args:
        user_input (str): 유저 인풋
        config_file (str): 시스템 메시지 path
        use_prompt_generator (bool): 그냥 True로 두세요...
        use_evaluator (bool): 그냥 True로 두세요2...

    Returns:
        Dict[str, str]: 오리지널 인풋, 아웃풋, 최적화된 프롬프트, 평가
    """
    gpt_system = GPTSGPTPEPEGPT(config_file=config_file, use_prompt_generator=use_prompt_generator, use_evaluator=use_evaluator)
    start_time = time.time()
    result = gpt_system.process_request(user_input)
    end_time = time.time()
    result["execution_time"] = end_time - start_time
    return result

# Example usage
if __name__ == "__main__":
    test_input = "There are N people and N tasks. Each person should be in charge of one task..."
    result = run_gptsgptpepegpt(test_input)
    print(json.dumps(result, indent=2))
