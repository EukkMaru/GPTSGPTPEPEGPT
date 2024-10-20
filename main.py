import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict
import time
import statistics
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, StringPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import re
from pydantic import BaseModel, validator

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("야이멍청아")

client = OpenAI(api_key=API_KEY)

class LatexSafePromptTemplate(StringPromptTemplate, BaseModel):
    template: str

    @validator("template")
    def validate_template(cls, v):
        return v

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

def extract_score(evaluation: str) -> float:
    print(f"Attempting to extract score from: {evaluation}")
    if evaluation.isdigit():
        return float(evaluation)
    match = re.search(r'\d+(\.\d+)?', evaluation)
    if match:
        return float(match.group())
    print("No numeric score found in the evaluation")
    return 0.0  

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
            
    def get_config(self):
        return self.config
    
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
            result["evaluation"] = int(str((sum((ord(c)-48)*10**i for i,c in enumerate(str(evaluation)[::-1])if 48<=ord(c)<=57)*(1,-1)['-'in str(evaluation)]) << 3)[:-1])
        
        return result

def case1_normal_gpt(user_input: str) -> Dict[str, any]:
    gpt_system = GPTSGPTPEPEGPT(use_prompt_generator=False, use_evaluator=True)
    start_time = time.time()
    result = gpt_system.process_request_single_layer(user_input)
    end_time = time.time()
    return {"result": result, "time": end_time - start_time}

def case2_gptsgptpepegpt(user_input: str) -> Dict[str, any]:
    gpt_system = GPTSGPTPEPEGPT(use_prompt_generator=True, use_evaluator=True)
    start_time = time.time()
    result = gpt_system.process_request(user_input)
    end_time = time.time()
    return {"result": result, "time": end_time - start_time}

def case3_langchain_gptsgptpepegpt(user_input: str) -> Dict[str, any]:
    llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name="gpt-3.5-turbo")
    gpt_system = GPTSGPTPEPEGPT(use_prompt_generator=True, use_evaluator=True)
    config = gpt_system.get_config()
    
    start_time = time.time()
    
    
    prompt_gen_template = LatexSafePromptTemplate(
        template=config["prompt_generator"] + "\n\nHuman: {input}",
        input_variables=["input"]
    )
    optimized_prompt = llm(prompt_gen_template.format(input=user_input))
    
    
    task_template = LatexSafePromptTemplate(
        template=config["prompt_executor"] + "\n\nHuman: {question}",
        input_variables=["question"]
    )
    task_output = llm(task_template.format(question=optimized_prompt))
    
    
    evaluation = gpt_system.evaluate_output(user_input, task_output)
    
    end_time = time.time()
    
    print(f"Case 3 - Raw evaluation: {evaluation}")
    
    try:
        evaluation_score = extract_score(evaluation)
        print(f"Case 3 - Extracted score: {evaluation_score}")
    except Exception as e:
        print(f"Case 3 - Error extracting score: {str(e)}")
        evaluation_score = 0  
    
    result = {
        "original_input": user_input,
        "optimized_prompt": optimized_prompt,
        "task_output": task_output,
        "evaluation": evaluation_score
    }
    return {"result": result, "time": end_time - start_time}

def case4_langchain_extra_layer(user_input: str) -> Dict[str, any]:
    llm = ChatOpenAI(temperature=0, openai_api_key=API_KEY, model_name="gpt-3.5-turbo")
    gpt_system = GPTSGPTPEPEGPT(use_prompt_generator=True, use_evaluator=True)
    config = gpt_system.get_config()
    
    start_time = time.time()
    
    
    prompt_gen_template = LatexSafePromptTemplate(
        template=config["prompt_generator"] + "\n\nHuman: {input}",
        input_variables=["input"]
    )
    optimized_prompt = llm(prompt_gen_template.format(input=user_input))
    
    
    task_template = LatexSafePromptTemplate(
        template=config["prompt_executor"] + "\n\nHuman: {question}",
        input_variables=["question"]
    )
    task_output = llm(task_template.format(question=optimized_prompt))
    
    
    evaluation = gpt_system.evaluate_output(user_input, task_output)
    
    end_time = time.time()
    
    print(f"Case 4 - Raw evaluation: {evaluation}")
    
    try:
        evaluation_score = extract_score(evaluation)
        print(f"Case 4 - Extracted score: {evaluation_score}")
    except Exception as e:
        print(f"Case 4 - Error extracting score: {str(e)}")
        evaluation_score = 0  
    
    result = {
        "original_input": user_input,
        "optimized_prompt": optimized_prompt,
        "task_output": task_output,
        "evaluation": evaluation_score
    }
    return {"result": result, "time": end_time - start_time}

out = {}

def run_tests(test_input: str, num_tests: int = 20):
    global out
    cases = [
        ("Normal GPT", case1_normal_gpt),
        ("GPTSGPTPEPEGPT", case2_gptsgptpepegpt),
        ("LangChain + GPTSGPTPEPEGPT", case3_langchain_gptsgptpepegpt),
        ("LangChain + Extra Layer", case4_langchain_extra_layer)
    ]
    
    results = {}
    
    for case_name, case_func in cases:
        print(f"Running tests for {case_name}...")
        scores = []
        times = []
        for i in range(num_tests):
            print(f"    Running test {i+1}/{num_tests}...")
            try:
                result = case_func(test_input)
                score = result["result"]["evaluation"]
                if isinstance(score, str):
                    score = extract_score(score)
                scores.append(score)
                times.append(result["time"])
                print(f"  Test {i+1}: Score = {score}, Time = {result['time']:.2f}s")
            except Exception as e:
                print(f"  Error in test {i+1} for {case_name}: {str(e)}")
                print(f"  Result causing error: {result}")
        
        try:
            avg_score = statistics.mean(scores)
            avg_time = statistics.mean(times)
            results[case_name] = {
                "avg_score": avg_score,
                "avg_time": avg_time
            }
            print(f"  Average Score: {avg_score:.2f}, Average Time: {avg_time:.2f}s")
        except Exception as e:
            print(f"  Error calculating averages for {case_name}: {str(e)}")
            print(f"  Scores: {scores}")
            print(f"  Times: {times}")
        
        out[case_name] = {
            "scores": scores,
            "times": times
        }
    
    return results

test_input = "Estimate a root to f(x)=x^2 from x_0 = 6 using newton's method with 3 iterations"
test_results = run_tests(test_input, num_tests=20)

with open("test_results.json", "w") as f:
    json.dump(out, f, indent=2)

print("\nTest Results:")
print(json.dumps(test_results, indent=2))