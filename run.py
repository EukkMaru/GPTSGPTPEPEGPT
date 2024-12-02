import json
import time
import logging
from isolated import run_gptsgptpepegpt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_ruleset(filename: str) -> str:
    with open(filename, "r") as f:
        ruleset = f.read()
    logging.info(f"Ruleset loaded from {filename}")
    return ruleset

def get_user_task() -> str:
    return input("Enter the task you want to evaluate against the ruleset: ")

def write_result_to_file(result: dict, filename: str = "gptsgptpepegpt_result.json"):
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Result written to {filename}")

def main():    
    ruleset_filename = "ruleset.txt"
    ruleset = load_ruleset(ruleset_filename)
    
    user_task = get_user_task()
    
    user_input = (
        f"Ruleset:\n{ruleset}\n\n"
        f"Task: {user_task}\n\n"
        "Determine if the task is possible without violating the rules. (The answer should be either 'yes' or 'no' followed by an explanation, not a 'not sure' with ambiguous explanation.) "
        "If possible, explain how far we can go without violating the rules. "
        "Identify any loopholes in the ruleset and suggest fixes."
    )
    
    config_file = "system_messages.json"
    use_prompt_generator = True
    use_evaluator = True

    start_time = time.time()
    
    logging.info("Running GPTSGPTPEPEGPT...")
    result = run_gptsgptpepegpt(user_input, config_file=config_file, use_prompt_generator=use_prompt_generator, use_evaluator=use_evaluator)
    
    end_time = time.time()
    execution_time = end_time - start_time
    logging.info(f"Execution time: {execution_time:.2f} seconds")

    print(json.dumps(result, indent=2))
    write_result_to_file(result)

if __name__ == "__main__":
    main()
