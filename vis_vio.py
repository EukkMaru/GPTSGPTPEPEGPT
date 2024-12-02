import json
import logging
import matplotlib.pyplot as plt
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_result(filename: str = "gptsgptpepegpt_result.json") -> Dict:
    with open(filename, "r") as f:
        result = json.load(f)
    logging.info(f"Result loaded from {filename}")
    return result

def analyze_result(result: Dict):
    if "evaluation" in result:
        evaluation = result["evaluation"]
        logging.info(f"Analyzing evaluation: {evaluation}")
        
        if "loophole" in evaluation.lower():
            logging.warning("Potential loophole detected in the ruleset.")
        if "violate" in evaluation.lower():
            logging.warning("Task might be violating some rules.")
    else:
        logging.error("No evaluation found in the result.")

def visualize_analysis(result: Dict):
    if "evaluation" in result:
        labels = ["Allowed Actions", "Restricted Actions", "Potential Loopholes"]
        counts = [0, 0, 0]

        evaluation = result["evaluation"].lower()
        if "allowed" in evaluation:
            counts[0] += 1
        if "violate" in evaluation:
            counts[1] += 1
        if "loophole" in evaluation:
            counts[2] += 1

        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color=['green', 'red', 'orange'])
        plt.xlabel("Evaluation Categories")
        plt.ylabel("Counts")
        plt.title("Analysis of Task Against Ruleset")
        plt.grid(axis='y')

        plt.savefig("rule_analysis_visualization.png")
        plt.show()
        logging.info("Visualization saved as 'rule_analysis_visualization.png'")
    else:
        logging.error("No evaluation found to visualize.")

def generate_summary_report(result: Dict, filename: str = "summary_report.txt"):
    with open(filename, "w") as f:
        f.write("Summary Report for Task Evaluation Against Ruleset\n")
        f.write("="*50 + "\n")
        f.write(f"Original Input: {result.get('original_input', 'N/A')}\n")
        f.write(f"Task Output: {result.get('task_output', 'N/A')}\n")
        f.write(f"Evaluation: {result.get('evaluation', 'N/A')}\n")
    logging.info(f"Summary report written to {filename}")

def main():
    result_filename = "gptsgptpepegpt_result.json"
    result = load_result(result_filename)

    analyze_result(result)    
    visualize_analysis(result)
    generate_summary_report(result)

if __name__ == "__main__":
    main()
