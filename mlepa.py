# 1. IMPORTS
import json
from collections import deque
import pandas as pd

# Import helpers and classes from other files
from utility import setup_local_llm
from ace_agent import ACEAgent
from prompts import stylistic_prompts

# ---
# 2. SETUP
# ---
# Load the model and tokenizer once using the utility function
print("--- Initializing Setup ---")
model, tokenizer = setup_local_llm()


# ---
# 3. AGENT INITIALIZATION
# ---
# Create an instance of the ACEAgent class
print("--- Creating Stylistic Agent ---")
stylistic_agent = ACEAgent(
    agent_type="stylistic", 
    model=model, 
    tokenizer=tokenizer,
    prompt_templates={
        'generator': stylistic_prompts.GENERATOR_PROMPT,
        'reflector': stylistic_prompts.REFLECTOR_PROMPT,
        'curator': stylistic_prompts.CURATOR_PROMPT,
    }
)
# You can create other agents here as well
# factual_agent = ACEAgent(...)
# source_agent = ACEAgent(...)


# ---
# 4. MAIN WORKFLOW FUNCTION
# ---
def process_csv(filepath):
    """
    Reads a CSV file, processes each article, and triggers the agent's
    learning loop based on ground truth labels.
    """
    try:
        df = pd.read_csv(filepath,nrows=10)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    # --- For Tracking Improvement ---
    total_correct = 0
    history = deque(maxlen=10) # For rolling accuracy
    
    print(f"\n--- Starting Processing of {len(df)} Articles ---")

    # Iterate over each row in the CSV file
    for index, row in df.iterrows():
        article_text = row['text']
        ground_truth_label = row['label'] # e.g., "Fake" or "Real"

        # --- Analysis Step ---
        analysis_result = stylistic_agent.generate({"text": article_text})
        print(f"DEBUG: Raw analysis_result: {analysis_result}") # <-- ADD THIS
        # --- Prediction Mapping ---
        predicted_label = "Fake" if "Non-Trusted" in analysis_result.get("classification", "") else "Real"
        
        # --- Comparison and Feedback Step ---
        is_correct = (predicted_label == ground_truth_label)
        total_correct += 1 if is_correct else 0
        history.append(is_correct)

        feedback = {
            "is_style_analysis_correct": is_correct,
            "reason": f"Prediction was '{predicted_label}' but ground truth was '{ground_truth_label}'."
        }

        # --- Learning Step (only happens if the prediction was wrong) ---
        if not is_correct:
            print(f"-> Incorrect prediction. Triggering learning loop for agent...")
          
            insights = stylistic_agent.reflect(analysis_result, feedback)
            print(f"DEBUG: Raw insights: {insights}") # <-- ADD THIS
            
            delta_items = stylistic_agent.curate(insights)
            print(f"DEBUG: Raw delta_items: {delta_items}") # <-- ADD THIS

            stylistic_agent.update_playbook(delta_items)
        
        # --- Dynamic Progress Reporting ---
        rolling_accuracy = (sum(history) / len(history)) * 100 if history else 0
        print(f"Row {index + 1}/{len(df)} | Prediction: {predicted_label} | Actual: {ground_truth_label} | Correct: {is_correct} | Rolling Accuracy (last 10): {rolling_accuracy:.2f}%")

    # --- Final Report ---
    overall_accuracy = (total_correct / len(df)) * 100
    print("\n--- Processing Complete ---")
    print(f"Overall Accuracy: {overall_accuracy:.2f}% ({total_correct}/{len(df)})")


# ---
# 5. EXECUTION
# ---
if __name__ == "__main__":
    # Define the path to your CSV file
    csv_file_path = './Dataset/Fake.csv' 
    # Call the main function to start processing
    process_csv(csv_file_path)