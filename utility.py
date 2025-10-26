import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

# Centralize model loading here
def setup_local_llm(model_id="meta-llama/Llama-3.1-8B-Instruct"):
    """Loads and returns the model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to(device)
    print("Model loaded successfully.")
    return model, tokenizer

# Centralize the LLM call function here
# In your utils.py or utility.py file

# ... (setup_local_llm function is the same) ...

def call_llm(prompt, model, tokenizer):
    """Performs inference and uses JSONDecoder to robustly parse the first valid JSON object."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)




    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)


    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # --- START OF FIX ---
    # Use JSONDecoder to find and parse only the first valid JSON object in the string
    try:
        # Find the first opening curly brace to start the search from
        start_index = response_text.find('{')
        if start_index == -1:
            print("Warning: No JSON object found in the LLM response.")
            return "{}" # Return an empty JSON string if no '{' is found
        
        # Create a decoder and use raw_decode which parses one object and returns its end position
        decoder = json.JSONDecoder()
        # The return value for json.loads() should be a string, so we need to slice it first
        # and then let the main agent's json.loads() handle the final conversion.
        obj, end = decoder.raw_decode(response_text[start_index:])
        
        # Extract the string segment that corresponds to the valid JSON object
        json_str = response_text[start_index : start_index + end]
        return json_str

    except json.JSONDecodeError:
        # This will catch errors if the found object is still not valid JSON
        print("Warning: Failed to decode JSON from LLM response.")
        return "{}"
    # --- END OF FIX ---