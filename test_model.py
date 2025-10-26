import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# LOAD MODEL WITH ADAPTER
# -------------------------
adapter_path = "./final_news_adapter"
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading base model and adapter on {device}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(adapter_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Load PEFT adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.to(device)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_news(text, max_length=256):
    """
    Predict whether news text is true or false
    """
    # Create the same prompt format used during training
    prompt = f"""### Instruction:
You are a text classification assistant. Your task is to analyze the tone of the provided news text and classify its stylistic trustworthiness.

- Classify as 'true' if the tone is trustworthy (neutral, formal, objective).
- Classify as 'false' if the tone is untrustworthy (sensationalist, emotionally charged, opinionated).

### Input:
{text}

### Response:
"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,  # We only need "true" or "false"
            temperature=0.1,    # Low temperature for more deterministic output
            do_sample=False,    # Greedy decoding
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the response part (after "### Response:")
    response = generated_text.split("### Response:")[-1].strip().lower()
    
    # Extract prediction (look for "true" or "false" in the response)
    if "true" in response and "false" not in response:
        prediction = "true"
    elif "false" in response:
        prediction = "false"
    else:
        # If unclear, analyze the first word
        first_word = response.split()[0] if response.split() else ""
        prediction = "true" if "true" in first_word else "false"
    
    return prediction, response

# -------------------------
# TEST ON SINGLE EXAMPLE
# -------------------------
def test_single_example():
    print("\n" + "="*50)
    print("TESTING SINGLE EXAMPLE")
    print("="*50)
    
    sample_text = """
    The president announced a new policy today that will change the healthcare system. 
    The policy includes several provisions aimed at reducing costs and improving access.
    """
    
    prediction, response = predict_news(sample_text)
    print(f"\nSample Text: {sample_text[:100]}...")
    print(f"\nModel Response: {response}")
    print(f"Prediction: {prediction}")

# -------------------------
# TEST ON DATASET
# -------------------------
def test_on_dataset():
    print("\n" + "="*50)
    print("TESTING ON DATASET")
    print("="*50)
    
    # Load test data (using new samples that weren't in training)
    print("\nLoading test dataset...")
    fake_data = pd.read_csv("./Dataset/Fake.csv")
    real_data = pd.read_csv("./Dataset/True.csv")
    
    # Skip the first 100 samples used in training
    fake_test = fake_data.iloc[100:200]  # Get 50 samples
    real_test = real_data.iloc[100:200]  # Get 50 samples
    
    fake_test["label"] = "false"
    real_test["label"] = "true"
    
    test_df = pd.concat([fake_test, real_test]).reset_index(drop=True)
    
    print(f"Test dataset size: {len(test_df)}")
    
    # Make predictions
    predictions = []
    true_labels = []
    
    print("\nMaking predictions...")
    for idx, row in test_df.iterrows():
        if idx % 10 == 0:
            print(f"Processing {idx}/{len(test_df)}...")
        
        text = str(row["text"])
        true_label = str(row["label"])
        
        prediction, _ = predict_news(text)
        
        predictions.append(prediction)
        true_labels.append(true_label)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=["false", "true"]))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predictions))
    
    # Show some examples
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for i in range(min(5, len(test_df))):
        print(f"\n--- Example {i+1} ---")
        print(f"Text: {test_df.iloc[i]['text'][:150]}...")
        print(f"True Label: {true_labels[i]}")
        print(f"Predicted: {predictions[i]}")
        print(f"Correct: {'✓' if predictions[i] == true_labels[i] else '✗'}")

# -------------------------
# INTERACTIVE MODE
# -------------------------
def interactive_mode():
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Enter news text to classify (or 'quit' to exit)")
    
    while True:
        print("\n" + "-"*50)
        user_input = input("\nEnter news text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
        
        prediction, response = predict_news(user_input)
        print(f"\nModel Response: {response}")
        print(f"Classification: {prediction.upper()}")

# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("\n" + "="*50)
    print("FAKE NEWS DETECTION - MODEL TESTING")
    print("="*50)
    
    # Test single example
    test_single_example()
    
    # Test on dataset
    try:
        test_on_dataset()
    except Exception as e:
        print(f"\nCouldn't test on dataset: {e}")
        print("Make sure Dataset/Fake.csv and Dataset/True.csv exist")
    
    # Interactive mode (optional)
    print("\n" + "="*50)
    use_interactive = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
    if use_interactive == 'y':
        interactive_mode()
    
    print("\nTesting completed!")