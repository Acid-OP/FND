import os
import pandas as pd
from random import sample
import requests
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# --- Load datasets ---
df_fake = pd.read_csv(r"G:\Lock in\New folder\Dataset\gossipcop_fake.csv")
df_real = pd.read_csv(r"G:\Lock in\New folder\Dataset\gossipcop_real.csv")

df_fake['tweet_ids'] = df_fake['tweet_ids'].apply(lambda x: str(x).split())
df_real['tweet_ids'] = df_real['tweet_ids'].apply(lambda x: str(x).split())

# --- Take samples ---
FAKE_SAMPLE_SIZE = 5
REAL_SAMPLE_SIZE = 5
fake_sample = sample(df_fake['title'].dropna().tolist(), min(len(df_fake), FAKE_SAMPLE_SIZE))
real_sample = sample(df_real['title'].dropna().tolist(), min(len(df_real), REAL_SAMPLE_SIZE))
test_samples = [('FAKE', t) for t in fake_sample] + [('REAL', t) for t in real_sample]

# --- Hugging Face mock function (replace with real HF pipeline if needed) ---
def run_huggingface(text):
    import random
    score = random.uniform(0, 1)
    label = 'FAKE' if score > 0.5 else 'REAL'
    return label, score

# --- Real Gemini API call ---
def run_gemini(text, model="gemini-1.5-flash"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"Rate the following text for clickbait style (0-10). Return only the number:\n\n{text}"}]}]
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        output_text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        score = float(output_text.strip())
    except Exception as e:
        print(f"Gemini API error: {e}")
        score = 0.5  # default score on error

    label = "FAKE" if score > 5 else "REAL"
    normalized_score = min(max(score / 10, 0), 1)
    return label, normalized_score

# --- Initialize stats ---
hf_wrong = 0
gem_wrong = 0
overall_wrong = 0

HF_WEIGHT = 0.5
GEM_WEIGHT = 0.5
CONFIDENCE_MARGIN = 0.1

# --- Run weighted ensemble ---
for true_label, tweet in test_samples:
    hf_label, hf_score = run_huggingface(tweet)
    gem_label, gem_score = run_gemini(tweet)
    
    if hf_label != true_label:
        hf_wrong += 1
    if gem_label != true_label:
        gem_wrong += 1

    vote_score = HF_WEIGHT * (hf_score if hf_label == 'FAKE' else -hf_score)
    vote_score += GEM_WEIGHT * (gem_score if gem_label == 'FAKE' else -gem_score)
    
    if vote_score > CONFIDENCE_MARGIN:
        final_label = 'FAKE'
    elif vote_score < -CONFIDENCE_MARGIN:
        final_label = 'REAL'
    else:
        final_label = 'UNCERTAIN'

    if final_label != true_label and final_label != 'UNCERTAIN':
        overall_wrong += 1

    print(f"Tweet: {tweet[:50]}...")
    print(f"True Label: {true_label}")
    print(f"HuggingFace -> Label: {hf_label}, Score: {hf_score:.2f}")
    print(f"Gemini      -> Label: {gem_label}, Score: {gem_score:.2f}")
    print(f"Final Decision (weighted): {final_label}")
    print("="*80)

# --- Summary ---
total = len(test_samples)
print(f"Total Samples: {total}")
print(f"Hugging Face wrong: {hf_wrong}/{total}")
print(f"Gemini wrong: {gem_wrong}/{total}")
print(f"Weighted final wrong: {overall_wrong}/{total}")
