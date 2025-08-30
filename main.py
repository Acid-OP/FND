import os
import pandas as pd
from random import sample
import requests
from dotenv import load_dotenv
# from langchain_community.llms import Ollama
from langchain_ollama.llms import OllamaLLM


# --- Ollama model --- #

llm = OllamaLLM(model="llama3:latest")

# --- Load environment variables ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# --- Load datasets ---
df_fake = pd.read_csv(r"./Dataset/gossipcop_fake.csv")
df_real = pd.read_csv(r"./Dataset/gossipcop_real.csv")

df_fake['tweet_ids'] = df_fake['tweet_ids'].apply(lambda x: str(x).split())
df_real['tweet_ids'] = df_real['tweet_ids'].apply(lambda x: str(x).split())

# --- Take samples ---
FAKE_SAMPLE_SIZE = 30
REAL_SAMPLE_SIZE = 30
fake_sample = sample(df_fake['title'].dropna().tolist(), min(len(df_fake), FAKE_SAMPLE_SIZE))
real_sample = sample(df_real['title'].dropna().tolist(), min(len(df_real), REAL_SAMPLE_SIZE))
test_samples = [('FAKE', t) for t in fake_sample] + [('REAL', t) for t in real_sample]



# --- Hugging Face mock function (replace with real HF pipeline if needed) ---
def run_model1(text):
    result = llm.invoke(f"Rate the text for sensational/emotional phrasing. Output only a float in (0,1). 0=definitely real, 1=definitely fake. No words, no explanation.Text:{text}")

    score = float(result.strip())
    label = 'FAKE' if score > 0.5 else 'REAL'
    return label, score

# --- Real Gemini API call ---
def run_model2(text, model="gemini-2.0-flash"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{"parts": [{"text": f"Rate the text for sensational/emotional phrasing. Output only a float in (0,1). 0=definitely real, 1=definitely fake. No words, no explanation.Text:{text}"}]}]
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

    label = "FAKE" if score > 0.5 else "REAL"
    return label, score

# --- Initialize stats ---
mod1_wrong = 0
mod2_wrong = 0
overall_wrong = 0
total_uncertain = 0

CONFIDENCE_MARGIN = 0.1

# --- Run weighted ensemble ---
for true_label, tweet in test_samples:
    mod1_label, mod1_score = run_model1(tweet)
    mod2_label, mod2_score = run_model2(tweet)
    
    if mod1_label != true_label:
        mod1_wrong += 1

    if mod2_label != true_label:
        mod2_wrong += 1
  

    vote_score = 0.5 * (mod1_score if mod1_label == 'FAKE' else -mod1_score)
    vote_score += 0.5 * (mod2_score if mod2_label == 'FAKE' else -mod2_score)
    
    if vote_score > CONFIDENCE_MARGIN:
        final_label = 'FAKE'
    elif vote_score < -CONFIDENCE_MARGIN:
        final_label = 'REAL'
    else:
        final_label = 'UNCERTAIN'
        total_uncertain += 1

    if final_label != true_label and final_label != 'UNCERTAIN':
        overall_wrong += 1

    print(f"Tweet: {tweet[:50]}...")
    print(f"True Label: {true_label}")
    print(f"Model1 -> Label: {mod1_label}, Score: {mod1_score:.2f}")
    print(f"Model2      -> Label: {mod2_label}, Score: {mod2_score:.2f}")
    print(f"Final Decision (weighted): {final_label}")
    print("="*80)



# --- Summary ---
total = len(test_samples)
wrong_prec = 100*(overall_wrong/total)
uncertain_perc = 100*(total_uncertain/total)
print(f"Total Samples: {total}")
print(f"Model1 wrong: {mod1_wrong}/{total}")
print(f"Model2 wrong: {mod2_wrong}/{total}")
print(f"Weighted final wrong: {overall_wrong}/{total}")
print(f"Weighted final wrong percentage: {wrong_prec}")
print(f"Weighted Uncertain percentage: {uncertain_perc}")
