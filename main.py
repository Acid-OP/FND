import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

# Function to call Gemini
def call_gemini(text, model="gemini-1.5-flash"):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": text}]}]}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    else:
        return f"Error {response.status_code}: {response.text}"

# Clickbait scoring example
def score_clickbait(text):
    prompt = f"""
    Rate the following text for clickbait style (0 = not clickbait, 10 = extremely clickbait).
    Return only the number.

    Text: "{text}"
    """
    return call_gemini(prompt)

if __name__ == "__main__":
    test_text = "Breaking news! You won’t believe what scientists just discovered about coffee!!!"
    print("Original text:", test_text)

    clickbait_score = score_clickbait(test_text)
    print("Clickbait score:", clickbait_score)
