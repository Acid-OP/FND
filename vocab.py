from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
import torch

class VocabAgent:
    def __init__(self):
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load model with proper device handling
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", dtype=torch.float16)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float32)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create pipeline without device parameter when using device_map
        pipeline_kwargs = {
            "model": self.model, "tokenizer": self.tokenizer, "max_new_tokens": 50,
            "do_sample": True, "temperature": 0.7, "top_p": 0.9, 
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        # Only add device if not using device_map
        if not torch.cuda.is_available():
            pipeline_kwargs["device"] = -1
            
        self.llm = pipeline("text-generation", **pipeline_kwargs)

    def create_prompts(self, news_list):
        template = """<|im_start|>system
You are an expert in detecting fake news through vocabulary analysis. Your task is to score news articles based on their language patterns.

Scoring rules:
- 0.00-0.30: Professional, factual, neutral language → REAL news
- 0.70-1.00: Sensational, emotional, conspiratorial language → FAKE news
- 0.31-0.69: Mixed characteristics

Examples:
Article: "Scientists at Harvard University published findings in Nature journal showing climate patterns."
Score: 0.15

Article: "SHOCKING! Government COVERS UP the REAL truth! Wake up sheeple!"
Score: 0.95
<|im_end|>

<|im_start|>user
Analyze this news article and provide ONLY a score between 0.00 and 1.00:

Article: {news_text}

Score:<|im_end|>
<|im_start|>assistant
"""
        prompts = []
        for news in news_list:
            tokens = self.tokenizer(news, truncation=True, max_length=300, return_tensors="pt")
            truncated_text = self.tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
            if len(truncated_text) > 500:
                truncated_text = truncated_text[:500] + "..."
            prompts.append(template.format(news_text=truncated_text))
        return prompts

    def extract_score(self, generated_text):
        print(f"Generated text: {generated_text[:200]}...")
        response_part = generated_text.split("<|im_start|>assistant")[-1] if "<|im_start|>assistant" in generated_text else generated_text
        
        # Try multiple score patterns
        for pattern in [r'Score:\s*(\d*\.?\d+)', r'(\d\.\d{2})', r'(\d\.\d{1})', r'(\d\.\d+)', r'(\d+\.\d+)', r'(\d+)']:
            matches = re.findall(pattern, response_part, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[0])
                    if score > 1.0:
                        score = score / 100.0 if score <= 100 else 1.0
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        
        # Keyword-based fallback
        response_lower = response_part.lower()
        fake_count = sum(1 for word in ['fake', 'false', 'sensational', 'bias', 'misleading', 'propaganda'] if word in response_lower)
        real_count = sum(1 for word in ['real', 'true', 'factual', 'neutral', 'professional', 'legitimate'] if word in response_lower)
        
        if fake_count > real_count: return 0.8
        elif real_count > fake_count: return 0.2
        return np.random.choice([0.3, 0.4, 0.6, 0.7])

    def run_batch(self, news_batch):
        scores = []
        for prompt in self.create_prompts(news_batch):
            try:
                output = self.llm(prompt, max_new_tokens=50, do_sample=True, temperature=0.8, 
                                 top_p=0.9, repetition_penalty=1.1, return_full_text=False)
                score = self.extract_score(output[0]["generated_text"].strip())
                scores.append(score)
            except Exception as e:
                print(f"Error processing prompt: {e}")
                scores.append(0.5)
        return scores

class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data_frame = dataframe
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        return self.data_frame.iloc[idx]["text"]

def main():
    # Load 5 samples from each dataset
    file_real = pd.read_csv('./Dataset/True.csv', nrows=5)[['text']]
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=5)[['text']]
    print(f"Loaded {len(file_fake)} fake and {len(file_real)} real news samples")

    agent = VocabAgent()
    all_scores, all_labels = [], []

    # Process both datasets
    for dataset, label, name in [(file_fake, 1, "fake"), (file_real, 0, "real")]:
        dataloader = DataLoader(NewsDataset(dataset), batch_size=2)
        print(f"\nProcessing {name} news...")
        for i, text_batch in enumerate(dataloader):
            print(f"Processing {name} batch {i+1}")
            scores = agent.run_batch(text_batch)
            all_scores.extend(scores)
            all_labels.extend([label] * len(scores))

    print(f"\nAll scores: {all_scores}")
    print(f"All labels: {all_labels}")

    # Calculate ROC and threshold
    if len(set(all_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        THRESHOLD = thresholds[np.argmax(tpr - fpr)] if len(thresholds) > 1 else 0.5
    else:
        roc_auc, THRESHOLD = 0.5, 0.5
        print("Warning: Only one class present, using default threshold")

    # Calculate accuracy
    fake_scores, real_scores = all_scores[:5], all_scores[5:]
    fake_correct = sum(1 for score in fake_scores if score >= THRESHOLD)
    real_correct = sum(1 for score in real_scores if score < THRESHOLD)
    total_correct = fake_correct + real_correct

    print("\n" + "="*50)
    print("VOCABULARY FAKE NEWS DETECTION - Qwen2.5-0.5B")
    print("="*50)
    print(f"Total samples processed: 10")
    print(f"Accuracy: {100 * total_correct / 10:.2f}%")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Optimal Threshold: {THRESHOLD:.3f}")
    print(f"\nFake News Results: {fake_correct}/5 correct, {5-fake_correct}/5 wrong")
    print(f"Real News Results: {real_correct}/5 correct, {5-real_correct}/5 wrong")
    print(f"\nDetailed Scores:")
    print(f"Fake news scores: {fake_scores}")
    print(f"Real news scores: {real_scores}")

if __name__ == "__main__":
    main()