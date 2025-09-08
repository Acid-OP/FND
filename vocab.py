from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
import torch
from langchain.prompts import PromptTemplate

class VocabAgent:
    def __init__(self):
        self.llm = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3" ,
        device_map='auto'
        )

    def create_prompts(self, news_list):
        template = """
        You are an expert in detecting fake news through vocabulary analysis. Your task is to score news articles based on their language patterns.

        Scoring rules:
        - 0.00: Professional, factual, neutral language → REAL news
        - 1.00: Sensational, emotional, conspiratorial language → FAKE news

        Examples:
        Article: "Scientists at Harvard University published findings in Nature journal showing climate patterns."
        Score: 0.15

        Article: "SHOCKING! Government COVERS UP the REAL truth! Wake up sheeple!"
        Score: 0.95
     
        Analyze this news article and provide ONLY a score between 0.00 and 1.00:

        Article: {news_text}
        Score:
        """
        prompts = []
        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(news_text=news)
            prompts.append(final_prompt)
        return prompts

    def extract_score(self, generated_text):
        score_pattern = r'score:\s*([0-1]?\.?\d{1,2})'
        match = re.search(score_pattern, generated_text.lower())
        
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            except (ValueError, IndexError):
                pass # Fall through to the next check

        # Fallback to find any float in the text, but only if the first check fails.
        decimal_pattern = r'([0-1]?\.\d{1,2})'
        decimal_matches = re.findall(decimal_pattern, generated_text)

        if decimal_matches:
            try:
                score = float(decimal_matches[-1])
                return min(1.0, max(0.0, score))
            except ValueError:
                pass

        # Return a fallback value that indicates an error, not a neutral score.
        # This will allow you to diagnose the problem later if it persists.
        return -1.0 


    def run_batch(self, news_batch):
        scores = []
        for prompt in self.create_prompts(news_batch):
            try:
                output = self.llm(prompt, max_new_tokens=50)
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