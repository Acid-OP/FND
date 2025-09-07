from transformers import pipeline, AutoTokenizer
from langchain.prompts import PromptTemplate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
import torch


class SentimentAgent:
    def __init__(self, batch_size=8):
        self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.batch_size = batch_size

        self.llm = pipeline(
            "text-generation",
            model=self.model_id,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=20,
            truncation=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        # template defined once (not in loop)
        self.template = """You are an expert sentiment analyzer.
Rate the emotional intensity of the following news article.

Give only a score between 0.00 and 1.00 (two decimals):
- 0.00 = completely neutral/objective
- 1.00 = extremely emotional/sensational

Article: {n_ews}

Return only the score:"""

    def make_batch_prompts(self, texts):
        prompts = []
        for news in texts:
            # truncate for safety
            tokens = self.tokenizer(news, truncation=True, max_length=300)
            truncated_text = self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

            prompt = PromptTemplate.from_template(self.template)
            final_prompt = prompt.format(n_ews=truncated_text)
            prompts.append(final_prompt)
        return prompts

    def run_batch(self, texts):
        prompts = self.make_batch_prompts(texts)
        outputs = self.llm(prompts, batch_size=self.batch_size)

        scores = []
        for out in outputs:
            # pipeline with list input returns list[dict]
            generated_text = out[0]["generated_text"].strip()
            found_numbers = re.findall(r'(\d+\.\d+)', generated_text)
            if found_numbers:
                score = float(found_numbers[0])
                score = max(0.0, min(1.0, score))
                scores.append(score)
            else:
                scores.append(0.5)
        return scores


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]["text"]


def main():
    total_samples = 200

    # load dataset
    file_real = pd.read_csv("./Dataset/True.csv", nrows=100)
    file_fake = pd.read_csv("./Dataset/Fake.csv", nrows=100)

    real_sub = file_real[["text"]]
    fake_sub = file_fake[["text"]]

    # init
    agent = SentimentAgent(batch_size=8)
    fake_dataset = NewsDataset(fake_sub)
    real_dataset = NewsDataset(real_sub)

    fake_loader = DataLoader(fake_dataset, batch_size=8)
    real_loader = DataLoader(real_dataset, batch_size=8)

    all_scores, all_labels = [], []

    print("Processing fake news in batches...")
    for batch in fake_loader:
        scores = agent.run_batch(batch)
        all_scores.extend(scores)
        all_labels.extend([1] * len(scores))

    print("Processing real news in batches...")
    for batch in real_loader:
        scores = agent.run_batch(batch)
        all_scores.extend(scores)
        all_labels.extend([0] * len(scores))

    # ROC & threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    THRESHOLD = thresholds[np.argmax(tpr - fpr)] if len(thresholds) > 1 else 0.5

    # Accuracy
    fake_correct = sum(score >= THRESHOLD for score in all_scores[:len(fake_sub)])
    fake_wrong = len(fake_sub) - fake_correct
    real_correct = sum(score < THRESHOLD for score in all_scores[len(fake_sub):])
    real_wrong = len(real_sub) - real_correct

    print("\n=== SENTIMENT FAKE NEWS DETECTION ===")
    print(f"Accuracy: {100 * (fake_correct + real_correct) / total_samples:.2f}%")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Optimal Threshold: {THRESHOLD:.3f}")
    print(f"Fake correct: {fake_correct}, Fake wrong: {fake_wrong}")
    print(f"Real correct: {real_correct}, Real wrong: {real_wrong}")


if __name__ == "__main__":
    main()
