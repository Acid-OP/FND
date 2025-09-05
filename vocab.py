from transformers import pipeline, AutoTokenizer
from langchain.prompts import PromptTemplate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
import torch


class VocabAgent:
    def __init__(self):
        model_id = "google/flan-t5-small"  # lightweight model (~120MB)
        self.llm = pipeline(
            "text2text-generation",
            model=model_id,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=10,
            truncation=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def create_prompts(self, news_list):
        prompts = []
        template = """You are an expert in detecting fake news through vocabulary.
Give a score between 0.00 and 1.00 (two decimals).

Rules:
- 0.00 = Neutral, factual, professional → REAL
- 1.00 = Sensational, emotional, conspiratorial → FAKE

Example:
Article: "SHOCKING! The government HIDES the TRUTH about vaccines!"
Score: 0.95

Now analyze this article:
{n_ews}

Return only the score:"""

        for news in news_list:
            # Truncate article text to stay under 512 tokens
            tokens = self.tokenizer(news, truncation=True, max_length=400)
            truncated_text = self.tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=truncated_text)
            prompts.append(final_prompt)

        return prompts

    def run_batch(self, news_batch):
        input_prompts = self.create_prompts(news_batch)
        outputs = self.llm(input_prompts, batch_size=len(input_prompts))  # batched on GPU

        scores = []
        for output in outputs:
            generated_text = output["generated_text"].strip()
            found_numbers = re.findall(r'(\d+\.\d+)', generated_text)

            if found_numbers:
                score = float(found_numbers[0])
                score = max(0.0, min(1.0, score))  # clamp to [0,1]
                scores.append(score)
            else:
                scores.append(0.5)  # fallback neutral

        return scores


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data_frame = dataframe

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        return row["text"]


def main():
    total_samples = 200  # Hard coded

    file_real = pd.read_csv('./Dataset/True.csv', nrows=100)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=100)

    cols = ['text']
    real_sub = file_real[cols]
    fake_sub = file_fake[cols]

    all_scores = []
    all_labels = []

    agent = VocabAgent()

    fake_dataset = NewsDataset(fake_sub)
    fake_dataloader = DataLoader(fake_dataset, batch_size=10)

    real_dataset = NewsDataset(real_sub)
    real_dataloader = DataLoader(real_dataset, batch_size=10)

    fake_correct = 0
    fake_wrong = 0

    print("Processing fake news...")
    for text in fake_dataloader:
        text_scores = agent.run_batch(text)
        all_scores.extend(text_scores)
        all_labels.extend([1] * len(text_scores))

    real_correct = 0
    real_wrong = 0

    print("Processing real news...")
    for text in real_dataloader:
        text_scores = agent.run_batch(text)
        all_scores.extend(text_scores)
        all_labels.extend([0] * len(text_scores))

    # ROC & threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    if len(thresholds) > 1:
        threshold_idx = np.argmax(tpr - fpr)
        THRESHOLD = thresholds[threshold_idx]
    else:
        THRESHOLD = 0.5  # fallback if ROC fails

    # Count accuracy
    for score in all_scores[:len(fake_sub)]:
        if score >= THRESHOLD:
            fake_correct += 1
        else:
            fake_wrong += 1

    for score in all_scores[len(fake_sub):]:
        if score < THRESHOLD:
            real_correct += 1
        else:
            real_wrong += 1

    total_correct = fake_correct + real_correct
    print("\n=== VOCABULARY FAKE NEWS DETECTION ===")
    print(f"Accuracy: {100 * total_correct / total_samples:.2f}%")
    print(f"ROC AUC: {roc_auc:.3f}")
    print(f"Optimal Threshold: {THRESHOLD:.3f}")
    print(f"Fake correct: {fake_correct}, Fake wrong: {fake_wrong}")
    print(f"Real correct: {real_correct}, Real wrong: {real_wrong}")


if __name__ == "__main__":
    main()
