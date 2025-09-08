from transformers import pipeline
from langchain.prompts import PromptTemplate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve
import torch


class sentiment_agent:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            device=0 if torch.cuda.is_available() else -1,
            return_full_text=False,
            max_new_tokens=10,   # allow space for decimal output
            do_sample=False      # deterministic output
        )

    def create_prompts(self, news_list):
        prompts = []
        template = """You are an expert sentiment analyzer. 
Judge how emotionally charged the following news article is. 
Return ONLY a decimal number between 0.00 and 1.00 (two decimals):

- 0.00 → completely neutral/objective
- 1.00 → extremely emotional/sensational

Examples:
Article: "Government passes new law after parliamentary debate."
Score: 0.05

Article: "Shocking betrayal! Citizens furious after leader’s outrageous scandal!"
Score: 0.95

Article: "Scientists discover new exoplanet orbiting a nearby star."
Score: 0.45

Article: {n_ews}
Score:"""

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=news)
            prompts.append(final_prompt)

        return prompts

    def run_batch(self, news_batch):
        input_prompts = self.create_prompts(news_batch)
        outputs = self.llm(input_prompts)

        scores = []
        for output in outputs:
            generated_text = output[0]['generated_text'].strip()
            found_numbers = re.findall(r'(0\.\d{2}|1\.00)', generated_text)

            if found_numbers:
                score = float(found_numbers[0])
                score = max(0.0, min(1.0, score))
                scores.append(score)
            else:
                scores.append(0.5)  # fallback if parsing fails

        return scores


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data_frame = dataframe

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        return self.data_frame.iloc[idx]['text']


def main():
    total_samples = 200

    file_real = pd.read_csv('./Dataset/True.csv', nrows=100)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=100)

    real_sub = file_real[['text']]
    fake_sub = file_fake[['text']]

    all_scores = []
    all_labels = []

    agent = sentiment_agent()

    fake_dataset = NewsDataset(fake_sub)
    fake_dataloader = DataLoader(fake_dataset, batch_size=20)

    real_dataset = NewsDataset(real_sub)
    real_dataloader = DataLoader(real_dataset, batch_size=20)

    print("\n=== Processing Fake News ===")
    idx = 1
    fake_scores = []
    for batch in fake_dataloader:
        text_scores = agent.run_batch(batch)
        for score in text_scores:
            print(f"Fake article {idx} score: {score:.2f}")
            idx += 1
        fake_scores.extend(text_scores)
        all_scores.extend(text_scores)
        all_labels.extend([1] * len(text_scores))

    print("\n=== Processing Real News ===")
    idx = 1
    real_scores = []
    for batch in real_dataloader:
        text_scores = agent.run_batch(batch)
        for score in text_scores:
            print(f"Real article {idx} score: {score:.2f}")
            idx += 1
        real_scores.extend(text_scores)
        all_scores.extend(text_scores)
        all_labels.extend([0] * len(text_scores))

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    threshold_idx = np.argmax(tpr - fpr)
    THRESHOLD = thresholds[threshold_idx]

    # 🔥 Safe fallback if threshold invalid
    if np.isinf(THRESHOLD) or np.isnan(THRESHOLD):
        THRESHOLD = 0.5

    avg_fake = np.mean(fake_scores)
    avg_real = np.mean(real_scores)
    print(f"\nAverage fake score: {avg_fake:.3f}")
    print(f"Average real score: {avg_real:.3f}")
    print(f"Threshold: {THRESHOLD:.3f}")

    # 🔥 Auto-orientation
    if avg_fake > avg_real:
        print("Orientation: Higher score = Fake")
        fake_correct = sum(score >= THRESHOLD for score in fake_scores)
        fake_wrong = len(fake_scores) - fake_correct
        real_correct = sum(score < THRESHOLD for score in real_scores)
        real_wrong = len(real_scores) - real_correct
    else:
        print("Orientation: Higher score = Real")
        fake_correct = sum(score < THRESHOLD for score in fake_scores)
        fake_wrong = len(fake_scores) - fake_correct
        real_correct = sum(score >= THRESHOLD for score in real_scores)
        real_wrong = len(real_scores) - real_correct

    total_correct = fake_correct + real_correct
    print("\n=== FINAL METRICS ===")
    print(f"Accuracy: {100 * (total_correct / total_samples):.2f}%")
    print(f"Fake correct: {fake_correct}, Fake wrong: {fake_wrong}")
    print(f"Real correct: {real_correct}, Real wrong: {real_wrong}")


if __name__ == "__main__":
    main()
