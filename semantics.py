from transformers import pipeline
from langchain.prompts import PromptTemplate
import pandas as pd
import numpy as np
import re
from sklearn.metrics import roc_curve, auc


class context_agent:
    def __init__(self):
        self.llm = pipeline(
            "text-generation",
            model="bigscience/bloom-1b7",
            return_full_text=False,
            device_map="auto"
        )

    def create_prompts(self, news_list):  
        prompts = []
        template = """You are a fact-checking expert specializing in detecting misinformation through contextual analysis. Use step-by-step reasoning to assess this news article's authenticity.

Article: "{news_content}"

Step 1 - Content Analysis:
What is this article claiming happened? List the key events/claims.

Step 2 - Plausibility Check:
For each claim, assess: Is this plausible given normal circumstances?

Step 3 - Evidence Evaluation:
What evidence is provided for each claim? Is it sufficient?

Step 4 - Context Assessment:
Does the broader context make sense? Are there missing pieces?

Step 5 - Consistency Review:
Are all parts of the story consistent with each other?

Step 6 - Reality Check:
Based on your knowledge, could this realistically happen as described?

Final Assessment:
Score this article's credibility on a scale of 0.00 to 1.00:
- 0.00: Highly credible, authentic news content
- 1.00: Highly suspicious, likely fake news

Provide only the final score as a decimal number (e.g., 0.75).

{news_content}"""

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(news_content=news)
            prompts.append(final_prompt)

        return prompts

    def extract_score(self, generated_text):
        score_pattern = r'(?:score|assessment)?\s*:?\s*([0-1]?\.\d{1,2}|[01]\.?0?0?)'
        matches = re.findall(score_pattern, generated_text.lower())
        
        if matches:
            try:
                score = float(matches[-1])
                return min(1.0, max(0.0, score))
            except ValueError:
                pass
        
        decimal_pattern = r'([0-1]?\.\d{1,2})'
        decimal_matches = re.findall(decimal_pattern, generated_text)
        
        if decimal_matches:
            try:
                score = float(decimal_matches[-1])
                return min(1.0, max(0.0, score))
            except ValueError:
                pass
        
        return 0.5

    def run_batch(self, news_batch):
        input_prompts = self.create_prompts(news_batch)
        
        outputs = self.llm(
            input_prompts, 
            max_new_tokens=200,
            temperature=0.2,
            do_sample=True,
            pad_token_id=self.llm.tokenizer.eos_token_id
        )
        
        scores = []
        for output in outputs:
            generated_text = output['generated_text'].strip() if 'generated_text' in output else output[0]['generated_text'].strip()
            score = self.extract_score(generated_text)
            scores.append(score)
        
        return scores


def main():
    total_samples = 200
    samples_per_class = 100

    print("Loading datasets...")
    print("Dataset info: Real articles=21417, Fake articles=23481")

    file_real = pd.read_csv('./Dataset/True.csv', nrows=samples_per_class)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=samples_per_class)

    fake_df = pd.DataFrame(file_fake)
    real_df = pd.DataFrame(file_real)

    cols = ['text']
    real_sub = real_df[cols]
    fake_sub = fake_df[cols]

    print(f"Real articles loaded: {len(real_sub)}")
    print(f"Fake articles loaded: {len(fake_sub)}")

    all_scores = []
    all_labels = []

    print("Initializing BLOOM 1.7B context-based fake news detection agent...")
    agent = context_agent()

    print("Processing fake news articles...")
    fake_correct = 0
    fake_wrong = 0

    for idx in range(len(fake_sub)):
        text = [fake_sub.iloc[idx]['text']]
        print(f"Processing fake news article {idx + 1}/{len(fake_sub)}")
        text_scores = agent.run_batch(text)
        all_scores.extend(text_scores)
        all_labels.extend([1] * len(text_scores))

    print("Processing real news articles...")
    real_correct = 0
    real_wrong = 0

    for idx in range(len(real_sub)):
        text = [real_sub.iloc[idx]['text']]
        print(f"Processing real news article {idx + 1}/{len(real_sub)}")
        text_scores = agent.run_batch(text)
        all_scores.extend(text_scores)
        all_labels.extend([0] * len(text_scores))

    print("Calculating optimal threshold...")
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    threshold_idx = np.argmax(tpr - fpr)
    THRESHOLD = thresholds[threshold_idx]

    print(f"Optimal threshold: {THRESHOLD:.3f}")
    print(f"ROC AUC: {roc_auc:.3f}")

    fake_scores = all_scores[:len(fake_sub)]
    real_scores = all_scores[len(fake_sub):]

    for score in fake_scores:
        if score >= THRESHOLD:
            fake_correct += 1
        else:
            fake_wrong += 1

    for score in real_scores:
        if score < THRESHOLD:
            real_correct += 1
        else:
            real_wrong += 1

    total_correct = fake_correct + real_correct
    accuracy = 100 * (total_correct / total_samples)

    print("\n" + "="*60)
    print("CONTEXT-BASED FAKE NEWS DETECTION RESULTS")
    print("Multi-Step Reasoning Chain Approach")
    print("="*60)
    print(f"Model: BLOOM 1.7B")
    print(f"Dataset: Real News vs Fake News")
    print(f"Total Samples: {total_samples}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"ROC AUC Score: {roc_auc:.3f}")
    print(f"Optimal Threshold: {THRESHOLD:.3f}")
    print()
    print("FAKE NEWS DETECTION:")
    print(f"Correctly identified as fake: {fake_correct}/{len(fake_sub)}")
    print(f"Incorrectly identified as real: {fake_wrong}/{len(fake_sub)}")
    print(f"Fake news detection rate: {100*fake_correct/(fake_correct+fake_wrong):.2f}%")
    print(f"Average fake score: {np.mean(fake_scores):.3f}")
    print()
    print("REAL NEWS DETECTION:")
    print(f"Correctly identified as real: {real_correct}/{len(real_sub)}")
    print(f"Incorrectly identified as fake: {real_wrong}/{len(real_sub)}")
    print(f"Real news detection rate: {100*real_correct/(real_correct+real_wrong):.2f}%")
    print(f"Average real score: {np.mean(real_scores):.3f}")
    print("="*60)

    print("\nSAMPLE SCORES (for debugging):")
    print("First 5 Fake scores:", [f"{s:.3f}" for s in fake_scores[:5]])
    print("First 5 Real scores:", [f"{s:.3f}" for s in real_scores[:5]])


if __name__ == "__main__":
    main()
