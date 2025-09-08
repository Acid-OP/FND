from transformers import pipeline
from langchain.prompts import PromptTemplate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc


class style_agent:
    def __init__(self):
        self.llm = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3" ,
        # load_in_8bit=True,
        return_full_text=False,
        device_map='auto'
        )


    def create_prompts(self,news_list):  
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
        Score:
        """

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=news)
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


    def run_batch(self,news_batch):


        input_prompts = self.create_prompts(news_batch);
        outputs = self.llm(input_prompts,max_new_tokens=50)
        print(outputs)
        scores = []
        for output in outputs:
            generated_text = output['generated_text'].strip() if 'generated_text' in output else output[0]['generated_text'].strip()
            score = self.extract_score(generated_text)
            scores.append(score)
        


        # scores_array = np.array(scores)

        # min_val = scores_array.min()
        # max_val = scores_array.max()

        # if max_val == min_val:
        #     normalized_scores = np.zeros_like(scores_array)
        # else:
        #     normalized_scores = (scores_array-min_val)/(max_val-min_val)
    
        print(scores)
        return scores
    


class NewsDataset(Dataset):
    def __init__(self,dataframe):
        self.data_frame = dataframe

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self,idx):
        row = self.data_frame.iloc[idx]
        text = row['text']
        return text

def main():

    total_samples = 10 # Hard coded for now

    file_real = pd.read_csv('./Dataset/True.csv',nrows=5);
    file_fake = pd.read_csv('./Dataset/Fake.csv',nrows=5);
   

    fake_df = pd.DataFrame(file_fake)
    real_df = pd.DataFrame(file_real)
    
    cols = ['text']
    real_sub = real_df[cols]
    fake_sub = fake_df[cols]

    all_scores = []
    all_labels = []

    agent = style_agent()

    fake_dataset = NewsDataset(fake_sub)
    fake_dataloader = DataLoader(fake_dataset,batch_size=20)

    real_dataset = NewsDataset(real_sub)
    real_dataloader = DataLoader(real_dataset,batch_size=20)

# 1 = FAKE
# 0 = REAL

    fake_correct = 0
    fake_wrong = 0

    for text in fake_dataloader:
        text_scores = agent.run_batch(text)
        all_scores.extend(text_scores)
        all_labels.extend([1]*len(text_scores))
        
    real_correct = 0
    real_wrong = 0


    for text in real_dataloader:
        text_scores = agent.run_batch(text)
        all_scores.extend(text_scores)
        all_labels.extend([0]*len(text_scores))
        

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    # roc_auc = auc(fpr, tpr)

    threshold_idx = np.argmax(tpr-fpr)
    THRESHOLD = thresholds[threshold_idx]

    for score in all_scores[:len(fake_sub)]:
        if score >= THRESHOLD:
            fake_correct += 1
        else:
            fake_wrong += 1
    
    for score in all_scores[len(real_sub):]:
        if score < THRESHOLD:
            real_correct += 1
        else:
            real_wrong += 1


    total_correct = fake_correct + real_correct
    print("Accuracy: ",100*( total_correct/total_samples),"\n")

    
    print("fake correct: ", fake_correct)
    print("fake wrong: ", fake_wrong, "\n")

    print("Real correct: ", real_correct)
    print("Real wrong: ", real_wrong)

if __name__ == "__main__":
    main()
