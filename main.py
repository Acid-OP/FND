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
        model="Qwen/Qwen2.5-0.5B-Instruct" ,
        return_full_text=False
        )


    def create_prompts(self,news_list):  
        prompts = []
        template = """You are an expert news stylist, responsible for judging the professionalism and reliability of news articles. Your score should reflect how closely the text resembles the style of an established news publisher (e.g., Reuters, BBC, Associated Press).

        Score on a scale of 0.00 to 1.00, to two decimal places:
        - 0.00: Perfect stylistic match to a reliable publisher. (e.g., highly formal, objective, professional tone)
        - 1.00: No resemblance to a reliable publisher. (e.g., sensationalist, opinionated, poor grammar)

        Example 1:
        Article: "Breaking news: sources say the government is in turmoil over a new bill."
        Score: 0.15

        Example 2:
        Article: "The president is a total disaster! He's running the country into the ground and everyone knows it."
        Score: 0.98

        Here is the news to be judged:
        {n_ews}
        """

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=news)
            prompts.append(final_prompt)
    

        return prompts


    def run_batch(self,news_batch):


        input_prompts = self.create_prompts(news_batch);
        outputs = self.llm(input_prompts,max_new_tokens=5)
        
        scores = []
        for output in outputs:
            generated_text = output[0]['generated_text'].strip()
            
            found_numbers = re.findall(r'(\d+\.\d+|\d+)', generated_text)
            
            if found_numbers:
                scores.append(float(found_numbers[0]))
            else:
  
                scores.append(0.0)
        


        scores_array = np.array(scores)

        min_val = scores_array.min()
        max_val = scores_array.max()

        if max_val == min_val:
            normalized_scores = np.zeros_like(scores_array)
        else:
            normalized_scores = (scores_array-min_val)/(max_val-min_val)
    
    
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

    total_samples = 200 # Hard coded for now

    file_real = pd.read_csv('./Dataset/True.csv',nrows=100);
    file_fake = pd.read_csv('./Dataset/Fake.csv',nrows=100);
   

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
