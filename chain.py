from transformers import pipeline
from langchain.prompts import PromptTemplate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple

class BossAgent:
    def __init__(self , agent_weights: Dict[str, float] = None):
        self.agents = {
            'style': StyleAgent(),
            'vocab': VocabAgent(),
            'sentiment': SentimentAgent(),
            'semantics': SemanticsAgent()
        }
        self.weights = self._validate_agent_weights(agent_weights)
        
    def _validate_agent_weights(self, agent_weights: Dict[str, float]) -> Dict[str, float]:
        if agent_weights is None:
            raise ValueError("Please provide weights for all agents. Expected agents: style, vocab, sentiment, semantics")
        
        required_agents = set(self.agents.keys())
        provided_agents = set(agent_weights.keys())
        
        if required_agents != provided_agents:
            missing = required_agents - provided_agents
            extra = provided_agents - required_agents
            error_msg = []
            if missing:
                error_msg.append(f"Missing weights for: {list(missing)}")
            if extra:
                error_msg.append(f"Unexpected agents: {list(extra)}")
            raise ValueError("Weight configuration error. " + " | ".join(error_msg))
        
        total_weight = sum(agent_weights.values())
        if abs(total_weight - 1.0) > 0.001:
            if total_weight > 1.0:
                raise ValueError(f"Weights sum to {total_weight:.3f} which exceeds 1.0. Please reduce the weights so they sum to exactly 1.0")
            else:
                raise ValueError(f"Weights sum to {total_weight:.3f} which is less than 1.0. Please increase the weights so they sum to exactly 1.0")
        
        negative_weights = [k for k, v in agent_weights.items() if v < 0]
        if negative_weights:
            raise ValueError(f"Negative weights not allowed for agents: {negative_weights}")
        
        return agent_weights.copy()
    
class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data_frame = dataframe
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = row['text']
        return text    
def main():
    total_samples = 10
    file_real = pd.read_csv('./Dataset/True.csv', nrows=5)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=5)

    fake_df = pd.DataFrame(file_fake)
    real_df = pd.DataFrame(file_real)
    
    cols = ['text']
    real_sub = real_df[cols]
    fake_sub = fake_df[cols]

    all_scores = []
    all_labels = []
    
    agent_weights = {
        'style': 0.3,
        'vocab': 0.2,
        'sentiment': 0.25,
        'semantics': 0.25
    }
    BOSS = BossAgent(agent_weights)

    fake_dataset = NewsDataset(fake_sub)
    fake_dataloader = DataLoader(fake_dataset, batch_size=20)

    real_dataset = NewsDataset(real_sub)
    real_dataloader = DataLoader(real_dataset, batch_size=20)

    all_weighted_scores = []
    all_labels = []
    all_individual_scores = {agent: [] for agent in BOSS.agents.keys()}
