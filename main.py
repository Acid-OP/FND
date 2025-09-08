from transformers import pipeline
from langchain.prompts import PromptTemplate
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple

class BaseAgent:
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.llm = pipeline(
            "text-generation",
            model="Qwen/Qwen2.5-0.5B-Instruct",
            return_full_text=False,
            device_map='auto'
        )
    
    def create_prompts(self, news_list: List[str]) -> List[str]:
        raise NotImplementedError("Each agent must implement create_prompts method")
    
    def extract_score(self, generated_text: str) -> float:
        """Common score extraction logic"""
        score_pattern = r'score:\s*([0-1]?\.?\d{1,2})'
        match = re.search(score_pattern, generated_text.lower())
        
        if match:
            try:
                score = float(match.group(1))
                return min(1.0, max(0.0, score))
            except (ValueError, IndexError):
                pass
        
        decimal_pattern = r'([0-1]?\.\d{1,2})'
        decimal_matches = re.findall(decimal_pattern, generated_text)
        
        if decimal_matches:
            try:
                score = float(decimal_matches[-1])
                return min(1.0, max(0.0, score))
            except ValueError:
                pass
        
        return -1.0
    
    def run_batch(self, news_batch: List[str]) -> List[float]:
        # Common batch processing logic
        input_prompts = self.create_prompts(news_batch)
        outputs = self.llm(input_prompts, max_new_tokens=50)
        
        scores = []
        for output in outputs:
            generated_text = output['generated_text'].strip() if 'generated_text' in output else output[0]['generated_text'].strip()
            score = self.extract_score(generated_text)
            scores.append(score)
        
        print(f"{self.agent_name} scores: {scores}")
        return scores

class StyleAgent(BaseAgent):
    def __init__(self):
        super().__init__("Style Agent")
    
    def create_prompts(self, news_list: List[str]) -> List[str]:
        prompts = []
        template =  """You are a highly skilled and impartial news style analyst. Your only task is to evaluate how professional the **writing style** of a news article is, based only on tone, clarity, and objectivity. Ignore the truthfulness or factual accuracy of the content.

        Process:
        1.  **Reasoning:** Identify key stylistic elements (e.g., formal/informal, objective/subjective, balanced/biased, factual/sensational). Be concise.
        2.  **Score:** Provide a single, final style score on a scale of 0.00 to 1.00 (with two decimal places).

        Rules:
        - 0.00 = Exemplary professional news style (formal, objective, balanced, factual presentation).
        - 1.00 = Flagrantly unprofessional news style (sensationalist, emotional, highly biased, sloppy language).

        Example 1: (Professional Style)
        Article: "The central bank announced a modest interest rate increase on Thursday, citing inflationary concerns and global market volatility."
        Reasoning: Tone is neutral and formal; objective presentation of facts.
        Score: 0.08

        Example 2: (Unprofessional Style)
        Article: "The government is totally out of control! These corrupt leaders are destroying the country and laughing at us."
        Reasoning: Highly emotional, subjective, and uses inflammatory language.
        Score: 0.85

        Article: "{n_ews}"
        Reasoning:
        Score:
        """
        
        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=news)
            prompts.append(final_prompt)
        
        return prompts

class VocabAgent(BaseAgent):
    def __init__(self):
        super().__init__("Vocabulary Agent")
    
    def create_prompts(self, news_list: List[str]) -> List[str]:
        prompts = []
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

        Article: {n_ews}
        Score:
        """
        
        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=news)
            prompts.append(final_prompt)
        
        return prompts

class SentimentAgent(BaseAgent):
    def __init__(self):
        super().__init__("Sentiment Agent")
    
    def create_prompts(self, news_list: List[str]) -> List[str]:
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

class SemanticsAgent(BaseAgent):
    def __init__(self):
        super().__init__("Semantics Agent")
    
    def create_prompts(self, news_list: List[str]) -> List[str]:
        # Replace with your semantics-specific prompt template
        prompts = []
        template = """You are an expert in news analysis. Your task is to evaluate the conceptual coherence and plausibility of a news article's narrative. Your judgment should be based on how well the story's events, context, and implications align with a believable reality.

        Process:
        1.  **Reasoning:** Analyze the narrative's internal logic and plausibility.
        2.  **Score:** Provide a single score from 0.00 (highly credible) to 1.00 (likely fabricated).

        Example 1 (Credible):
        Article: "NASA Confirms Artemis II Astronaut Crew, Aims for 2024 Launch"
        Reasoning: This mission and crew selection are consistent with publicly known facts and NASA's plans.
        Score: 0.15

        Example 2 (Fabricated):
        Article: "Scientists Discover Method for Faster-Than-Light Travel"
        Reasoning: The claim contradicts established physics.
        Score: 0.98

        Article: "{n_ews}"
        Reasoning:
        Score:  
        """
        
        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(n_ews=news)
            prompts.append(final_prompt)
        
        return prompts

class MultiAgentDetector:
    def __init__(self, agent_weights: Dict[str, float] = None):
        """
        Initialize the multi-agent system with configurable weights
        
        Args:
            agent_weights: Dictionary mapping agent names to weights
                          Default: equal weights for all agents
        """
        self.agents = {
            'style': StyleAgent(),
            'vocab': VocabAgent(),
            'sentiment': SentimentAgent(),
            'semantics': SemanticsAgent()
        }
        # checks for weights
        self.weights = self._validate_agent_weights(agent_weights)
        print(f"Agent weights validated successfully: {self.weights}")
    
    def _validate_agent_weights(self, agent_weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Validate agent weights for correctness and completeness
        
        Args:
            agent_weights: Dictionary mapping agent names to weights
            
        Returns:
            Dictionary of validated weights
            
        Raises:
            ValueError: If weights are invalid, missing, or incorrectly configured
        """
        if agent_weights is None:
            raise ValueError("Please provide weights for all agents. Expected agents: style, vocab, sentiment, semantics")
        
        # Check if all required agents have weights
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
        
        # Check if weights sum to exactly 1.0
        total_weight = sum(agent_weights.values())
        if abs(total_weight - 1.0) > 0.001:  # Small tolerance for floating point errors
            if total_weight > 1.0:
                raise ValueError(f"Weights sum to {total_weight:.3f} which exceeds 1.0. Please reduce the weights so they sum to exactly 1.0")
            else:
                raise ValueError(f"Weights sum to {total_weight:.3f} which is less than 1.0. Please increase the weights so they sum to exactly 1.0")
        
        # Check for negative weights
        negative_weights = [k for k, v in agent_weights.items() if v < 0]
        if negative_weights:
            raise ValueError(f"Negative weights not allowed for agents: {negative_weights}")
        
        return agent_weights.copy()
    
    def run_all_agents(self, news_batch: List[str]) -> Tuple[List[float], Dict[str, List[float]]]:
        # Returns:Tuple of (weighted_scores, individual_agent_scores)
        agent_scores = {}
        
        # agent looping
        for agent_name, agent in self.agents.items():
            print(f"\nRunning {agent_name} agent...")
            scores = agent.run_batch(news_batch)
            agent_scores[agent_name] = scores
        
        # Calculate weighted scores
        weighted_scores = self._calculate_weighted_scores(agent_scores)
        
        return weighted_scores, agent_scores
    
    def _calculate_weighted_scores(self, agent_scores: Dict[str, List[float]]) -> List[float]:
        """Calculate weighted average scores across all agents"""
        num_samples = len(next(iter(agent_scores.values())))
        weighted_scores = []
        
        for i in range(num_samples):
            weighted_score = 0.0
            for agent_name, scores in agent_scores.items():
                if scores[i] != -1.0:  
                    weighted_score += scores[i] * self.weights[agent_name]
            weighted_scores.append(weighted_score)
        
        return weighted_scores

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
    agent_weights = {
        'style': 0.3,
        'vocab': 0.2,     
        'sentiment': 0.25, 
        'semantics': 0.25  
    }
    
    file_real = pd.read_csv('./Dataset/True.csv', nrows=5)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=5)
    fake_df = pd.DataFrame(file_fake)
    real_df = pd.DataFrame(file_real)
    
    cols = ['text']
    real_sub = real_df[cols]
    fake_sub = fake_df[cols] 

    detector = MultiAgentDetector(agent_weights)
    fake_dataset = NewsDataset(fake_sub)
    fake_dataloader = DataLoader(fake_dataset, batch_size=20)  
    real_dataset = NewsDataset(real_sub)
    real_dataloader = DataLoader(real_dataset, batch_size=20)

    all_weighted_scores = []
    all_labels = []
    all_individual_scores = {agent: [] for agent in detector.agents.keys()}
# 1 = FAKE
# 0 = REAL
    for text in fake_dataloader:
        weighted_scores, individual_scores = detector.run_all_agents(text)
        all_weighted_scores.extend(weighted_scores)
        all_labels.extend([1] * len(weighted_scores))  
        
        for agent_name, scores in individual_scores.items():
            all_individual_scores[agent_name].extend(scores)
    
    for text in real_dataloader:
        weighted_scores, individual_scores = detector.run_all_agents(text)
        all_weighted_scores.extend(weighted_scores)
        all_labels.extend([0] * len(weighted_scores)) 
        
        for agent_name, scores in individual_scores.items():
            all_individual_scores[agent_name].extend(scores)
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_weighted_scores)
    roc_auc = auc(fpr, tpr)
    
    threshold_idx = np.argmax(tpr - fpr)
    THRESHOLD = thresholds[threshold_idx]
    
    fake_correct = 0
    fake_wrong = 0
    real_correct = 0
    real_wrong = 0
    
    # Check fake news predictions
    for score in all_weighted_scores[:len(fake_sub)]:
        if score >= THRESHOLD:
            fake_correct += 1
        else:
            fake_wrong += 1
    
    # Check real news predictions
    for score in all_weighted_scores[len(fake_sub):]:
        if score < THRESHOLD:
            real_correct += 1
        else:
            real_wrong += 1
    
    total_correct = fake_correct + real_correct
    
    # Print results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)
    print(f"Agent Weights Used: {detector.weights}")
    print(f"Optimal Threshold: {THRESHOLD:.3f}")
    print(f"ROC AUC Score: {roc_auc:.3f}")
    print(f"Overall Accuracy: {100 * (total_correct / total_samples):.2f}%")
    print("\nDetailed Results:")
    print(f"Fake News - Correct: {fake_correct}, Wrong: {fake_wrong}")
    print(f"Real News - Correct: {real_correct}, Wrong: {real_wrong}")
    
    # Print individual agent performance for analysis
    print("\n" + "=" * 50)
    print("INDIVIDUAL AGENT ANALYSIS")
    print("=" * 50)
    
    for agent_name in detector.agents.keys():
        agent_scores = all_individual_scores[agent_name]
        fpr_agent, tpr_agent, thresholds_agent = roc_curve(all_labels, agent_scores)
        roc_auc_agent = auc(fpr_agent, tpr_agent)
        print(f"{agent_name.capitalize()} Agent - ROC AUC: {roc_auc_agent:.3f}")

if __name__ == "__main__":
    main()