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

Process - Follow these steps in order:

Step 1: **Language Formality Analysis**
- Assess vocabulary level (formal vs. colloquial)
- Check for proper grammar and sentence structure
- Note any slang, informal expressions, or unprofessional language
- Identify technical jargon appropriateness

Step 2: **Tone and Emotional Charge**
- Evaluate emotional language vs. neutral reporting
- Check for loaded words, adjectives, or inflammatory terms
- Assess whether tone is measured or sensationalist
- Note use of exclamation points, ALL CAPS, or emphatic punctuation

Step 3: **Objectivity Assessment**
- Identify subjective vs. objective statements
- Look for editorial opinions presented as facts
- Check for balanced perspective vs. one-sided presentation
- Note presence of speculation vs. verified information

Step 4: **Professional Structure**
- Evaluate logical flow and organization
- Check attribution and sourcing indicators
- Assess clarity and readability
- Note adherence to journalism standards

Step 5: **Final Scoring Decision**
- Synthesize findings from all previous steps
- Apply scoring rubric consistently
- Provide final score with justification

Scoring Rubric:
- 0.00-0.20: Exemplary professional style (formal, objective, well-structured)
- 0.21-0.40: Good professional style (minor stylistic issues)
- 0.41-0.60: Adequate but noticeable unprofessional elements
- 0.61-0.80: Poor professional standards (biased, emotional, unclear)
- 0.81-1.00: Flagrantly unprofessional (sensationalist, inflammatory, sloppy)

Example 1: (Professional Style)
Article: "The central bank announced a modest interest rate increase on Thursday, citing inflationary concerns and global market volatility."

Step 1 - Language Formality: Formal vocabulary, proper grammar, professional terminology
Step 2 - Tone and Emotional Charge: Neutral tone, factual presentation, no inflammatory language
Step 3 - Objectivity Assessment: Objective reporting, no editorial opinions, balanced presentation
Step 4 - Professional Structure: Clear, well-organized, appropriate attribution
Step 5 - Final Scoring: All elements indicate high professional standards
Score: 0.08

Example 2: (Unprofessional Style)
Article: "The government is totally out of control! These corrupt leaders are destroying the country and laughing at us."

Step 1 - Language Formality: Informal language ("totally"), emotional expressions
Step 2 - Tone and Emotional Charge: Highly emotional, inflammatory language, exclamation point
Step 3 - Objectivity Assessment: Completely subjective, opinion presented as fact, one-sided
Step 4 - Professional Structure: Poor organization, no attribution, unclear claims
Step 5 - Final Scoring: Multiple serious professional style violations
Score: 0.85

Now analyze this article following the same process:

Article: "{n_ews}"

Step 1 - Language Formality Analysis:

Step 2 - Tone and Emotional Charge:

Step 3 - Objectivity Assessment:

Step 4 - Professional Structure:

Step 5 - Final Scoring Decision:

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
        template = """You are an expert in detecting fake news through vocabulary analysis. Your task is to score news articles based on their language patterns by following a systematic analysis process.

Process - Follow these steps in order:

Step 1: **Credibility Indicators Analysis**
- Identify specific sources, institutions, or publications mentioned
- Check for expert quotes, studies, or official statements
- Note presence of verifiable facts, dates, and locations
- Assess use of evidence-based language vs. unsupported claims

Step 2: **Sensationalism Detection**
- Look for ALL CAPS words, excessive punctuation (!!!, ???)
- Identify clickbait phrases ("SHOCKING!", "You won't believe!", "EXPOSED!")
- Check for emotional manipulation tactics
- Note hyperbolic language or extreme superlatives

Step 3: **Conspiratorial Language Assessment**
- Identify conspiracy theory keywords ("cover-up", "they don't want you to know", "hidden truth")
- Look for us-vs-them mentality language ("sheeple", "wake up", "the elite")
- Check for unsubstantiated claims about powerful entities
- Note appeals to secret knowledge or insider information

Step 4: **Professional Language Evaluation**
- Assess vocabulary sophistication and proper grammar
- Check for neutral, objective reporting tone
- Identify balanced presentation vs. one-sided narratives
- Note adherence to journalistic standards and attribution

Step 5: **Pattern Synthesis and Scoring**
- Combine findings from all previous steps
- Apply scoring rubric based on cumulative evidence
- Consider overall reliability indicators vs. fake news markers

Scoring Rubric:
- 0.00-0.20: Professional, factual, neutral (credible sources, objective tone)
- 0.21-0.40: Mostly reliable with minor sensational elements
- 0.41-0.60: Mixed signals (some professional, some questionable elements)
- 0.61-0.80: Significant fake news indicators (sensational, biased, unsourced)
- 0.81-1.00: Clear fake news patterns (conspiratorial, emotional manipulation)

Example 1: (Reliable News)
Article: "Scientists at Harvard University published findings in Nature journal showing climate patterns."

Step 1 - Credibility Indicators: Specific institution (Harvard), reputable journal (Nature), scientific methodology
Step 2 - Sensationalism Detection: No caps, exclamations, or clickbait language
Step 3 - Conspiratorial Language: No conspiracy theories or manipulative appeals
Step 4 - Professional Language: Neutral tone, proper attribution, factual presentation
Step 5 - Pattern Synthesis: Strong credibility markers, professional standards maintained
Score: 0.15

Example 2: (Fake News)
Article: "SHOCKING! Government COVERS UP the REAL truth! Wake up sheeple!"

Step 1 - Credibility Indicators: No sources, institutions, or verifiable facts provided
Step 2 - Sensationalism Detection: ALL CAPS, exclamation points, "SHOCKING" clickbait
Step 3 - Conspiratorial Language: "COVERS UP", "REAL truth", "Wake up sheeple" - classic conspiracy phrases
Step 4 - Professional Language: Unprofessional, emotional, no journalistic standards
Step 5 - Pattern Synthesis: Multiple fake news indicators, no credibility markers
Score: 0.95

Now analyze this article following the same process:

Article: {n_ews}

Step 1 - Credibility Indicators Analysis:

Step 2 - Sensationalism Detection:

Step 3 - Conspiratorial Language Assessment:

Step 4 - Professional Language Evaluation:

Step 5 - Pattern Synthesis and Scoring:

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
        template = """You are an expert sentiment analyzer. Judge how emotionally charged the following news article is by following a systematic analysis process.

Process - Follow these steps in order:

Step 1: **Emotional Vocabulary Analysis**
- Identify emotionally charged adjectives (shocking, outrageous, incredible, devastating, etc.)
- Look for intense verbs (slams, destroys, explodes, betrays, etc.)
- Note emotional nouns (scandal, crisis, disaster, triumph, etc.)
- Check for emotional amplifiers (extremely, absolutely, completely, etc.)

Step 2: **Punctuation and Typography Assessment**
- Count exclamation marks and question marks
- Identify ALL CAPS words or phrases
- Note excessive punctuation (..., !!!, ???)
- Check for emphatic formatting or unusual typography

Step 3: **Tone and Voice Evaluation**
- Assess whether language is neutral/factual vs. opinionated/emotional
- Identify direct emotional appeals to the reader
- Check for inflammatory or provocative phrasing
- Note use of first/second person vs. third person reporting

Step 4: **Subjective vs. Objective Content**
- Distinguish between factual statements and emotional interpretations
- Identify opinion words vs. neutral reporting
- Check for loaded language that implies judgment
- Note presence of emotional context vs. bare facts

Step 5: **Overall Emotional Impact Assessment**
- Consider cumulative effect of all emotional elements
- Evaluate likely reader emotional response
- Compare against neutral, objective reporting standards
- Synthesize findings into final emotional charge score

Scoring Rubric:
- 0.00-0.20: Completely neutral/objective (dry facts, no emotional language)
- 0.21-0.40: Slightly emotional (mild subjective elements, some human interest)
- 0.41-0.60: Moderately emotional (noticeable emotional language, engaging tone)
- 0.61-0.80: Highly emotional (strong emotional appeals, provocative language)
- 0.81-1.00: Extremely emotional/sensational (overwhelming emotional manipulation)

Example 1: (Low Emotional Charge)
Article: "Government passes new law after parliamentary debate."

Step 1 - Emotional Vocabulary: No emotional adjectives, neutral verbs ("passes"), factual nouns
Step 2 - Punctuation/Typography: Standard punctuation, no caps or exclamations
Step 3 - Tone and Voice: Neutral reporting tone, third person, no emotional appeals
Step 4 - Subjective vs. Objective: Purely factual statement, no opinions or judgments
Step 5 - Overall Impact: Minimal emotional response expected, standard news reporting
Score: 0.05

Example 2: (High Emotional Charge)
Article: "Shocking betrayal! Citizens furious after leader's outrageous scandal!"

Step 1 - Emotional Vocabulary: "Shocking," "betrayal," "furious," "outrageous," "scandal"
Step 2 - Punctuation/Typography: Exclamation mark, emotionally charged presentation
Step 3 - Tone and Voice: Highly emotional tone, inflammatory language, provocative
Step 4 - Subjective vs. Objective: Heavy emotional interpretation, loaded judgments
Step 5 - Overall Impact: Designed to provoke strong emotional response, very sensational
Score: 0.95

Example 3: (Moderate Emotional Charge)
Article: "Scientists discover new exoplanet orbiting a nearby star."

Step 1 - Emotional Vocabulary: "Discover" has mild positive connotation, otherwise neutral
Step 2 - Punctuation/Typography: Standard punctuation, no emotional formatting
Step 3 - Tone and Voice: Professional but engaging, third person reporting
Step 4 - Subjective vs. Objective: Mostly factual with inherent human interest appeal
Step 5 - Overall Impact: Generates interest and wonder but remains professional
Score: 0.10

Now analyze this article following the same process:

Article: {n_ews}

Step 1 - Emotional Vocabulary Analysis:

Step 2 - Punctuation and Typography Assessment:

Step 3 - Tone and Voice Evaluation:

Step 4 - Subjective vs. Objective Content:

Step 5 - Overall Emotional Impact Assessment:

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
        template = """You are an expert in news analysis. Your task is to evaluate the conceptual coherence and plausibility of a news article's narrative by following a systematic analysis process. Your judgment should be based on how well the story's events, context, and implications align with a believable reality.

Process - Follow these steps in order:

Step 1: **Scientific and Technical Plausibility**
- Evaluate claims against established scientific knowledge
- Check for violations of known physical laws or principles
- Assess technical feasibility of described processes or technologies
- Note any extraordinary claims that lack supporting evidence

Step 2: **Logical Consistency Analysis**
- Examine internal logic of the narrative
- Check for contradictory statements or impossible timelines
- Assess cause-and-effect relationships for logical coherence
- Identify any gaps or inconsistencies in the story flow

Step 3: **Contextual Reality Check**
- Compare events to known historical, political, or social context
- Verify alignment with established institutions and procedures
- Check consistency with publicly known facts and timelines
- Assess whether described actors behave realistically

Step 4: **Scale and Impact Assessment**
- Evaluate if claimed consequences match the described events
- Check for proportional responses from institutions or individuals
- Assess whether the scale of claims matches available evidence
- Note any implausibly dramatic or understated reactions

Step 5: **Evidence and Source Credibility**
- Examine types of evidence presented (if any)
- Assess plausibility of quoted sources and their statements
- Check for verifiable details vs. vague assertions
- Note absence of expected corroborating information

Step 6: **Overall Plausibility Synthesis**
- Integrate findings from all previous steps
- Consider cumulative credibility vs. fabrication indicators
- Apply scoring rubric based on comprehensive analysis

Scoring Rubric:
- 0.00-0.20: Highly credible (consistent with reality, logical, well-supported)
- 0.21-0.40: Mostly credible (minor implausibilities, generally believable)
- 0.41-0.60: Questionable (significant concerns, mixed credibility signals)
- 0.61-0.80: Likely fabricated (major implausibilities, poor logic)
- 0.81-1.00: Almost certainly fabricated (impossible claims, complete incoherence)

Example 1: (Credible Narrative)
Article: "NASA Confirms Artemis II Astronaut Crew, Aims for 2024 Launch"

Step 1 - Scientific/Technical Plausibility: Consistent with known space technology and NASA capabilities
Step 2 - Logical Consistency: Timeline and procedures align with established space mission protocols
Step 3 - Contextual Reality Check: Matches NASA's publicly announced Artemis program timeline
Step 4 - Scale and Impact: Appropriate level of institutional response and media coverage
Step 5 - Evidence and Source Credibility: NASA as source is authoritative and verifiable
Step 6 - Overall Synthesis: All elements support high credibility
Score: 0.15

Example 2: (Fabricated Narrative)
Article: "Scientists Discover Method for Faster-Than-Light Travel"

Step 1 - Scientific/Technical Plausibility: Violates Einstein's theory of relativity, contradicts established physics
Step 2 - Logical Consistency: No explanation of how fundamental physics laws are overcome
Step 3 - Contextual Reality Check: Such discovery would be unprecedented, lacking expected institutional response
Step 4 - Scale and Impact: Claim has universe-changing implications but presented casually
Step 5 - Evidence and Source Credibility: Vague "scientists" without specific attribution or peer review
Step 6 - Overall Synthesis: Multiple major red flags indicate fabrication
Score: 0.98

Now analyze this article following the same process:

Article: "{n_ews}"

Step 1 - Scientific and Technical Plausibility:

Step 2 - Logical Consistency Analysis:

Step 3 - Contextual Reality Check:

Step 4 - Scale and Impact Assessment:

Step 5 - Evidence and Source Credibility:

Step 6 - Overall Plausibility Synthesis:

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
    total_samples = 30  
    agent_weights = {
        'style': 0.49,
        'vocab': 0.22,     
        'sentiment': 0.11, 
        'semantics': 0.18 
    }
    
    file_real = pd.read_csv('./Dataset/True.csv', nrows=15)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=15)
    fake_df = pd.DataFrame(file_fake)
    real_df = pd.DataFrame(file_real)
    
    cols = ['text']
    real_sub = real_df[cols]
    fake_sub = fake_df[cols] 

    detector = MultiAgentDetector(agent_weights)
    fake_dataset = NewsDataset(fake_sub)
    fake_dataloader = DataLoader(fake_dataset, batch_size=15)  
    real_dataset = NewsDataset(real_sub)
    real_dataloader = DataLoader(real_dataset, batch_size=15)

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