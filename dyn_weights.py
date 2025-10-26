import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import re
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from langchain.prompts import PromptTemplate
import numpy as np
import gc


class NewsDataset(Dataset):
    def __init__(self, dataframe):
        self.data_frame = dataframe
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        text = row['text']
        label = row['label']

        return [text, label]

class BaseAgentClassifier:
    def __init__(self, agent_name, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent_name = agent_name
        self.model_name = model_name
        print(f"{agent_name} initializing...")
        
        # Model and tokenizer will be loaded on-demand
        self.model = None
        self.tokenizer = None

    def _load_model(self):
        """Load model only when needed"""
        if self.model is not None:
            return  # Already loaded
            
        print(f"Loading {self.agent_name} on {self.device}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True
        )

        model_kwargs = {
            'low_cpu_mem_usage': True,
            'quantization_config': bnb_config,
            'device_map': 'auto',
            'max_memory': {0: "6GB", "cpu": "30GB"}
        }

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side='left')

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
    def _unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            print(f"Unloading {self.agent_name}...")
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    def create_prompts(self, news_list):
        raise NotImplementedError("Each agent must implement create_prompts method")
    
    def _extract_score(self, response_text):
        patterns = [
            r'SCORE:\s*([0-1]\.?\d*)',
            r'Score:\s*([0-1]\.?\d*)',
            r'score:\s*([0-1]\.?\d*)',
            r'"score":\s*([0-1]\.?\d*)',
            r'\b([0]\.\d+|1\.0+|0|1)\b(?!.*\b[0-1]\.\d+\b)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    score = float(matches[-1])
                    if 0 <= score <= 1:
                        return score
                except ValueError:
                    continue
        
        return None
    
    def batch_classify(self, news_batch, max_length=20, temperature=0.3):
        # Load model before classification
        self._load_model()
        
        prompts = self._create_prompts(news_batch)

        inputs = self.tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        results = []

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        for i, (output, prompt) in enumerate(zip(outputs, prompts)):
            print(f"  [{self.agent_name}] Classifying article {i+1}/{len(news_batch)}...")
            full_response = self.tokenizer.decode(output, skip_special_tokens=True)
            response_text = full_response[len(prompt):].strip()
            score = self._extract_score(response_text)
            if score is None:
                print(f"  Warning: Could not extract score from response. Defaulting to 0.5")
                score = 0.5
            result = {
                'score': score,
                'analysis': response_text,
                'confidence': 'high' if temperature < 0.5 else 'medium'
            }
            results.append(result)
        
        # Unload model after classification
        self._unload_model()
        
        return results
    

class NewsStyleClassifier(BaseAgentClassifier):
    def __init__(self):
        super().__init__("Style Agent")
    
    def _create_prompts(self, news_list):
        prompts = []
        template_str = """Rate this news article's professionalism from 0 to 1:
0 = professional/reputed publisher (good grammar, objective, well-sourced)
1 = unprofessional/untrusted (poor grammar, sensational, clickbait, biased)

Article: {news}

Output only JSON: {{"score": <number>, "reasoning": "<explanation>"}}"""
                
        prompt_template = PromptTemplate.from_template(template_str)
        
        for news in news_list:
            final_prompt = prompt_template.format(news=news[:1000])
            prompts.append(final_prompt)
        
        return prompts

class NewsSentimentClassifier(BaseAgentClassifier):
    def __init__(self):
        super().__init__("Sentiment Agent")
    
    def _create_prompts(self, news_list):
        prompts = []
        template = """Rate this news article's sentiment from 0 to 1:
0 = objective/neutral (fact-based, impartial, unemotional tone)
1 = highly emotional/opinionated (uses strong emotional language, subjective, persuasive)

Article: {news}

Output only JSON: {{"score": <number>, "reasoning": "<explanation>"}}"""

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(news=news[:1000])
            prompts.append(final_prompt)

        return prompts

class NewsVocabClassifier(BaseAgentClassifier):
    def __init__(self):
        super().__init__("Vocab Agent")
    
    def _create_prompts(self, news_list):
        prompts = []
        template = """Rate this news article's vocabulary complexity from 0 to 1:
0 = simple/accessible (common words, easy to read for a broad audience)
1 = complex/jargon-heavy (uses technical terms, academic language, complex sentences)

Article: {news}

Output only JSON: {{"score": <number>, "reasoning": "<explanation>"}}"""

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(news=news[:1000])
            prompts.append(final_prompt)

        return prompts

class NewsSemanticClassifier(BaseAgentClassifier):
    def __init__(self):
        super().__init__("Semantic Agent")
    
    def _create_prompts(self, news_list):
        prompts = []
        template = """Rate the semantic clarity of this news article from 0 to 1:
0 = direct/literal (clear meaning, straightforward, unambiguous)
1 = abstract/figurative (uses metaphor, analogy, requires interpretation, nuanced)

Article: {news}

Output only JSON: {{"score": <number>, "reasoning": "<explanation>"}}"""

        for news in news_list:
            prompt = PromptTemplate.from_template(template)
            final_prompt = prompt.format(news=news[:1000])
            prompts.append(final_prompt)

        return prompts


class WeightModule(nn.Module):
    """Learnable weight module with softmax normalization"""
    def __init__(self, num_agents=4):
        super(WeightModule, self).__init__()
        # Initialize weights uniformly
        self.weights = nn.Parameter(torch.ones(num_agents) / num_agents)
    
    def forward(self):
        # Use softmax to ensure weights sum to 1 and are positive
        return torch.softmax(self.weights, dim=0)


class AgregatedNewsClassifier:

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Initialize agents (models not loaded yet)
        self.style_agent = NewsStyleClassifier()
        self.sentiment_agent = NewsSentimentClassifier()
        self.vocab_agent = NewsVocabClassifier()
        self.semantic_agent = NewsSemanticClassifier()
        
        # Initialize learnable weights
        self.device = device
        self.weight_module = WeightModule(num_agents=4).to(device)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
    
    def get_agent_scores(self, news_batch):
        """Get scores from all agents"""
        print(f"\n{'='*60}")
        print(f"Processing batch of {len(news_batch)} articles")
        print(f"{'='*60}\n")
        
        # Process each agent sequentially
        print("Step 1/4: Running Style Agent...")
        style_results = self.style_agent.batch_classify(news_batch)
        
        print("\nStep 2/4: Running Sentiment Agent...")
        sentiment_results = self.sentiment_agent.batch_classify(news_batch)
        
        print("\nStep 3/4: Running Vocabulary Agent...")
        vocab_results = self.vocab_agent.batch_classify(news_batch)
        
        print("\nStep 4/4: Running Semantic Agent...")
        semantic_results = self.semantic_agent.batch_classify(news_batch)

        # Turn scores to arrays
        pred_style_scores = np.array([result['score'] for result in style_results])
        pred_sentiment_scores = np.array([result['score'] for result in sentiment_results])
        pred_vocab_scores = np.array([result['score'] for result in vocab_results])
        pred_semantic_scores = np.array([result['score'] for result in semantic_results])

        # Stack all scores into a matrix [batch_size, num_agents]
        all_scores = np.stack([pred_style_scores, pred_sentiment_scores, 
                               pred_vocab_scores, pred_semantic_scores], axis=1)
        
        return all_scores, {
            'style': style_results,
            'sentiment': sentiment_results,
            'vocabulary': vocab_results,
            'semantics': semantic_results
        }
    
    def classify(self, news_batch):
        """Classify using current weights"""
        all_scores, detailed_analysis = self.get_agent_scores(news_batch)
        
        # Convert to tensor
        scores_tensor = torch.FloatTensor(all_scores).to(self.device)
        
        # Get current weights
        weights = self.weight_module()
        
        # Calculate weighted aggregation
        aggregated_scores = torch.matmul(scores_tensor, weights)
        aggregated_scores = aggregated_scores.cpu().numpy()

        # Return list of results for each article
        final_results = []
        for i in range(len(news_batch)):
            final_results.append({
                'aggregated_score': aggregated_scores[i],
                'detailed_analysis': {
                    'style': detailed_analysis['style'][i],
                    'sentiment': detailed_analysis['sentiment'][i],
                    'vocabulary': detailed_analysis['vocabulary'][i],
                    'semantics': detailed_analysis['semantics'][i]
                }
            })

        print(f"\n{'='*60}")
        print("Batch processing complete!")
        print(f"{'='*60}\n")
        
        return final_results
    
    def train_weights(self, dataloader, epochs=10, lr=0.01, save_path='best_weights.pth'):
        """Train the aggregation weights"""
        optimizer = torch.optim.Adam(self.weight_module.parameters(), lr=lr)
        
        best_loss = float('inf')
        best_weights = None
        
        print(f"\n{'='*60}")
        print(f"STARTING WEIGHT TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {epochs}, Learning Rate: {lr}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            print(f"\n{'#'*60}")
            print(f"EPOCH {epoch + 1}/{epochs}")
            print(f"{'#'*60}\n")
            
            for batch_idx, (news, labels) in enumerate(dataloader):
                print(f"\nBatch {batch_idx + 1}/{len(dataloader)}")
                
                # Get agent scores (this will run all 4 agents sequentially)
                all_scores, _ = self.get_agent_scores(news)
                
                # Convert to tensors
                scores_tensor = torch.FloatTensor(all_scores).to(self.device)
                labels_tensor = torch.FloatTensor(labels).to(self.device)
                
                # Forward pass
                weights = self.weight_module()
                aggregated_logits = torch.matmul(scores_tensor, weights)
                
                # Calculate loss
                loss = self.criterion(aggregated_logits, labels_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Print batch results
                print(f"\n  Batch Loss: {loss.item():.4f}")
                print(f"  Current Weights: {weights.detach().cpu().numpy()}")
            
            avg_loss = epoch_loss / num_batches
            
            # Save best weights
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = self.weight_module.state_dict()
                torch.save(best_weights, save_path)
                print(f"\n  ✓ New best weights saved! Loss: {avg_loss:.4f}")
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch + 1} SUMMARY")
            print(f"{'='*60}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Best Loss: {best_loss:.4f}")
            current_weights = self.weight_module().detach().cpu().numpy()
            print(f"Current Weights:")
            print(f"  Style:     {current_weights[0]:.4f}")
            print(f"  Sentiment: {current_weights[1]:.4f}")
            print(f"  Vocab:     {current_weights[2]:.4f}")
            print(f"  Semantic:  {current_weights[3]:.4f}")
            print(f"{'='*60}\n")
        
        # Load best weights
        self.weight_module.load_state_dict(best_weights)
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best Loss: {best_loss:.4f}")
        final_weights = self.weight_module().detach().cpu().numpy()
        print(f"Final Optimized Weights:")
        print(f"  Style:     {final_weights[0]:.4f}")
        print(f"  Sentiment: {final_weights[1]:.4f}")
        print(f"  Vocab:     {final_weights[2]:.4f}")
        print(f"  Semantic:  {final_weights[3]:.4f}")
        print(f"{'='*60}\n")
        
        return final_weights
    
    def load_weights(self, path='best_weights.pth'):
        """Load pre-trained weights"""
        self.weight_module.load_state_dict(torch.load(path))
        weights = self.weight_module().detach().cpu().numpy()
        print(f"Loaded weights: {weights}")
        return weights
    
    def evaluate(self, dataloader):
        """Evaluate the classifier"""
        total_samples = 0
        fake_correct = 0
        real_correct = 0
        
        print(f"\n{'='*60}")
        print(f"EVALUATION")
        print(f"{'='*60}\n")
        
        for batch_idx, (news, labels) in enumerate(dataloader):
            print(f"\nEvaluating Batch {batch_idx + 1}/{len(dataloader)}")
            
            results = self.classify(news)
            
            real_scores = np.array(labels)
            pred_scores = np.array([result['aggregated_score'] for result in results])
            
            # Apply sigmoid to convert logits to probabilities
            pred_probs = 1 / (1 + np.exp(-pred_scores))
            
            print(f"\nGround Truths:    {real_scores.tolist()}")
            print(f"Predicted Probs:  {[f'{p:.3f}' for p in pred_probs]}")
            
            for i in range(len(real_scores)):
                total_samples += 1
                if (real_scores[i] == 1 and pred_probs[i] >= 0.5) or (real_scores[i] == 0 and pred_probs[i] < 0.5):
                    if real_scores[i] == 1:
                        fake_correct += 1
                    else:
                        real_correct += 1

        total_correct = fake_correct + real_correct
        accuracy = (total_correct / total_samples) * 100
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Total samples:     {total_samples}")
        print(f"Fake news correct: {fake_correct}")
        print(f"Real news correct: {real_correct}")
        print(f"Total correct:     {total_correct}")
        print(f"Accuracy:          {accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        return accuracy


if __name__ == "__main__":
    print("Initializing Aggregated News Classifier...")
    classifier = AgregatedNewsClassifier()

    print("\nLoading datasets...")
    file_real = pd.read_csv('./Dataset/True.csv', nrows=30)
    file_fake = pd.read_csv('./Dataset/Fake.csv', nrows=30)
    fake_df = pd.DataFrame(file_fake)
    real_df = pd.DataFrame(file_real)
    fake_df['label'] = 1
    real_df['label'] = 0

    full_df = pd.concat([fake_df, real_df])
    print(f"Total articles: {len(full_df)}")
    
    cols = ['text', 'label'] 
    full_sub = full_df[cols]
    full_dataset = NewsDataset(full_sub)
    
    # Split into train and test
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    print(f"Training samples: {train_size}")
    print(f"Testing samples: {test_size}")
    
    # Train weights
    # final_weights = classifier.train_weights(train_dataloader, epochs=5, lr=0.01)
    
    # Evaluate on test set
    test_accuracy = classifier.evaluate(test_dataloader)