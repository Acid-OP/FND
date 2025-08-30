import pandas as pd
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import string

class VocabBasedFakeNewsDetector:
    def __init__(self):
        # Heuristic patterns for fake news
        self.fake_indicators = {
            'sensational_words': [
                'shocking', 'unbelievable', 'amazing', 'incredible', 'devastating',
                'explosive', 'bombshell', 'exclusive', 'breaking', 'urgent',
                'secret', 'hidden', 'revealed', 'exposed', 'scandal',
                'you won\'t believe', 'must see', 'viral', 'going viral'
            ],
            'emotional_words': [
                'outraged', 'furious', 'disgusting', 'terrifying', 'horrifying',
                'infuriating', 'devastating', 'heartbreaking', 'alarming',
                'disturbing', 'shocking', 'appalling'
            ],
            'clickbait_phrases': [
                'you won\'t believe', 'what happens next', 'will shock you',
                'doctors hate', 'this one trick', 'number will surprise',
                'before it\'s too late', 'they don\'t want you to know'
            ],
            'weak_sources': [
                'sources say', 'according to reports', 'it is believed',
                'allegedly', 'rumored', 'some say', 'it is said',
                'unconfirmed', 'anonymous source'
            ],
            'absolute_language': [
                'always', 'never', 'everyone', 'nobody', 'all', 'none',
                'every', 'completely', 'totally', 'absolutely', 'definitely'
            ]
        }
        
        self.real_indicators = {
            'credible_sources': [
                'according to', 'study shows', 'research indicates',
                'data suggests', 'experts say', 'officials confirm',
                'spokesperson said', 'statement released'
            ],
            'factual_language': [
                'approximately', 'estimated', 'reported', 'confirmed',
                'verified', 'documented', 'recorded', 'observed'
            ],
            'neutral_tone': [
                'however', 'meanwhile', 'additionally', 'furthermore',
                'according to', 'in contrast', 'similarly'
            ]
        }
        
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.model = LogisticRegression()
        
    def extract_heuristic_features(self, text):
        """Extract vocabulary-based heuristic features"""
        text_lower = text.lower()
        features = {}
        
        # Count fake indicators
        for category, words in self.fake_indicators.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'fake_{category}_count'] = count
            features[f'fake_{category}_density'] = count / len(text.split()) if text.split() else 0
        
        # Count real indicators
        for category, words in self.real_indicators.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'real_{category}_count'] = count
            features[f'real_{category}_density'] = count / len(text.split()) if text.split() else 0
        
        # Additional heuristics
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['total_words'] = len(text.split())
        
        # Punctuation density
        punct_count = sum(1 for c in text if c in string.punctuation)
        features['punct_density'] = punct_count / len(text) if text else 0
        
        return features
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', str(text)).strip()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        return text
    
    def train(self, texts, labels):
        """Train the vocabulary-based classifier"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract heuristic features
        heuristic_features = []
        for text in processed_texts:
            features = self.extract_heuristic_features(text)
            heuristic_features.append(list(features.values()))
        
        # TF-IDF features
        tfidf_features = self.tfidf.fit_transform(processed_texts)
        
        # Combine features
        heuristic_array = np.array(heuristic_features)
        combined_features = np.hstack([heuristic_array, tfidf_features.toarray()])
        
        # Train model
        self.model.fit(combined_features, labels)
        
        return self
    
    def predict(self, texts):
        """Predict using vocabulary-based features"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Extract heuristic features
        heuristic_features = []
        for text in processed_texts:
            features = self.extract_heuristic_features(text)
            heuristic_features.append(list(features.values()))
        
        # TF-IDF features
        tfidf_features = self.tfidf.transform(processed_texts)
        
        # Combine features
        heuristic_array = np.array(heuristic_features)
        combined_features = np.hstack([heuristic_array, tfidf_features.toarray()])
        
        return self.model.predict(combined_features)
    
    def predict_proba(self, texts):
        """Get prediction probabilities"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        heuristic_features = []
        for text in processed_texts:
            features = self.extract_heuristic_features(text)
            heuristic_features.append(list(features.values()))
        
        tfidf_features = self.tfidf.transform(processed_texts)
        heuristic_array = np.array(heuristic_features)
        combined_features = np.hstack([heuristic_array, tfidf_features.toarray()])
        
        return self.model.predict_proba(combined_features)

# --- Enhanced Evaluation Script ---
def enhanced_vocab_evaluation():
    # Load datasets
    df_fake = pd.read_csv(r"./Dataset/gossipcop_fake.csv")
    df_real = pd.read_csv(r"./Dataset/gossipcop_real.csv")
    
    # Prepare data
    fake_titles = df_fake['title'].dropna().head(100).tolist()  # More samples for training
    real_titles = df_real['title'].dropna().head(100).tolist()
    
    # Create training and test sets
    train_fake = fake_titles[:70]  # 70 for training
    train_real = real_titles[:70]
    test_fake = fake_titles[70:100]  # 30 for testing
    test_real = real_titles[70:100]
    
    # Prepare training data
    train_texts = train_fake + train_real
    train_labels = [0] * len(train_fake) + [1] * len(train_real)  # 0=FAKE, 1=REAL
    
    # Prepare test data
    test_texts = test_fake + test_real
    test_labels = [0] * len(test_fake) + [1] * len(test_real)
    
    # Train the enhanced detector
    detector = VocabBasedFakeNewsDetector()
    detector.train(train_texts, train_labels)
    
    # Make predictions
    predictions = detector.predict(test_texts)
    probabilities = detector.predict_proba(test_texts)
    
    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    
    print("===== ENHANCED VOCABULARY-BASED FAKE NEWS DETECTION =====")
    print(f"Training Samples: {len(train_texts)} (70 fake + 70 real)")
    print(f"Test Samples: {len(test_texts)} (30 fake + 30 real)")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, predictions, target_names=['FAKE', 'REAL']))
    
    # Show some examples
    print("\n===== SAMPLE PREDICTIONS =====")
    for i in range(min(10, len(test_texts))):
        true_label = "FAKE" if test_labels[i] == 0 else "REAL"
        pred_label = "FAKE" if predictions[i] == 0 else "REAL"
        confidence = max(probabilities[i])
        
        print(f"\nHeadline: {test_texts[i][:80]}...")
        print(f"True: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.3f}")
        print(f"Correct: {true_label == pred_label}")
    
    return detector, accuracy

# Run the enhanced evaluation
if __name__ == "__main__":
    detector, final_accuracy = enhanced_vocab_evaluation()
    
    print(f"\n🎯 FINAL ACCURACY: {final_accuracy:.2%}")
    
    if final_accuracy > 0.65:
        print("✅ Good performance for vocabulary-based heuristics!")
    elif final_accuracy > 0.55:
        print("⚠️  Moderate performance - room for improvement")
    else:
        print("❌ Poor performance - needs significant improvement")