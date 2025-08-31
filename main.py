import pandas as pd
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import pipeline
from random import sample

# --- Vocabulary-based detector ---
class VocabBasedFakeNewsDetector:
    def __init__(self):
        self.fake_indicators = {
            'sensational_words': ['shocking','unbelievable','amazing','incredible','devastating','explosive','bombshell','exclusive','breaking','urgent','secret','hidden','revealed','exposed','scandal','you won\'t believe','must see','viral','going viral'],
            'emotional_words': ['outraged','furious','disgusting','terrifying','horrifying','infuriating','devastating','heartbreaking','alarming','disturbing','shocking','appalling'],
            'clickbait_phrases': ['you won\'t believe','what happens next','will shock you','doctors hate','this one trick','number will surprise','before it\'s too late','they don\'t want you to know'],
            'weak_sources': ['sources say','according to reports','it is believed','allegedly','rumored','some say','it is said','unconfirmed','anonymous source'],
            'absolute_language': ['always','never','everyone','nobody','all','none','every','completely','totally','absolutely','definitely']
        }
        self.real_indicators = {
            'credible_sources': ['according to','study shows','research indicates','data suggests','experts say','officials confirm','spokesperson said','statement released'],
            'factual_language': ['approximately','estimated','reported','confirmed','verified','documented','recorded','observed'],
            'neutral_tone': ['however','meanwhile','additionally','furthermore','according to','in contrast','similarly']
        }
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 3))
        self.model = LogisticRegression(max_iter=1000)

    def preprocess_text(self, text):
        text = re.sub(r'\s+', ' ', str(text)).strip()
        text = re.sub(r'http\S+|www\S+', '', text)
        return text

    def extract_heuristic_features(self, text):
        text_lower = text.lower()
        features = {}
        for category, words in self.fake_indicators.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'fake_{category}_count'] = count
            features[f'fake_{category}_density'] = count / len(text.split()) if text.split() else 0
        for category, words in self.real_indicators.items():
            count = sum(1 for word in words if word in text_lower)
            features[f'real_{category}_count'] = count
            features[f'real_{category}_density'] = count / len(text.split()) if text.split() else 0
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['total_words'] = len(text.split())
        punct_count = sum(1 for c in text if c in string.punctuation)
        features['punct_density'] = punct_count / len(text) if text else 0
        return features

    def train(self, texts, labels):
        processed_texts = [self.preprocess_text(text) for text in texts]
        heuristic_features = [list(self.extract_heuristic_features(text).values()) for text in processed_texts]
        tfidf_features = self.tfidf.fit_transform(processed_texts)
        combined_features = np.hstack([np.array(heuristic_features), tfidf_features.toarray()])
        self.model.fit(combined_features, labels)
        return self

    def predict_score(self, texts):
        processed_texts = [self.preprocess_text(text) for text in texts]
        heuristic_features = [list(self.extract_heuristic_features(text).values()) for text in processed_texts]
        tfidf_features = self.tfidf.transform(processed_texts)
        combined_features = np.hstack([np.array(heuristic_features), tfidf_features.toarray()])
        proba = self.model.predict_proba(combined_features)
        score = proba[:, 0] - proba[:, 1]
        return score

# --- Hybrid Detector with WORKING clickbait model ---
class HybridFakeNewsDetector:
    def __init__(self):
        self.vocab_detector = VocabBasedFakeNewsDetector()
        self.tfidf_model = None
        self.tfidf_vectorizer = None
        self.label_encoder = LabelEncoder()
        
        # Use lightweight sentiment model
        print("Loading sentiment model...")
        self.sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        
        # --- WORKING PUBLIC CLICKBAIT MODEL ---
        print("Loading clickbait model...")
        try:
            self.clickbait_model = pipeline(
                "text-classification",
                model="caush/Clickbait1",  # This model exists and works
                return_all_scores=True
            )
            print("✓ Clickbait model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Clickbait model failed to load: {e}")
            print("Continuing without clickbait component...")
            self.clickbait_model = None

    @staticmethod
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def train(self, fake_texts, real_texts):
        print("Training hybrid detector...")
        all_texts = fake_texts + real_texts
        labels = ['FAKE']*len(fake_texts) + ['REAL']*len(real_texts)
        y = self.label_encoder.fit_transform(labels)
        
        # Train TF-IDF component
        self.tfidf_vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,3))
        X = self.tfidf_vectorizer.fit_transform([self.preprocess(t) for t in all_texts])
        self.tfidf_model = LogisticRegression(max_iter=1000)
        self.tfidf_model.fit(X, y)
        
        # Train vocabulary component
        self.vocab_detector.train(all_texts, y)
        print("✓ Training completed!")

    def predict(self, text):
        text_clean = self.preprocess(text)

        # --- Sentiment score ---
        try:
            result = self.sentiment_model(text[:512])[0]
            sent_score = result["score"] if result["label"]=="NEGATIVE" else -result["score"]
        except:
            sent_score = 0

        # --- TF-IDF score ---
        try:
            tfidf_score_array = self.tfidf_model.predict_proba(self.tfidf_vectorizer.transform([text_clean]))[0]
            tfidf_pred = 'FAKE' if tfidf_score_array[self.label_encoder.transform(['FAKE'])[0]] > 0.5 else 'REAL'
            tfidf_score = tfidf_score_array[self.label_encoder.transform([tfidf_pred])[0]]
            tfidf_score = tfidf_score if tfidf_pred=='FAKE' else -tfidf_score
        except:
            tfidf_score = 0

        # --- Vocab score ---
        try:
            vocab_score = self.vocab_detector.predict_score([text])[0]
        except:
            vocab_score = 0

        # --- Clickbait score ---
        cb_score = 0
        if self.clickbait_model:
            try:
                clickbait_scores = self.clickbait_model(text[:512])[0]
                for item in clickbait_scores:
                    if 'clickbait' in item['label'].lower() or item['label'] == 'LABEL_1':
                        cb_score = item['score']
                    else:
                        cb_score -= item['score']
            except:
                cb_score = 0

        # --- Weighted combination ---
        if self.clickbait_model:
            final_score = 0.25*sent_score + 0.35*tfidf_score + 0.25*vocab_score + 0.15*cb_score
        else:
            # Rebalance weights if no clickbait model
            final_score = 0.3*sent_score + 0.4*tfidf_score + 0.3*vocab_score
        
        final_label = 'FAKE' if final_score > 0 else 'REAL'
        return final_label, final_score

# --- Main ---
if __name__ == "__main__":
    print("=" * 60)
    print("LOADING GOSSIPCOP DATASET")
    print("=" * 60)
    
    df_fake = pd.read_csv(r"./Dataset/gossipcop_fake.csv",nrows=1024)
    df_real = pd.read_csv(r"./Dataset/gossipcop_real.csv",nrows=1024)

    train_df_fake, test_df_fake = train_test_split(df_fake, test_size=0.2, random_state=42)
    train_df_real, test_df_real = train_test_split(df_real, test_size=0.2, random_state=42)


    # --- 80% as training samples ---
    train_fake_titles = train_df_fake['title'].dropna().tolist()
    train_real_titles = train_df_real['title'].dropna().tolist()

    # --- 20% as testing samples ---
    test_fake_titles = test_df_fake['title'].dropna().tolist()
    test_real_titles = test_df_real['title'].dropna().tolist()


    print(f"Loaded {len(train_fake_titles)} fake titles and {len(train_real_titles)} real titles")
    
    print("\n" + "=" * 60)
    print("INITIALIZING HYBRID DETECTOR")
    print("=" * 60)
    
    hybrid_detector = HybridFakeNewsDetector()
    hybrid_detector.train(train_fake_titles, train_real_titles)

    print("\n" + "=" * 60)
    print("TESTING ON GOSSIPCOP DATASET")
    print("=" * 60)
    
    test_samples = [('FAKE', t) for t in test_fake_titles] + [('REAL', t) for t in test_real_titles]
    wrong = 0
    correct = 0
    
    for i, (true_label, text) in enumerate(test_samples):
        pred_label, score = hybrid_detector.predict(text)
        is_correct = pred_label == true_label
        
        if is_correct:
            correct += 1
        else:
            wrong += 1
        
        # Show first 10 results
        if i < 10:
            status = "✓" if is_correct else "✗"
            print(f"{status} '{text[:50]}...'")
            print(f"   True: {true_label} | Predicted: {pred_label} | Score: {score:.3f}")
            print("-" * 60)

    accuracy = 100*(1 - wrong/len(test_samples))
    print(f"\nFINAL RESULTS ON GOSSIPCOP:")
    print(f"Total Samples: {len(test_samples)}")
    print(f"Correct: {correct}")
    print(f"Wrong: {wrong}")
    print(f"Hybrid Detector Accuracy: {accuracy:.2f}%")
    
    # Function for pipeline
    def fake_news_detector_for_pipeline(text):
        """Use this in your pipeline"""
        return hybrid_detector.predict(text)[0]  # Returns 'FAKE' or 'REAL'