import joblib
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline

# Import our custom modules
import config
from text_processing import clean_text

class NewsAnalysisSystem:
    def __init__(self):
        """
        Loads all pre-trained models and data needed for analysis.
        """
        print("Loading analysis system...")
        try:
            # Load Topic Model
            self.topic_classifier = joblib.load(config.TOPIC_MODEL_PATH)
            
            # Load Topic Map
            with open(config.TOPIC_MAP_PATH, 'r') as f:
                self.topic_map = {int(k): v for k, v in json.load(f).items()}
                
            # Load Sentiment Model
            self.sentiment_pipeline = hf_pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL)
            
            # Load Recommendation 1 assets
            self.tfidf_vectorizer = joblib.load(config.TFIDF_VECTORIZER_PATH)
            self.tfidf_matrix = joblib.load(config.TFIDF_MATRIX_PATH)
            
            # Load Recommendation 2 assets (the processed data)
            self.processed_df = pd.read_csv(config.PROCESSED_DATA_PATH)
            
            print("✓ System ready.")
            
        except FileNotFoundError as e:
            print(f"ERROR: Model file not found. Have you run 'train_models.py' first?")
            print(f"Missing file: {e.filename}")
            self.topic_classifier = None
        except Exception as e:
            print(f"An error occurred during model loading: {e}")
            self.topic_classifier = None

    def analyze_article(self, raw_article_text):
        """Analyzes a single unseen article for topic and sentiment."""
        if not self.topic_classifier:
            return {"error": "System not initialized. Run training."}
            
        try:
            # 1. Clean the text
            cleaned_article = clean_text(raw_article_text)
            
            # 2. Predict topic
            predicted_topic_id = self.topic_classifier.predict([cleaned_article])[0]
            topic_name = self.topic_map.get(predicted_topic_id, "Unknown Topic") 
            
            # 3. Predict sentiment
            sentiment_result = self.sentiment_pipeline(raw_article_text[:512])[0]
            sentiment_label = sentiment_result['label'].capitalize()
            sentiment_score = round(sentiment_result['score'], 4)
            
            return {
                "topic": topic_name,
                "sentiment": sentiment_label,
                "sentiment_score": sentiment_score
            }
        except Exception as e:
            return {"error": str(e)}

    def recommend_similar_articles(self, raw_article_text, n=3):
        """Recommends N articles similar to the input text."""
        if not self.topic_classifier:
            return [("System not initialized.", 0.0)]
            
        cleaned_new = clean_text(raw_article_text)
        new_vec = self.tfidf_vectorizer.transform([cleaned_new])
        sim_scores = cosine_similarity(new_vec, self.tfidf_matrix).flatten()
        
        top_indices = sim_scores.argsort()[-n-1:][::-1]
        
        similar_articles = []
        for i in top_indices:
            # Don't recommend the exact same article
            if sim_scores[i] < 0.99:
                similar_articles.append((self.processed_df.iloc[i]['HEADLINE'], sim_scores[i]))
            if len(similar_articles) == n:
                break
                
        return similar_articles

    def recommend_articles_for_topic(self, topic_name, n=3, sort_by='positive'):
        """Recommends N articles for a given topic."""
        if not self.topic_classifier:
            return ["System not initialized."]
            
        topic_articles = self.processed_df[self.processed_df['topic_name'] == topic_name]
        
        if topic_articles.empty:
            return [f"No articles found for topic: {topic_name}"]
        
        if sort_by == 'positive':
            sorted_articles = topic_articles.nlargest(n, 'sentiment_score')
        elif sort_by == 'negative':
            negative_articles = topic_articles[topic_articles['sentiment_label'] == 'Negative']
            sorted_articles = negative_articles.nlargest(n, 'sentiment_score')
        else:
            sorted_articles = topic_articles.head(n)
            
        if sorted_articles.empty:
            sorted_articles = topic_articles.head(n)
            
        return sorted_articles['HEADLINE'].tolist()