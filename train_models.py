import pandas as pd
import warnings
import os
import json
import joblib
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from transformers import pipeline as hf_pipeline

# Import our custom modules
import config
from text_processing import clean_text

# Suppress warnings
warnings.filterwarnings('ignore')

def train_and_save_models():
    """
    Loads raw data, processes it, trains all models (topic, sentiment, rec),
    and saves them to disk.
    """
    
    # --- 0. Setup ---
    print("Starting model training pipeline...")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # --- 1. Load & Clean Data (Phase 1) ---
    print(f"Loading raw data from {config.RAW_DATA_PATH}...")
    try:
        df = pd.read_excel(config.RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Raw data file not found at {config.RAW_DATA_PATH}")
        return
    
    print(f"Loaded {len(df)} articles.")
    
    print("Cleaning text data (this may take a while)...")
    df['full_text'] = df['HEADLINE'].fillna('') + " " + df['NEWS_CONTENT'].fillna('')
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    print("✓ Text cleaning complete.")

    # --- 2. Discover Topics (Phase 2) ---
    print("\nDiscovering topics with BERTopic...")
    docs = df['cleaned_text'].tolist()
    topic_model = BERTopic(min_topic_size=config.BERTOPIC_MIN_TOPIC_SIZE, verbose=True)
    topics, _ = topic_model.fit_transform(docs)
    df['discovered_topic_id'] = topics
    print("✓ BERTopic modeling complete.")

    # Create and save topic map
    topic_info = topic_model.get_topic_info()
    topic_id_to_name_map = {int(k): v for k, v in topic_info.set_index('Topic')['Name'].to_dict().items()}
    with open(config.TOPIC_MAP_PATH, 'w') as f:
        json.dump(topic_id_to_name_map, f, indent=2)
    print(f"✓ Topic map saved to {config.TOPIC_MAP_PATH}")
    print("\nDiscovered Topic Categories:")
    print(topic_info.head())

    # --- 3. Train Classifier (Phase 3) ---
    print("\nTraining supervised topic classifier...")
    labeled_df = df[df['discovered_topic_id'] != -1].copy()
    
    if len(labeled_df) < 10:
        print("Not enough labeled data to train classifier. Exiting.")
        return

    X = labeled_df['cleaned_text']
    y = labeled_df['discovered_topic_id']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TRAIN_TEST_SPLIT_SIZE, random_state=42, stratify=y)

    topic_classifier = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42))
    ])
    
    topic_classifier.fit(X_train, y_train)
    print("✓ Classifier trained.")

    # Evaluate
    print("\nClassifier Evaluation:")
    predictions = topic_classifier.predict(X_test)
    # Ensure all test labels are in the map for the report
    valid_labels = [label for label in sorted(y.unique()) if label in topic_id_to_name_map]
    target_names = [topic_id_to_name_map[label] for label in valid_labels]
    # Filter predictions/test data to only include valid labels
    valid_indices = [i for i, label in enumerate(y_test) if label in valid_labels]
    y_test_filtered = y_test.iloc[valid_indices]
    predictions_filtered = predictions[valid_indices]
    
    print(classification_report(y_test_filtered, predictions_filtered, target_names=target_names))

    # Save the trained classifier
    joblib.dump(topic_classifier, config.TOPIC_MODEL_PATH)
    print(f"✓ Topic classifier saved to {config.TOPIC_MODEL_PATH}")

    # --- 4. Process for Recommendations (Phase 4 & 6) ---
    
    # Rec 1: Article-to-Article
    print("\nTraining recommender system (TF-IDF)...")
    tfidf_vectorizer_rec = TfidfVectorizer()
    all_article_matrix = tfidf_vectorizer_rec.fit_transform(df['cleaned_text'])
    
    joblib.dump(tfidf_vectorizer_rec, config.TFIDF_VECTORIZER_PATH)
    joblib.dump(all_article_matrix, config.TFIDF_MATRIX_PATH)
    print(f"✓ TF-IDF vectorizer and matrix saved for recommendations.")

    # Rec 2: Topic-to-Article (Needs sentiment)
    print("\nAnalyzing all articles for sentiment (for Rec 2)...")
    sentiment_pipeline = hf_pipeline("sentiment-analysis", model=config.SENTIMENT_MODEL)
    
    def get_sentiment(text):
        if not text or pd.isna(text):
            return {'label': 'Neutral', 'score': 0.0}
        return sentiment_pipeline(text[:512])[0]

    df['sentiment_result'] = df['full_text'].apply(get_sentiment)
    df['sentiment_label'] = df['sentiment_result'].apply(lambda x: x['label'].capitalize())
    df['sentiment_score'] = df['sentiment_result'].apply(lambda x: x['score'])
    df['topic_name'] = df['discovered_topic_id'].map(topic_id_to_name_map).fillna("Outlier")
    print("✓ Full dataset analyzed for sentiment.")

    # Save the final, fully processed DataFrame
    final_df = df[['HEADLINE', 'full_text', 'cleaned_text', 'topic_name', 'sentiment_label', 'sentiment_score']]
    final_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"✓ Fully processed data saved to {config.PROCESSED_DATA_PATH}")

    print("\n--- MODEL TRAINING PIPELINE COMPLETE ---")

if __name__ == "__main__":
    train_and_save_models()