# File paths
RAW_DATA_PATH = "/Users/apple/Documents/Github/NewsScope/Copy_of_NewsScope.xlsx"
PROCESSED_DATA_PATH = "data/processed_news_with_topics.csv"

# Saved model paths
MODELS_DIR = "models"
TOPIC_MODEL_PATH = f"{MODELS_DIR}/topic_classifier.pkl"
TOPIC_MAP_PATH = f"{MODELS_DIR}/topic_map.json"
TFIDF_VECTORIZER_PATH = f"{MODELS_DIR}/tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = f"{MODELS_DIR}/tfidf_matrix.pkl"

# Model parameters
SPACY_MODEL = "en_core_web_sm"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Training parameters
# You have 45k articles, so a larger min_topic_size is better.
BERTOPIC_MIN_TOPIC_SIZE = 150
TRAIN_TEST_SPLIT_SIZE = 0.2