import pandas as pd
import re
import spacy
from bertopic import BERTopic
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline as hf_pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("Script started. Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Error: spaCy model 'en_core_web_sm' not found.")
    print("Please run: python -m spacy download en_core_web_sm")
    exit()

stopwords = nlp.Defaults.stop_words


data = pd.read_excel("/Users/apple/Documents/Courses-Projects/Hackathon/NewsScope/Copy_of_NewsScope.xlsx")
df = pd.DataFrame(data)

print(f"Loaded sample data with {len(df)} articles.")

# --- PHASE 1: PREPARE & CLEAN DATA ---
print("\n--- PHASE 1: Cleaning Text Data ---")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    doc = nlp(text)
    cleaned_tokens = []
    for token in doc:
        if token.text not in stopwords and not token.is_punct and not token.is_space and len(token.lemma_) > 2:
            cleaned_tokens.append(token.lemma_)
    return " ".join(cleaned_tokens)


df['full_text'] = df['HEADLINE'].fillna('') + " " + df['NEWS_CONTENT'].fillna('')
df['cleaned_text'] = df['full_text'].apply(clean_text)

print("✓ Text cleaning complete.")


# --- PHASE 2: DISCOVER TOPICS (BERTopic) ---
print("\n--- PHASE 2: Discovering Topics with BERTopic ---")
print("(This may take a few minutes...)")

docs = df['cleaned_text'].tolist()

# We set min_topic_size low for the small sample data.
# For a large dataset, use min_topic_size=15 or higher.
topic_model = BERTopic(min_topic_size=2, verbose=False)
topics, probabilities = topic_model.fit_transform(docs)

df['discovered_topic_id'] = topics

print("✓ BERTopic modeling complete.")

# --- Automatically create the topic map (Solves NameError) ---
topic_info = topic_model.get_topic_info()
# We use the default BERTopic name (e.g., "0_stock_market_investor")
topic_id_to_name_map = topic_info.set_index('Topic')['Name'].to_dict()

# Convert integer keys from string (BERTopic map might have int keys)
topic_id_to_name_map = {int(k): v for k, v in topic_id_to_name_map.items()}

print("\nDiscovered Topic Categories:")
print(topic_info)
print(f"\nSuccessfully created topic map with {len(topic_id_to_name_map)} topics.")


# --- PHASE 3: TRAIN TOPIC CLASSIFIER ---
print("\n--- PHASE 3: Training Supervised Topic Classifier ---")

# Filter out outliers (Topic -1)
labeled_df = df[df['discovered_topic_id'] != -1].copy()

if len(labeled_df) > 1:
    X = labeled_df['cleaned_text']
    y = labeled_df['discovered_topic_id']

    # Handle cases with very few samples (like our demo)
    test_size = 0.2 if len(labeled_df) >= 10 else 1
    if len(labeled_df) < 2:
        print("Not enough data to train classifier. Skipping Phase 3.")
        topic_classifier = None
    else:
        # Stratify helps with imbalanced topic sizes
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        except ValueError:
            # Cannot stratify with only 1 sample per class
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Create the model pipeline
        topic_classifier = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42))
        ])

        # Train the classifier
        topic_classifier.fit(X_train, y_train)

        print("✓ Classifier trained.")

        # Evaluate
        if len(X_test) > 0:
            print("\nClassifier Evaluation:")
            predictions = topic_classifier.predict(X_test)
            print(classification_report(y_test, predictions, target_names=[topic_id_to_name_map[t] for t in sorted(y.unique())]))
else:
    print("Not enough data to train classifier (after outlier removal).")
    topic_classifier = None


# --- PHASE 4: LOAD SENTIMENT SYSTEM ---
print("\n--- PHASE 4: Loading Sentiment Analysis Model ---")
print("(This may download the model on first run...)")

sentiment_pipeline = hf_pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
print("✓ Sentiment model loaded.")


# --- (PREP FOR PHASE 5 & 6) ---
# We run sentiment on all articles *now* to help with Rec 2
print("\nAnalyzing all articles for sentiment (for recommendation engine)...")

def get_sentiment(text):
    if not text or pd.isna(text):
        return {'label': 'Neutral', 'score': 0.0}
    # Truncate for efficiency (models have token limits)
    return sentiment_pipeline(text[:512])[0]

df['sentiment_result'] = df['full_text'].apply(get_sentiment)
df['sentiment_label'] = df['sentiment_result'].apply(lambda x: x['label'].capitalize())
df['sentiment_score'] = df['sentiment_result'].apply(lambda x: x['score'])
df['topic_name'] = df['discovered_topic_id'].map(topic_id_to_name_map)
df['topic_name'] = df['topic_name'].fillna("Outlier")
print("✓ Full dataset analyzed.")


# --- PHASE 5: APPLY TO UNSEEN ARTICLES ---
print("\n--- PHASE 5: Analyzing Unseen Articles ---")

def analyze_article(raw_article_text, topic_model, sentiment_model, topic_map):
    """Analyzes raw article text for topic and sentiment."""
    if not topic_model:
        return {"error": "Topic classifier not trained."}
        
    try:
        # 1. Clean the text
        cleaned_article = clean_text(raw_article_text)
        
        # 2. Predict topic
        predicted_topic_id = topic_model.predict([cleaned_article])[0]
        topic_name = topic_map.get(predicted_topic_id, "Unknown Topic") 
        
        # 3. Predict sentiment
        sentiment_result = sentiment_model(raw_article_text[:512])[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']
        
        return {
            "topic": topic_name,
            "sentiment": sentiment_label.capitalize(),
            "sentiment_score": round(sentiment_score, 4)
        }
    except Exception as e:
        return {"error": str(e)}

# Example unseen articles
unseen_articles = [
    "The government just announced new tax cuts aimed at helping small businesses.",
    "The Hawks lost a heartbreaking game in overtime, ending their playoff hopes.",
    "A new study reveals shocking details about climate change impacts.",
    "Tech company 'Innovate Inc.' saw its shares plummet after a poor earnings report."
]

for article in unseen_articles:
    analysis = analyze_article(
        article,
        topic_model=topic_classifier,
        sentiment_model=sentiment_pipeline,
        topic_map=topic_id_to_name_map
    )
    print(f"\nARTICLE: \"{article[:50]}...\"")
    print(f"  -> ANALYSIS: {analysis}")


# --- PHASE 6: BONUS RECOMMENDATIONS ---
print("\n--- PHASE 6: Bonus Recommendation Systems ---")

# --- Part 1: Article-to-Article Similarity ---
print("\nInitializing Article-to-Article recommender...")
tfidf_vectorizer_rec = TfidfVectorizer()
all_article_matrix = tfidf_vectorizer_rec.fit_transform(df['cleaned_text'])
print("✓ Recommender 1 ready.")

def recommend_similar_articles(raw_article_text, n=3):
    # 1. Clean and transform the new article
    cleaned_new = clean_text(raw_article_text)
    new_vec = tfidf_vectorizer_rec.transform([cleaned_new])
    
    # 2. Compute similarity
    sim_scores = cosine_similarity(new_vec, all_article_matrix).flatten()
    
    # 3. Get top N most similar article indices
    # [::-1] reverses the list to get highest scores
    # We take top n+1 and skip the first one (index 0) in case it's the article itself
    top_indices = sim_scores.argsort()[-n-1:][::-1]
    
    # Filter out the article if it's too similar (score > 0.99)
    # This prevents recommending the *exact same* article
    similar_articles = []
    for i in top_indices:
        if sim_scores[i] < 0.99:
            similar_articles.append((df.iloc[i]['headline'], sim_scores[i]))
        if len(similar_articles) == n:
            break
            
    return similar_articles

# --- Part 2: Topic-to-Article ---
print("\nInitializing Topic-to-Article recommender...")
print("✓ Recommender 2 ready.")

def recommend_articles_for_topic(topic_name, n=3, sort_by='positive'):
    
    # Find all articles for that topic (excluding outliers)
    topic_articles = df[df['topic_name'] == topic_name]
    
    if topic_articles.empty:
        return [f"No articles found for topic: {topic_name}"]
    
    if sort_by == 'positive':
        sorted_articles = topic_articles[topic_articles['sentiment_label'] == 'Positive']\
                                        .sort_values(by='sentiment_score', ascending=False)
    elif sort_by == 'negative':
        sorted_articles = topic_articles[topic_articles['sentiment_label'] == 'Negative']\
                                        .sort_values(by='sentiment_score', ascending=False)
    else: # Just get recent or random ones
        sorted_articles = topic_articles.sort_index(ascending=False) # Assumes newer articles have higher index
        
    # Fallback if no articles match sentiment
    if sorted_articles.empty:
        sorted_articles = topic_articles
        
    return sorted_articles.head(n)['headline'].tolist()

# --- Demonstrate Recommendations ---

print("\n--- DEMONSTRATING RECOMMENDATIONS ---")

# Demo 1: Find articles similar to an unseen one
rec_test_article = "The stock market fell hard. Investors are worried about the economy."
print(f"\nRec 1: Articles similar to \"{rec_test_article}\"")
similar = recommend_similar_articles(rec_test_article)
for headline, score in similar:
    print(f"  -> {headline} (Score: {score:.2f})")

# Demo 2: Find positive articles for a specific topic
# We get the first valid topic name from our map (excluding -1)
if -1 in topic_id_to_name_map:
    demo_topic_name = topic_id_to_name_map.get(0, "Outlier")
else:
    demo_topic_name = topic_id_to_name_map.get(list(topic_id_to_name_map.keys())[0])

if demo_topic_name != "Outlier":
    print(f"\nRec 2: Top 'Positive' articles for topic \"{demo_topic_name}\"")
    topic_recs = recommend_articles_for_topic(demo_topic_name, n=2, sort_by='positive')
    for headline in topic_recs:
        print(f"  -> {headline}")

print("\n\n--- SCRIPT COMPLETE ---")