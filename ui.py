import streamlit as st
import re
from analysis_system import NewsAnalysisSystem

# --- Page Configuration ---
# This sets the title in the browser tab, the icon, and a wide layout
st.set_page_config(
    page_title="NewsScope Analyzer",
    page_icon="📰",
    layout="wide"
)

# --- Model Loading ---
# This is the most important part!
# @st.cache_resource tells Streamlit to run this function ONCE
# and keep the result (your big 2GB+ 'analyzer' object) in memory.
@st.cache_resource
def load_analysis_system():
    """
    Loads and caches the heavy NewsAnalysisSystem object.
    """
    print("--- LOADING ANALYSIS SYSTEM (one-time operation) ---")
    system = NewsAnalysisSystem()
    print("--- SYSTEM LOADED ---")
    return system

# Load the system. This will be instant on subsequent runs.
analyzer = load_analysis_system()

# --- Helper Function ---
def format_topic_name(topic_name):
    """
    Cleans up the default BERTopic name for display.
    e.g., "0_exam_student_iit" -> "Exam Student Iit"
    """
    try:
        # Remove the leading number and underscore
        parts = topic_name.split('_', 1)[1]
        # Replace underscores with spaces and capitalize
        return ' '.join([word.capitalize() for word in parts.split('_')])
    except Exception:
        # If it fails (e.g., "Outlier"), return as-is
        return topic_name

# --- Main App UI ---
st.title("📰 NewsScope: Article Analyzer & Recommender")
st.markdown("Enter a news headline or the full article text below to discover its topic, sentiment, and find related content.")

# Check if the models loaded correctly
if not analyzer.topic_classifier:
    st.error(
        "FATAL: Analysis System failed to load. Have you run `train_models.py` first? Check the terminal logs for errors."
    )
    st.stop() # Halt the app if models aren't ready

# --- Input Area ---
# We use a text area for multi-line article input
# We also provide a default example to make it easy to test
default_article = "The government just announced new tax cuts aimed at helping small businesses. This move is expected to boost the economy and provide relief for struggling entrepreneurs, though some critics argue it won't be enough."

article_input = st.text_area(
    "Enter Article Text:",
    value=default_article,
    height=200,
    key="article_input"
)

# The "Analyze" button
analyze_button = st.button("Analyze Article", type="primary", use_container_width=True)

st.divider()

# --- Results Area ---
# This code block only runs if the button is clicked
if analyze_button:
    if not article_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # Show a spinner while processing
        with st.spinner("Analyzing... This may take a moment."):
            
            # 1. Run Core Analysis
            analysis = analyzer.analyze_article(article_input)
            
            # 2. Get Topic Name for Recs
            raw_topic_name = analysis.get('topic', 'Unknown')
            
            # 3. Get Recommendations
            similar_articles = analyzer.recommend_similar_articles(article_input, n=3)
            topic_articles = analyzer.recommend_articles_for_topic(
                raw_topic_name, 
                n=3, 
                sort_by=analysis.get('sentiment', 'Positive').lower()
            )

        # --- Display Results ---
        st.header("Analysis Results")
        
        # Use columns for a "classy" dashboard layout
        col1, col2 = st.columns([1.5, 1])

        with col1:
            st.subheader("Primary Analysis")
            
            # Format the topic name
            formatted_topic = format_topic_name(raw_topic_name)
            
            # Use st.metric for a beautiful "card" display
            st.metric("Predicted Topic", formatted_topic)
            
            sentiment = analysis.get('sentiment', 'N/A')
            score = analysis.get('sentiment_score', 0)
            
            st.metric("Sentiment", sentiment, f"{score * 100:.1f}% Score")

            # Add a visual emoji for sentiment
            if sentiment == 'Positive':
                st.markdown(f"### <span style='color:green;'>{sentiment} 😊</span>", unsafe_allow_html=True)
            elif sentiment == 'Negative':
                st.markdown(f"### <span style='color:red;'>{sentiment} 😞</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"### <span style='color:gray;'>{sentiment} 😐</span>", unsafe_allow_html=True)

        with col2:
            st.subheader("Related Content")
            
            st.markdown("**Similar to this article:**")
            if similar_articles:
                for headline, score in similar_articles:
                    st.info(f"_{headline}_  (Score: {score:.2f})")
            else:
                st.info("No similar articles found.")
            
            st.markdown("**More in this topic:**")
            if topic_articles:
                for headline in topic_articles:
                    st.success(f"_{headline}_")
            else:
                st.info("No other articles found for this topic.")