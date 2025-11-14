from analysis_system import NewsAnalysisSystem

def run_demos():
    # 1. Initialize the system (loads all models)
    analyzer = NewsAnalysisSystem()
    
    # Check if system loaded correctly
    if not analyzer.topic_classifier:
        print("Exiting due to model loading failure.")
        return

    # --- PHASE 5: APPLY TO UNSEEN ARTICLES ---
    print("\n--- PHASE 5: Analyzing Unseen Articles ---")
    
    unseen_articles = [
        "The government just announced new tax cuts aimed at helping small businesses.",
        "The Hawks lost a heartbreaking game in overtime, ending their playoff hopes.",
        "A new study reveals shocking details about climate change impacts.",
        "Tech company 'Innovate Inc.' saw its shares plummet after a poor earnings report."
    ]

    for article in unseen_articles:
        analysis = analyzer.analyze_article(article)
        print(f"\nARTICLE: \"{article[:50]}...\"")
        print(f"  -> ANALYSIS: {analysis}")

    # --- PHASE 6: BONUS RECOMMENDATIONS ---
    print("\n--- PHASE 6: Bonus Recommendation Systems ---")

    # Demo 1: Find articles similar to an unseen one
    rec_test_article = "The stock market fell hard. Investors are worried about the economy."
    print(f"\nRec 1: Articles similar to \"{rec_test_article}\"")
    similar = analyzer.recommend_similar_articles(rec_test_article)
    for headline, score in similar:
        print(f"  -> {headline} (Score: {score:.2f})")

    # Demo 2: Find positive articles for a specific topic
    # Get the first valid topic name from the map (e.g., topic '0')
    if 0 in analyzer.topic_map:
        demo_topic_name = analyzer.topic_map[0]
        print(f"\nRec 2: Top 'Positive' articles for topic \"{demo_topic_name}\"")
        topic_recs = analyzer.recommend_articles_for_topic(demo_topic_name, n=2, sort_by='positive')
        for headline in topic_recs:
            print(f"  -> {headline}")
    else:
        print("\nRec 2: Could not find Topic '0' to run demo.")

    print("\n\n--- SCRIPT COMPLETE ---")

if __name__ == "__main__":
    run_demos()