import nltk
import pandas as pd
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
nltk.download('vader_lexicon', quiet=True)

# Sample tweets dataset
tweets = [
    "I absolutely love this product! Best thing ever!",
    "This is terrible, worst experience of my life",
    "The weather today is okay, nothing special",
    "Just had the most amazing meal at this restaurant!",
    "I hate when people are rude for no reason",
    "Today was a pretty normal day at work",
    "OMG this movie is incredible, totally blown away!",
    "The service was disappointing and slow",
    "Feeling neutral about the whole situation",
    "Best day ever, everything went perfectly!",
    "Absolutely awful, would not recommend to anyone",
    "It was fine, met my basic expectations",
]

labels = ['positive', 'negative', 'neutral', 'positive', 
          'negative', 'neutral', 'positive', 'negative',
          'neutral', 'positive', 'negative', 'neutral']

# Initialize VADER
sia = SentimentIntensityAnalyzer()

print("Twitter Sentiment Analyzer")
print("=" * 50)

results = []
for tweet, true_label in zip(tweets, labels):
    scores = sia.polarity_scores(tweet)
    compound = scores['compound']
    
    if compound >= 0.05:
        predicted = 'positive'
    elif compound <= -0.05:
        predicted = 'negative'
    else:
        predicted = 'neutral'
    
    results.append({
        'tweet': tweet[:50] + '...' if len(tweet) > 50 else tweet,
        'true': true_label,
        'predicted': predicted,
        'compound_score': round(compound, 3),
        'correct': true_label == predicted
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

accuracy = df['correct'].mean()
print(f"\nAccuracy: {accuracy*100:.1f}%")

# Test on new text
test_texts = [
    "This is absolutely fantastic work!",
    "I am so disappointed with this outcome",
    "The results were as expected"
]

print("\nNew Predictions:")
print("-" * 40)
for text in test_texts:
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = '😊 Positive'
    elif compound <= -0.05:
        sentiment = '😞 Negative'
    else:
        sentiment = '😐 Neutral'
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (score: {compound:.3f})")
    print()

print("Done!")