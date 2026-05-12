import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.sia = SentimentIntensityAnalyzer()
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove song-structural noise
        text = re.sub(r'verse|chorus|bridge|outro|intro|prechorus|hook', '', text)
        text = re.sub(r'\b(oh|yeah|ooh|ah|la|na)\b', '', text)
        
        # Tokenize (basic whitespace)
        words = text.split()
        # Remove stopwords and lemmatize
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words if word not in self.stop_words
        ]
        return " ".join(cleaned_words)
    
    def get_sentiment(self, text):
        if not isinstance(text, str) or text.strip() == "":
            return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
        return self.sia.polarity_scores(text)

    def transform(self, texts):
        return [self.clean_text(text) for text in texts]

    def transform_with_features(self, texts):
        cleaned = self.transform(texts)
        sentiments = [self.get_sentiment(text) for text in texts]
        return cleaned, sentiments
