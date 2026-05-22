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
        # Strip Genius embed footer e.g. "You might also like ... 42Embed"
        text = re.sub(r'you might also like.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+embed\s*$', '', text.strip(), flags=re.IGNORECASE)
        # Remove bracketed section headers like [Verse 1], [Chorus], [Pre-Chorus], etc.
        text = re.sub(r'\[.*?\]', '', text)
        # Lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Strip non-ASCII words (removes Cyrillic, CJK, Arabic, etc.)
        text = re.sub(r'\b[^\x00-\x7F]+\b', '', text)
        # Remove structural keywords and common filler sounds
        text = re.sub(r'\b(verse|chorus|bridge|outro|intro|prechorus|hook)\b', '', text)
        text = re.sub(r'\b(oh|yeah|ooh|ah|la|na|hey|whoa|mmm|hmm|uh|um|gonna|wanna|gotta)\b', '', text)
        # Tokenize
        words = text.split()
        # Keep only ASCII-only tokens, remove stopwords, lemmatize
        cleaned_words = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word.isascii() and word not in self.stop_words and len(word) > 1
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
