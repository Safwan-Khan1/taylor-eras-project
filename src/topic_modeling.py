import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    def __init__(self, n_components=5):
        self.n_components = n_components
        self.vectorizer = CountVectorizer(stop_words='english', max_df=0.95, min_df=2)
        self.lda = LatentDirichletAllocation(n_components=self.n_components, random_state=42)

    def fit_transform(self, texts):
        dtm = self.vectorizer.fit_transform(texts)
        topic_matrix = self.lda.fit_transform(dtm)
        return topic_matrix

    def get_topics(self, n_top_words=10):
        feature_names = self.vectorizer.get_feature_names_out()
        topics = {}
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics[f'Topic {topic_idx + 1}'] = top_words
        return topics
