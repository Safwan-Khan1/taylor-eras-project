import numpy as np
import pickle
import os
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import Word2Vec
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import word_tokenize
from openai import OpenAI
from dotenv import load_dotenv
from .preprocessing import TextPreprocessor

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"

ERA_DESCRIPTIONS = {
    "Taylor Swift": "country debut, innocent and young, simple love stories, small-town themes",
    "Fearless":     "romantic fairytale country-pop, hopeful and dreamy, young love",
    "Speak Now":    "entirely self-written, theatrical storytelling, bold and adventurous",
    "Red":          "emotional whiplash between joy and heartbreak, experimental pop-country",
    "1989":         "pure synth-pop, confident, New York City, moving on and independence",
    "Reputation":   "dark, defensive, hip-hop influenced, themes of betrayal and reclaiming power",
    "Lover":        "pastel pop, romantic and optimistic, vulnerable and joyful",
    "Folklore":     "indie folk, fictional storytelling, melancholy, introspective, cottagecore",
    "Evermore":     "indie folk continuation, heartbreak, grief, more rustic and wintery",
    "Midnights":    "late-night synth-pop, self-reflection, insomnia, paranoia, modern romance",
}

# Distinctive lyrical/thematic keywords per era.
# Matched against raw lowercase lyrics so multi-word phrases work.
ERA_KEYWORDS = {
    "Taylor Swift": [
        "small town", "boots", "bleacher", "georgia", "front porch",
        "friday night", "tim mcgraw", "teardrops", "pickup truck", "country road",
    ],
    "Fearless": [
        "fairytale", "romeo", "juliet", "white horse", "prince",
        "fearless", "fifteen", "love story", "breathe", "you belong with me",
    ],
    "Speak Now": [
        "enchanted", "sparks fly", "mean", "long live", "dear john",
        "speak now", "last kiss", "back to december",
    ],
    "Red": [
        "all too well", "treacherous", "burning", "scarlet",
        "trouble", "begin again", "state of grace", "holy ground", "22",
    ],
    "1989": [
        "new york", "shake", "blank space", "bad blood", "style",
        "clean", "wildest dreams", "out of the woods", "polaroid", "photograph",
    ],
    "Reputation": [
        "reputation", "snake", "delicate", "gorgeous", "getaway",
        "phantom", "killer", "ransom", "thieves", "darkness", "ready for it",
    ],
    "Lover": [
        "lover", "daylight", "cornelia", "cruel summer", "paper rings",
        "rainbow", "butterflies", "calm down", "london", "afterglow",
    ],
    "Folklore": [
        "cardigan", "august", "betty", "folklore", "exile",
        "mirrorball", "seven", "james", "invisible string", "mad woman",
    ],
    "Evermore": [
        "willow", "champagne problems", "gold rush", "tolerate",
        "dorothea", "coney island", "marjorie", "closure", "evermore",
    ],
    "Midnights": [
        "lavender haze", "anti hero", "midnight rain", "vigilante",
        "bejeweled", "labyrinth", "karma", "mastermind", "glitch",
    ],
}


class TextAgent:
    """NLP agent that classifies Taylor Swift lyrics into Eras using TF-IDF + era keyword features."""

    def __init__(self, max_features=3000, C=1.5, classifier=None):
        self.max_features = max_features
        self.C = C
        self.preprocessor = TextPreprocessor()

        self.tfidf = TfidfVectorizer(
            max_features=8000,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.85,
        )

        self.char_tfidf = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            max_features=3000,
            sublinear_tf=True,
            min_df=2,
        )
        self.n_char_features_ = None

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = LogisticRegression(
                C=self.C,
                class_weight='balanced',
                max_iter=2000,
                random_state=42,
                solver='lbfgs',
            )

        self.w2v_model = None
        self.w2v_size = 100
        self.use_w2v = False

        self.lda = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=20,
        )
        self.lda_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            min_df=3,
            max_df=0.80,
        )
        self.n_lda_features_ = None
        self.lda_fitted = False

        self.classes_ = None
        self.n_tfidf_features_ = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Keyword features
    # ------------------------------------------------------------------

    def _keyword_features(self, raw_texts: list) -> np.ndarray:
        """Returns (n_samples, n_eras) matrix of normalized keyword match scores."""
        era_order = list(ERA_KEYWORDS.keys())
        result = np.zeros((len(raw_texts), len(era_order)))
        for i, text in enumerate(raw_texts):
            text_lower = text.lower()
            for j, era in enumerate(era_order):
                keywords = ERA_KEYWORDS[era]
                matches = sum(1 for kw in keywords if kw in text_lower)
                result[i, j] = matches / len(keywords)
        return result

    # ------------------------------------------------------------------
    # Word2Vec features
    # ------------------------------------------------------------------

    def _w2v_features(self, texts: list) -> np.ndarray:
        """Mean-pool Word2Vec vectors. Returns zeros if model not fitted."""
        if self.w2v_model is None:
            return np.zeros((len(texts), self.w2v_size))
        vecs = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            valid = [self.w2v_model.wv[t] for t in tokens if t in self.w2v_model.wv]
            vecs.append(np.mean(valid, axis=0) if valid else np.zeros(self.w2v_size))
        return np.array(vecs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, lyrics_list, era_labels):
        clean = self.preprocessor.transform(lyrics_list)

        # TF-IDF word features
        X_tfidf = self.tfidf.fit_transform(clean)
        self.n_tfidf_features_ = X_tfidf.shape[1]

        # Character n-gram features
        X_char = self.char_tfidf.fit_transform(clean)
        self.n_char_features_ = X_char.shape[1]

        # Keyword features
        X_kw = csr_matrix(self._keyword_features(lyrics_list))

        # Word2Vec — trained for topic reporting but not included in classifier features
        try:
            tokenized = [word_tokenize(t.lower()) for t in clean]
            self.w2v_model = Word2Vec(
                tokenized, vector_size=self.w2v_size,
                window=5, min_count=2, workers=4, epochs=10, seed=42
            )
            self.use_w2v = True
        except Exception as e:
            print(f"[TextAgent] Word2Vec skipped: {e}")
            self.use_w2v = False

        # LDA — fitted for topic reporting but not included in classifier features
        try:
            X_lda_input = self.lda_vectorizer.fit_transform(clean)
            self.lda.fit(X_lda_input)
            self.lda_fitted = True
        except Exception as e:
            print(f"[TextAgent] LDA skipped: {e}")
            self.lda_fitted = False

        X = hstack([X_tfidf, X_char, X_kw])

        self.classifier.fit(X, era_labels)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_with_evidence(self, lyrics: str, use_llm: bool = True) -> dict:
        """
        Classify lyrics and return debate-ready evidence.

        Returns
        -------
        {
            "predicted_era":  str,           # e.g. "Folklore"
            "probabilities":  dict[str,float], # all 8 eras
            "evidence": {
                "top_keywords": list[str],   # top TF-IDF signal words
                "top_topics": [              # top 2 LDA topics
                    {"topic_id": int, "top_words": list[str], "weight": float}
                ],
                "sentiment": dict,           # VADER compound/pos/neg/neu
            },
            "reasoning": str,                # 2-3 sentence debate argument (LLM or template)
        }

        NOTE: Evermore and Midnights are NOT valid predicted_era values.
        The model was trained on 8 eras only:
        Taylor Swift, Fearless, Speak Now, Red, 1989, Reputation, Lover, Folklore.
        """
        if not self.is_fitted:
            raise RuntimeError("TextAgent must be fitted before calling predict_with_evidence().")

        clean = self.preprocessor.clean_text(lyrics)
        sentiment = self.preprocessor.get_sentiment(lyrics)

        X_tfidf = self.tfidf.transform([clean])
        X_char  = self.char_tfidf.transform([clean])
        X_kw    = csr_matrix(self._keyword_features([lyrics]))

        # W2V and LDA are not in the classifier feature stack (see fit())
        X = hstack([X_tfidf, X_char, X_kw])

        proba = self.classifier.predict_proba(X)[0]
        probabilities = {era: float(p) for era, p in zip(self.classes_, proba)}
        predicted_era = max(probabilities, key=probabilities.get)

        top_keywords = self._extract_top_keywords(X_tfidf, predicted_era, n=8)

        # Top 2 LDA topics for this prediction
        top_topics = []
        if self.lda_fitted:
            X_lda_input = self.lda_vectorizer.transform([clean])
            topic_dist = self.lda.transform(X_lda_input)[0]
            top_topic_idx = topic_dist.argsort()[::-1][:2]
            lda_vocab = np.array(self.lda_vectorizer.get_feature_names_out())
            for idx in top_topic_idx:
                top_words = lda_vocab[self.lda.components_[idx].argsort()[::-1][:5]]
                top_topics.append({
                    "topic_id": int(idx),
                    "top_words": top_words.tolist(),
                    "weight": float(topic_dist[idx]),
                })

        evidence = {
            "top_keywords": top_keywords,
            "top_topics": top_topics,
            "sentiment": sentiment,
        }

        if use_llm and OPENROUTER_API_KEY:
            reasoning = self._llm_reasoning(
                lyrics, predicted_era, probabilities, top_keywords, sentiment
            )
        else:
            reasoning = self._template_reasoning(
                predicted_era, probabilities, top_keywords, sentiment
            )

        return {
            "predicted_era": predicted_era,
            "probabilities": probabilities,
            "evidence": evidence,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # LLM reasoning via OpenRouter
    # ------------------------------------------------------------------

    def _llm_reasoning(self, lyrics, predicted_era, probabilities, top_keywords, sentiment):
        top3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"{era} ({conf:.0%})" for era, conf in top3)
        kw_str = ", ".join(top_keywords[:6]) if top_keywords else "none detected"
        sent_label = (
            "strongly negative" if sentiment['compound'] < -0.5 else
            "mildly negative"   if sentiment['compound'] < -0.1 else
            "neutral"           if sentiment['compound'] <  0.1 else
            "mildly positive"   if sentiment['compound'] <  0.5 else
            "strongly positive"
        )
        era_desc = ERA_DESCRIPTIONS.get(predicted_era, "")

        confidence = probabilities.get(predicted_era, 0)
        confidence_note = (
            "Classifier confidence is low — acknowledge genuine uncertainty."
            if confidence < 0.4 else ""
        )

        prompt = f"""You are the Text Analysis Agent in a multi-agent debate system classifying Taylor Swift songs into their Eras.

Your ML pipeline has produced the following signals for an unknown song:

- TF-IDF classifier top prediction: {predicted_era} ({probabilities.get(predicted_era, 0):.0%} confidence)
- Top 3 candidates: {top3_str}
- Key lyrical signals (TF-IDF): {kw_str}
- Sentiment: {sent_label} (compound score: {sentiment['compound']:.2f})

Era context — {predicted_era}: {era_desc}

Write a confident debate argument (2-3 complete sentences) making the case that this song belongs to the **{predicted_era}** era.
Reference the lyrical signals and sentiment as evidence. Acknowledge the runner-up era if the gap is close.
Do not use bullet points. Write in first person as the Text Agent.
{confidence_note}
IMPORTANT: Always end with a complete sentence. Do not trail off or use ellipsis."""

        try:
            import httpx
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                http_client=httpx.Client(verify=False),
            )
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return self._template_reasoning(
                predicted_era, probabilities, top_keywords, sentiment
            ) + f" [LLM unavailable: {e}]"

    # ------------------------------------------------------------------
    # Template reasoning fallback
    # ------------------------------------------------------------------

    def _template_reasoning(self, era, probs, keywords, sentiment):
        top2 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
        confidence = top2[0][1]
        runner_up = top2[1][0] if len(top2) > 1 else None
        runner_up_conf = top2[1][1] if len(top2) > 1 else 0.0
        kw_str = ", ".join(f'"{w}"' for w in keywords[:5]) if keywords else "none"
        sent_label = (
            "strongly negative" if sentiment['compound'] < -0.5 else
            "mildly negative"   if sentiment['compound'] < -0.1 else
            "neutral"           if sentiment['compound'] <  0.1 else
            "mildly positive"   if sentiment['compound'] <  0.5 else
            "strongly positive"
        )
        reasoning = (
            f"TF-IDF + keyword classifier predicts **{era}** with {confidence:.0%} confidence. "
            f"Key lyrical signals: {kw_str}. "
            f"Overall sentiment is {sent_label} (compound={sentiment['compound']:.2f}). "
        )
        if runner_up and runner_up_conf > 0.2:
            reasoning += f"Second candidate is {runner_up} ({runner_up_conf:.0%})."
        return reasoning

    # ------------------------------------------------------------------
    # Keyword extraction (TF-IDF portion only)
    # ------------------------------------------------------------------

    def _extract_top_keywords(self, X_tfidf, predicted_era, n=8):
        feature_names = np.array(self.tfidf.get_feature_names_out())
        tfidf_weights = X_tfidf.toarray()[0]
        if hasattr(self.classifier, 'coef_'):
            era_idx = list(self.classes_).index(predicted_era)
            clf_weights = self.classifier.coef_[era_idx, :self.n_tfidf_features_]
            combined = tfidf_weights * np.maximum(clf_weights, 0)
        elif hasattr(self.classifier, 'feature_importances_'):
            # RF: use global feature importances (not per-class); slice to TF-IDF portion
            importances = self.classifier.feature_importances_[:self.n_tfidf_features_]
            combined = tfidf_weights * importances
        else:
            combined = tfidf_weights
        top_idx = combined.argsort()[::-1][:n]
        keywords = [feature_names[i] for i in top_idx if tfidf_weights[i] > 0]
        return keywords[:n]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path='models/text_agent.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        if self.w2v_model is not None:
            self.w2v_model.save(path.replace('.pkl', '_w2v.model'))

    @staticmethod
    def load(path='models/text_agent.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
