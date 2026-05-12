import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import pickle
import os


class MultimodalAudioLyricsModel(BaseEstimator, ClassifierMixin):
    def __init__(self, fusion_type='late'):
        self.fusion_type = fusion_type
        self.audio_scaler = StandardScaler()
        self.text_vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
        )

        if self.fusion_type == 'early':
            estimators = [
                ('svm', SVC(probability=True, kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', random_state=0)),
                ('rf', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=0))),
            ]
            self.classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced'))
        elif self.fusion_type == 'late':
            self.audio_classifier = SVC(probability=True, kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', random_state=0)
            self.text_classifier = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=0))
            self.meta_classifier = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
        elif self.fusion_type == 'text_only':
            self.text_classifier = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', max_iter=2000, random_state=0))
        elif self.fusion_type == 'audio_only':
            self.audio_classifier = SVC(probability=True, kernel='rbf', C=5.0, gamma='scale', class_weight='balanced', random_state=0)

    def _oof_proba(self, clf, X, y, cv=5):
        """Out-of-fold probability predictions to avoid meta-learner data leakage."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        classes = np.unique(y)
        oof = np.zeros((len(y), len(classes)))
        for train_idx, val_idx in skf.split(X, y):
            fold_clf = clone(clf)
            fold_clf.fit(X[train_idx], y[train_idx])
            # Align columns to global class order in case a fold is missing a class
            proba = fold_clf.predict_proba(X[val_idx])
            for j, cls in enumerate(fold_clf.classes_):
                col = np.where(classes == cls)[0][0]
                oof[val_idx, col] = proba[:, j]
        return oof

    def fit(self, X_audio, X_text, y):
        if self.fusion_type in ['early', 'late', 'audio_only']:
            X_audio_scaled = self.audio_scaler.fit_transform(X_audio)
        if self.fusion_type in ['early', 'late', 'text_only']:
            X_text_vec = self.text_vectorizer.fit_transform(X_text).toarray()
            self.vocab_size_ = len(self.text_vectorizer.get_feature_names_out())
            self.sparsity_ = 1.0 - (np.count_nonzero(X_text_vec) / float(X_text_vec.size))

        if self.fusion_type == 'early':
            X_combined = np.hstack((X_audio_scaled, X_text_vec))
            self.classifier.fit(X_combined, y)
            self.classes_ = self.classifier.classes_

        elif self.fusion_type == 'late':
            # OOF predictions for meta-learner (no data leakage)
            audio_oof = self._oof_proba(self.audio_classifier, X_audio_scaled, y)
            text_oof = self._oof_proba(self.text_classifier, X_text_vec, y)
            # Weighted Late Fusion (Audio 0.6, Text 0.4)
            self.meta_classifier.fit(np.hstack([audio_oof * 0.6, text_oof * 0.4]), y)
            # Refit base classifiers on full training set
            self.audio_classifier.fit(X_audio_scaled, y)
            self.text_classifier.fit(X_text_vec, y)
            self.classes_ = self.meta_classifier.classes_

        elif self.fusion_type == 'text_only':
            self.text_classifier.fit(X_text_vec, y)
            self.classes_ = self.text_classifier.classes_

        elif self.fusion_type == 'audio_only':
            self.audio_classifier.fit(X_audio_scaled, y)
            self.classes_ = self.audio_classifier.classes_

        return self

    def predict_proba(self, X_audio, X_text):
        if self.fusion_type in ['early', 'late', 'audio_only']:
            X_audio_scaled = self.audio_scaler.transform(X_audio)
        if self.fusion_type in ['early', 'late', 'text_only']:
            X_text_vec = self.text_vectorizer.transform(X_text).toarray()

        if self.fusion_type == 'early':
            X_combined = np.hstack((X_audio_scaled, X_text_vec))
            return self.classifier.predict_proba(X_combined)
        elif self.fusion_type == 'late':
            audio_probs = self.audio_classifier.predict_proba(X_audio_scaled)
            text_probs = self.text_classifier.predict_proba(X_text_vec)
            
            # DYNAMIC WEIGHTING LOGIC
            # If one classifier is extremely confident, we give it even more weight
            # Otherwise, use the meta-classifier's learned aggregation
            
            audio_max = np.max(audio_probs, axis=1)
            text_max = np.max(text_probs, axis=1)
            
            # Base results from meta-classifier
            meta_input = np.hstack([audio_probs * 0.6, text_probs * 0.4])
            final_probs = self.meta_classifier.predict_proba(meta_input)
            
            # Optional: Override with extremely high confidence from a single source
            # If audio is > 95% sure it's 1989/Reputation, trust it more
            for i in range(len(final_probs)):
                if audio_max[i] > 0.95:
                    final_probs[i] = 0.8 * audio_probs[i] + 0.2 * text_probs[i]
                elif text_max[i] > 0.95:
                    final_probs[i] = 0.2 * audio_probs[i] + 0.8 * text_probs[i]
            
            return final_probs
        elif self.fusion_type == 'text_only':
            return self.text_classifier.predict_proba(X_text_vec)
        elif self.fusion_type == 'audio_only':
            return self.audio_classifier.predict_proba(X_audio_scaled)

    def predict(self, X_audio, X_text):
        prob = self.predict_proba(X_audio, X_text)
        return self.classes_[np.argmax(prob, axis=1)]


def save_model(model, filepath='model.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath='model.pkl'):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def generate_confusion_matrix(model, X_audio, X_text, y_true):
    y_pred = model.predict(X_audio, X_text)
    cm = confusion_matrix(y_true, y_pred, labels=model.classes_)
    return cm, model.classes_
