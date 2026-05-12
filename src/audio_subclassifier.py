import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

class AudioSubClassifier(BaseEstimator, ClassifierMixin):
    """
    Expert Sub-Model for Acoustic Feature Analysis.
    Owned by Member 3.
    """
    def __init__(self, C=5.0, kernel='rbf'):
        self.C = C
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.model = SVC(probability=True, kernel=self.kernel, C=self.C, 
                         gamma='scale', class_weight='balanced', random_state=42)
        
    def _engineer_features(self, df):
        """
        Internal logic for derived acoustic metrics.
        Encapsulates Member 3's feature engineering expertise.
        """
        df = df.copy()
        # Ratios & Products
        df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 0.001)
        df['dance_valence_product'] = df['danceability'] * df['valence']
        
        # Normalization
        df['loudness_norm'] = (df['loudness'] + 60) / 60
        
        # Binned Tempo (Member 3 Specialty)
        # 0: Slow, 1: Chill, 2: Mid, 3: Upbeat, 4: Fast
        df['tempo_bin'] = pd.cut(df['tempo'], bins=[0, 90, 110, 130, 160, 250], 
                                 labels=[0, 1, 2, 3, 4]).astype(float)
        
        # Contrastive Features
        df['speech_acoustic_diff'] = df['speechiness'] - df['acousticness']
        
        # Key/Mode One-Hot Encoding
        key_dummies = pd.get_dummies(df['key'], prefix='key').astype(float)
        mode_dummies = pd.get_dummies(df['mode'], prefix='mode').astype(float)
        
        # Padding missing keys if necessary
        for k in range(12):
            if f'key_{k}' not in key_dummies.columns: key_dummies[f'key_{k}'] = 0.0
        for m in [0, 1]:
            if f'mode_{m}' not in mode_dummies.columns: mode_dummies[f'mode_{m}'] = 0.0
            
        audio_cols = [
            'danceability', 'energy', 'loudness', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 'valence', 
            'tempo', 'duration_ms',
            'acoustic_energy_ratio', 'dance_valence_product', 'loudness_norm',
            'tempo_bin', 'speech_acoustic_diff'
        ] + [f'key_{i}' for i in range(12)] + [f'mode_{i}' for i in range(2)]
        
        return df[audio_cols]

    def fit(self, X_df, y):
        X_engineered = self._engineer_features(X_df)
        X_scaled = self.scaler.fit_transform(X_engineered)
        self.model.fit(X_scaled, y)
        self.classes_ = self.model.classes_
        self.feature_names_ = X_engineered.columns.tolist()
        return self

    def predict_proba(self, X_df):
        X_engineered = self._engineer_features(X_df)
        X_scaled = self.scaler.transform(X_engineered)
        return self.model.predict_proba(X_scaled)

    def predict(self, X_df):
        X_engineered = self._engineer_features(X_df)
        X_scaled = self.scaler.transform(X_engineered)
        return self.model.predict(X_scaled)

    def get_acoustic_signature(self):
        """
        Returns a high-level summary of the typical acoustic markers.
        """
        return {
            "1989": "High Danceability + Mid Valence (Synth-Pop)",
            "Reputation": "Low Acousticness + High Energy + Low Loudness Norm (Electronic/Trap)",
            "Folklore": "High Acousticness + Low Energy + Long Durations (Indie-Folk)",
            "Lover": "High Valence + Mid Energy (Bubblegum Pop)",
            "Evermore": "High Acousticness + Minor Keys (Alt-Folk)",
            "Midnights": "Mid Tempo + Low Speechiness (Electro-Pop)"
        }
