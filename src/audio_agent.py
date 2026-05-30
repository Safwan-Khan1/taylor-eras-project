import numpy as np
import pickle
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# --- New Imports for LLM Reasoning ---
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "openai/gpt-oss-20b:free"


class AudioAgent:
    """Numerical agent that classifies Taylor Swift audio features into Eras."""

    def __init__(self, n_components=8):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        
        self.classifier = RandomForestClassifier(
            n_estimators=150, 
            random_state=42, 
            class_weight='balanced',
            max_depth=10
        )
        
        self.features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'duration_ms'
        ]
        
        self.classes_ = None
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X_df, era_labels):
        """Fits the scaler, PCA, and classifier."""
        X_raw = X_df[self.features].values
        
        X_scaled = self.scaler.fit_transform(X_raw)
        X_pca = self.pca.fit_transform(X_scaled)
        
        self.classifier.fit(X_pca, era_labels)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_with_evidence(self, audio_data: dict, use_llm: bool = True) -> dict:
        """
        Classify audio and return debate-ready evidence.
        Expects a dictionary of the 12 audio features.
        """
        if not self.is_fitted:
            raise RuntimeError("AudioAgent must be fitted before calling predict_with_evidence().")

        raw_values = [[audio_data.get(feat, 0) for feat in self.features]]
        
        scaled_data = self.scaler.transform(raw_values)
        pca_data = self.pca.transform(scaled_data)
        
        proba = self.classifier.predict_proba(pca_data)[0]
        probabilities = {era: float(p) for era, p in zip(self.classes_, proba)}
        predicted_era = max(probabilities, key=probabilities.get)
        
        evidence = {
            "raw_features": {feat: raw_values[0][i] for i, feat in enumerate(self.features)},
            "pca_components": pca_data[0].tolist()
        }

        # --- Check if we should use the LLM or the Template ---
        if use_llm and OPENROUTER_API_KEY:
            reasoning = self._llm_reasoning(
                predicted_era, probabilities, audio_data
            )
        else:
            reasoning = self._template_reasoning(
                predicted_era, probabilities, audio_data
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

    def _llm_reasoning(self, predicted_era, probabilities, raw_data):
        # Format the top 3 predictions so the LLM knows what the close calls were
        top3 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"{era} ({conf:.0%})" for era, conf in top3)
        
        # Grab a few key audio stats to feed to the LLM
        dance = raw_data.get('danceability', 0)
        energy = raw_data.get('energy', 0)
        acoustic = raw_data.get('acousticness', 0)
        valence = raw_data.get('valence', 0)
        tempo = raw_data.get('tempo', 0)

        confidence = probabilities.get(predicted_era, 0)
        confidence_note = (
            "Classifier confidence is low — acknowledge genuine uncertainty."
            if confidence < 0.4 else ""
        )

        # This is the prompt that instructs the LLM on how to act
        prompt = f"""You are the Audio Analysis Agent in a multi-agent debate system classifying Taylor Swift songs into their Eras.

Your Random Forest pipeline has produced the following signals for an unknown song:

- Top prediction: {predicted_era} ({confidence:.0%} confidence)
- Top 3 candidates: {top3_str}
- Key audio features: Danceability: {dance:.2f}, Energy: {energy:.2f}, Acousticness: {acoustic:.2f}, Valence: {valence:.2f}, Tempo: {tempo:.0f} BPM

Write a confident debate argument (2-3 complete sentences) making the case that this song belongs to the **{predicted_era}** era.
Reference the specific numerical audio features as evidence (e.g., "The high energy of 0.85 points to..."). Acknowledge the runner-up era if the gap is close.
Do not use bullet points. Write in first person as the Audio Agent.
{confidence_note}
IMPORTANT: Always end with a complete sentence. Do not trail off or use ellipsis."""

        try:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            response = client.chat.completions.create(
                model=OPENROUTER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # If the API fails, fall back to our hardcoded templates!
            return self._template_reasoning(
                predicted_era, probabilities, raw_data
            ) + f" [LLM unavailable: {e}]"

    # ------------------------------------------------------------------
    # Template reasoning fallback
    # ------------------------------------------------------------------

    def _template_reasoning(self, era, probs, raw_data):
        top2 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:2]
        confidence = top2[0][1]
        runner_up = top2[1][0] if len(top2) > 1 else None
        runner_up_conf = top2[1][1] if len(top2) > 1 else 0.0
        
        dance = raw_data.get('danceability', 0)
        energy = raw_data.get('energy', 0)
        acoustic = raw_data.get('acousticness', 0)
        valence = raw_data.get('valence', 0)
        
        reasoning = f"Based on the acoustic signature, I predict **{era}** ({confidence:.0%} confidence). "
        
        if acoustic > 0.65:
            reasoning += f"The high acousticness ({acoustic:.2f}) strongly suggests a stripped-down, indie/folk sound. "
        elif energy > 0.75 and dance > 0.6:
            reasoning += f"The high energy ({energy:.2f}) and danceability ({dance:.2f}) point to heavy pop/synth production. "
        elif valence < 0.3:
            reasoning += f"The low valence ({valence:.2f}) indicates a melancholic or darker instrumental tone. "
        else:
            reasoning += "The tempo and instrumental patterns match the production style of this era. "

        if runner_up and runner_up_conf > 0.2:
            reasoning += f"However, the beat profile also shares similarities with {runner_up} ({runner_up_conf:.0%})."
            
        return reasoning

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path='models/audio_agent.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='models/audio_agent.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)