import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import classification_report, confusion_matrix
from src.model import load_model
from src.preprocessing import TextPreprocessor

def run_pipeline():
    model = load_model('model.pkl')
    if model is None:
        print("Model not found.")
        return

    test_csv = 'data/raw/testing_enriched.csv'
    if not os.path.exists(test_csv):
        print(f"{test_csv} not found. Run enrichment script first.")
        return

    df = pd.read_csv(test_csv)
    
    # Filter for eras the model knows
    df_known = df[df['era'].isin(model.classes_)].copy()
    if len(df_known) == 0:
        print("No songs found in testing data belonging to model's known eras.")
        return

    print(f"Evaluating model on {len(df_known)} songs from 6 known eras...")

    # Feature Engineering (Audio)
    df_known['acoustic_energy_ratio'] = df_known['acousticness'] / (df_known['energy'] + 0.001)
    df_known['dance_valence_product'] = df_known['danceability'] * df_known['valence']
    df_known['loudness_norm'] = (df_known['loudness'] + 60) / 60
    df_known['tempo_bin'] = pd.cut(df_known['tempo'], bins=[0, 90, 110, 130, 160, 250], labels=[0, 1, 2, 3, 4]).astype(float)
    df_known['speech_acoustic_diff'] = df_known['speechiness'] - df_known['acousticness']

    # Text Preprocessing & Sentiment
    preprocessor = TextPreprocessor()
    X_text_cleaned, sentiments = preprocessor.transform_with_features(df_known['lyrics'].tolist())
    sentiment_df = pd.DataFrame(sentiments)
    df_known = pd.concat([df_known.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)

    for i in range(12):
        df_known[f'key_{float(i)}'] = (df_known['key'] == i).astype(float)
    for i in range(2):
        df_known[f'mode_{float(i)}'] = (df_known['mode'] == i).astype(float)

    audio_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 
        'tempo', 'duration_ms',
        'acoustic_energy_ratio', 'dance_valence_product', 'loudness_norm',
        'tempo_bin', 'speech_acoustic_diff',
        'compound', 'pos', 'neg', 'neu' # Sentiment Features
    ] + [f'key_{float(i)}' for i in range(12)] + [f'mode_{float(i)}' for i in range(2)]

    X_audio = df_known[audio_cols].values
    X_text = np.array(X_text_cleaned)
    
    # Multimodal Prediction
    y_true = df_known['era'].values
    y_pred = model.predict(X_audio, X_text)
    
    # Audio-only for comparison
    X_audio_scaled = model.audio_scaler.transform(X_audio)
    y_audio_probs = model.audio_classifier.predict_proba(X_audio_scaled)
    y_pred_audio = model.classes_[np.argmax(y_audio_probs, axis=1)]
    
    # Generate Report
    print("\nMULTIMODAL CLASSIFICATION REPORT")
    print("-" * 30)
    print(classification_report(y_true, y_pred, zero_division=0))
    
    print("\nAUDIO-ONLY CLASSIFICATION REPORT")
    print("-" * 30)
    print(classification_report(y_true, y_pred_audio, zero_division=0))
    
    # Save errors for analysis
    errors = df_known[y_pred != y_true].copy()
    errors['predicted'] = y_pred[y_pred != y_true]
    errors.to_csv('artifacts/detailed_errors.csv', index=False)
    print(f"\nSaved {len(errors)} misclassifications to artifacts/detailed_errors.csv")

if __name__ == '__main__':
    run_pipeline()
