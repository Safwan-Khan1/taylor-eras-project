import pandas as pd
import numpy as np
import pickle
import os
import json
from src.preprocessing import TextPreprocessor
from src.model import MultimodalAudioLyricsModel, load_model

def verify():
    # 1. Load Model
    model = load_model('model.pkl')
    if model is None:
        print("Model not found at model.pkl")
        return

    # 2. Load Testing Data
    test_df = pd.read_csv('data/raw/testing.csv')
    
    # 3. Load Ground Truth/Lyrics Source (since testing.csv is missing them)
    source_df = pd.read_csv('data/raw/taylor_eras_multimodal_dataset.csv')
    
    # 4. Select 4 songs from the 6 known eras
    known_eras = ['1989', 'Reputation', 'Lover', 'Folklore', 'Evermore', 'Midnights']
    # Mapping for album to era for ground truth
    album_to_era = {
        '1989': '1989',
        'reputation': 'Reputation',
        'Lover': 'Lover',
        'folklore': 'Folklore',
        'evermore': 'Evermore',
        'Midnights': 'Midnights'
    }
    
    selected_songs = [
        'Style',          # 1989
        'champagne problems', # Evermore
        'Karma',          # Midnights
        '...Ready For It?' # Reputation
    ]
    
    # Filter test_df for these songs
    subset = test_df[test_df['track_name'].isin(selected_songs)].copy()
    
    # Merge lyrics from source_df
    # We'll use track_name for merging. 
    # Drop empty lyrics column from subset first to avoid collision
    subset = subset.drop(columns=['lyrics'])
    source_subset = source_df[['track_name', 'lyrics', 'era']].drop_duplicates('track_name')
    subset = subset.merge(source_subset, on='track_name', how='left')
    
    # Fill lyrics if missing (though they should be there)
    subset['lyrics'] = subset['lyrics'].fillna("Missing lyrics")
    subset['actual_era'] = subset['era']
    
    if subset['actual_era'].isnull().any():
        print("Warning: Some songs could not be matched to the source dataset for era/lyrics.")
        print(subset[subset['actual_era'].isnull()][['track_name']])
    
    # 5. Preprocess Features (Must match train.py logic)
    # Audio Engineering
    subset['acoustic_energy_ratio'] = subset['acousticness'] / (subset['energy'] + 0.001)
    subset['dance_valence_product'] = subset['danceability'] * subset['valence']
    subset['loudness_norm'] = (subset['loudness'] + 60) / 60
    subset['tempo_bin'] = pd.cut(subset['tempo'], bins=[0, 90, 110, 130, 160, 250], labels=[0, 1, 2, 3, 4]).astype(float)
    subset['speech_acoustic_diff'] = subset['speechiness'] - subset['acousticness']

    # One-Hot Encoding for Key and Mode (Must match training columns)
    # The training set had keys 0-11 and modes 0-1.
    for i in range(12):
        col = f'key_{float(i)}'
        subset[col] = (subset['key'] == i).astype(float)
    for i in range(2):
        col = f'mode_{float(i)}'
        subset[col] = (subset['mode'] == i).astype(float)

    audio_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 
        'tempo', 'duration_ms',
        'acoustic_energy_ratio', 'dance_valence_product', 'loudness_norm',
        'tempo_bin', 'speech_acoustic_diff'
    ] + [f'key_{float(i)}' for i in range(12)] + [f'mode_{float(i)}' for i in range(2)]

    X_audio = subset[audio_cols].values
    
    # Text Preprocessing
    preprocessor = TextPreprocessor()
    X_text = np.array(preprocessor.transform(subset['lyrics'].tolist()))

    # 6. Predict (Multimodal vs Audio-Only)
    # Multimodal
    predictions_multi = model.predict(X_audio, X_text)
    
    # Audio-Only Sub-prediction
    # If the model is 'late' fusion, we can tap into the audio_classifier directly
    # Note: We must scale the audio first using the model's scaler
    X_audio_scaled = model.audio_scaler.transform(X_audio)
    audio_probs = model.audio_classifier.predict_proba(X_audio_scaled)
    predictions_audio = model.classes_[np.argmax(audio_probs, axis=1)]
    
    # 7. Print Results
    print("\n" + "="*60)
    print("COMPARISON: MULTIMODAL VS AUDIO-ONLY (WITHOUT LYRICS)")
    print("="*60)
    
    correct_multi = 0
    correct_audio = 0
    for idx, row in subset.iterrows():
        p_multi = predictions_multi[idx]
        p_audio = predictions_audio[idx]
        actual = row['actual_era']
        
        is_multi_correct = (str(p_multi).lower() == str(actual).lower())
        is_audio_correct = (str(p_audio).lower() == str(actual).lower())
        
        if is_multi_correct: correct_multi += 1
        if is_audio_correct: correct_audio += 1
        
        print(f"Song: {row['track_name']} ({actual})")
        print(f"  [Multimodal] Predicted: {p_multi:12} | {'✅' if is_multi_correct else '❌'}")
        print(f"  [Audio Only] Predicted: {p_audio:12} | {'✅' if is_audio_correct else '❌'}")
        print("-" * 40)
    
    print(f"\nMultimodal Accuracy: {correct_multi}/4 ({(correct_multi/4)*100}%)")
    print(f"Audio-Only Accuracy: {correct_audio}/4 ({(correct_audio/4)*100}%)")
    print("="*60)

if __name__ == '__main__':
    verify()
