import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from src.dataset_generator import create_mock_dataset
from src.preprocessing import TextPreprocessor
from src.model import MultimodalAudioLyricsModel, save_model, generate_confusion_matrix
from src.topic_modeling import TopicModeler

def train():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('artifacts', exist_ok=True)
    train_file = 'data/processed/train_dataset.csv'
    test_file = 'data/processed/test_dataset.csv'
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Processed datasets not found. Please run the dataset split script first.")
        return
    
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    print(f"Loaded {len(df_train)} training records and {len(df_test)} testing records.")
    
    # NLP Preprocessing & sentiment
    print("Preprocessing lyrics and analyzing sentiment...")
    preprocessor = TextPreprocessor()
    # Preprocess Train
    df_train['clean_lyrics'], s_train = preprocessor.transform_with_features(df_train['lyrics'].tolist())
    df_train = pd.concat([df_train, pd.DataFrame(s_train, index=df_train.index)], axis=1)
    
    # Preprocess Test
    df_test['clean_lyrics'], s_test = preprocessor.transform_with_features(df_test['lyrics'].tolist())
    df_test = pd.concat([df_test, pd.DataFrame(s_test, index=df_test.index)], axis=1)
    
    df_train.to_csv('data/processed/taylor_swift_eras_processed.csv', index=False)
    
    # Phase 1B: Engineer Derived Audio Features
    def engineer_features(df):
        df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 0.001)
        df['dance_valence_product'] = df['danceability'] * df['valence']
        df['loudness_norm'] = (df['loudness'] + 60) / 60
        df['tempo_bin'] = pd.cut(df['tempo'], bins=[0, 90, 110, 130, 160, 250], labels=[0, 1, 2, 3, 4]).astype(float)
        df['speech_acoustic_diff'] = df['speechiness'] - df['acousticness']
        
        key_dummies = pd.get_dummies(df['key'], prefix='key').astype(float)
        mode_dummies = pd.get_dummies(df['mode'], prefix='mode').astype(float)
        
        # Ensure all keys (0-11) and modes (0,1) exist even if not in sample
        for k in range(12):
            if f'key_{k}' not in key_dummies.columns: key_dummies[f'key_{k}'] = 0.0
        for m in [0, 1]:
            if f'mode_{m}' not in mode_dummies.columns: mode_dummies[f'mode_{m}'] = 0.0
            
        return pd.concat([df, key_dummies, mode_dummies], axis=1)

    df_train = engineer_features(df_train)
    df_test = engineer_features(df_test)

    audio_cols = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 
        'tempo', 'duration_ms',
        'acoustic_energy_ratio', 'dance_valence_product', 'loudness_norm',
        'tempo_bin', 'speech_acoustic_diff',
        'compound', 'pos', 'neg', 'neu'
    ] + [f'key_{i}' for i in range(12)] + [f'mode_{i}' for i in range(2)]

    X_train_audio = df_train[audio_cols].values
    X_train_text = df_train['clean_lyrics'].to_numpy(dtype=str)
    y_train = df_train['era'].to_numpy(dtype=str)
    
    X_test_audio = df_test[audio_cols].values
    X_test_text = df_test['clean_lyrics'].to_numpy(dtype=str)
    y_test = df_test['era'].to_numpy(dtype=str)
    
    models = {
        'Text Only (RF)': 'text_only',
        'Audio Only (SVM)': 'audio_only',
        'Early Fusion': 'early',
        'Late Fusion': 'late'
    }
    
    results = {}
    best_acc = 0
    best_model = None
    best_classes = None
    best_name = ""

    print("\n" + "="*40)
    print("EXPERIMENTAL RESULTS COMPARISON")
    print("="*40)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, m_type in models.items():
        print(f"\n--- Training {name} ---")
        model = MultimodalAudioLyricsModel(fusion_type=m_type)
        model.fit(X_train_audio, X_train_text, y_train)
        
        # Evaluate on the dedicated Test Set
        y_test_preds = model.predict(X_test_audio, X_test_text)
        
        acc = accuracy_score(y_test, y_test_preds)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_test_preds, average='weighted', zero_division=0)

        results[m_type] = {
            'accuracy': float(acc),
            'precision': float(p),
            'recall': float(r),
            'f1_score': float(f)
        }

        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {p:.4f}")
        print(f"Recall:    {r:.4f}")
        print(f"F1-Score:  {f:.4f}")

        if acc >= best_acc:
            best_acc = acc
            best_model = model
            best_preds = y_test_preds
            best_name = name

        # Print TF-IDF specs for models that use text
        if hasattr(model, 'vocab_size_'):
            print(f"[TF-IDF Specs] Vocabulary Size: {model.vocab_size_}")
            print(f"[TF-IDF Specs] Sparsity (%): {model.sparsity_ * 100:.2f}%")

        # Audio Feature Importance if audio-only (proxied via permutation or coefficient if linear, but using simple visual boxplots below)
            
    # Visualizing Era Differences
    # Sample from train for visualization
    rows = (len(audio_cols) + 3) // 4
    plt.figure(figsize=(15, 3.5 * rows))
    for i, feature in enumerate(audio_cols, 1):
        plt.subplot(rows, 4, i)
        sns.boxplot(x='era', y=feature, data=df_train, hue='era', legend=False)
        plt.xticks(rotation=45)
        plt.title(f'{feature}')
    plt.tight_layout()
    plt.savefig('artifacts/era_differences.png')
    plt.close()
    
    # Topic Modeling
    print("\n--- Running Topic Modeling (LDA) ---")
    tm = TopicModeler(n_components=5)
    tm.fit_transform(X_train_text)
    topics = tm.get_topics()
    for topic_name, words in topics.items():
        print(f"{topic_name}: {', '.join(words)}")
        
    # Generate confusion matrix and per-era report for the best model on test set
    print(f"\n--- Best Model: {best_name} (Test accuracy={best_acc:.4f}) ---")
    best_cm, best_classes = generate_confusion_matrix(best_model, X_test_audio, X_test_text, y_test)
    print(f"\nPer-Era Classification Report (TEST SET - {best_name}):")
    print(classification_report(y_test, best_preds, zero_division=0))

    plt.figure(figsize=(8, 6))
    sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_classes, yticklabels=best_classes, annot_kws={"size": 12})
    plt.title(f'{best_name} Confusion Matrix')
    plt.xlabel('Predicted Era')
    plt.ylabel('Actual Era')
    plt.tight_layout()
    plt.savefig('artifacts/confusion_matrix.png')
    plt.close()

    print("\n--- Saving Final Model and Metrics ---")
    save_model(best_model, 'model.pkl')

    report = classification_report(y_test, best_preds, zero_division=0, output_dict=True)
    
    metrics_to_export = {
        'classes': best_classes.tolist(),
        'vocab_size': int(best_model.vocab_size_) if hasattr(best_model, 'vocab_size_') else None,
        'sparsity_perc': float(best_model.sparsity_) if hasattr(best_model, 'sparsity_') else None,
        'topics': topics,
        'results': results,
        'best_model': best_name,
        'era_report': report
    }
    with open('artifacts/eval_metrics.json', 'w') as f:
        json.dump(metrics_to_export, f, indent=4)

    print("Training complete! Run 'streamlit run app.py' to launch the web portal.")

if __name__ == '__main__':
    train()
