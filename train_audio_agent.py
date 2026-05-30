import sys
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import your newly created agent!
from src.audio_agent import AudioAgent

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent
MERGED_CSV = BASE / "TaylorSwiftEras" / "Data" / "Final" / "taylor_merged_df.csv"

ALBUM_ERA = {
    "Taylor Swift":                       "Taylor Swift",
    "Beautiful Eyes":                     "Taylor Swift",
    "The Taylor Swift Holiday Collection": "Taylor Swift",
    "Fearless":                           "Fearless",
    "Fearless (Taylor's Version)":        "Fearless",
    "Speak Now":                          "Speak Now",
    "Red":                                "Red",
    "Red (Taylor's Version)":             "Red",
    "1989":                               "1989",
    "reputation":                         "Reputation",
    "Lover":                              "Lover",
    "folklore":                           "Folklore",
}

TAYLORS_VERSION_ALBUMS = {"Fearless (Taylor's Version)", "Red (Taylor's Version)"}

ERA_ORDER_8 = [
    "Taylor Swift", "Fearless", "Speak Now", "Red",
    "1989", "Reputation", "Lover", "Folklore"
]

def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(MERGED_CSV)

    # Filter out Taylor's Versions exactly like Member 2 did
    original_titles = set(
        df.loc[~df["album_name"].isin(TAYLORS_VERSION_ALBUMS), "track_name"]
        .str.lower().str.strip()
    )
    df = df[
        ~df["album_name"].isin(TAYLORS_VERSION_ALBUMS) |
        ~df["track_name"].str.lower().str.strip().isin(original_titles)
    ].copy()

    df["era"] = df["album_name"].map(ALBUM_ERA)
    
    # We drop NAs based on era to ensure we only have the 8 eras
    df = df.dropna(subset=["era"])
    return df.reset_index(drop=True)

def main():
    df = load_dataset()

    print(f"Dataset: {len(df)} songs across {df['era'].nunique()} eras")
    print(df["era"].value_counts().to_string())
    print()

    print("=== 5-Fold Stratified Cross-Validation (Audio) ===")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    # Features we need to pass to the agent
    audio_cols = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo', 'duration_ms'
    ]

    for fold, (train_idx, val_idx) in enumerate(kf.split(df[audio_cols], df["era"])):
        fold_agent = AudioAgent()
        
        # Fit on training fold
        fold_agent.fit(
            df.iloc[train_idx],
            df["era"].iloc[train_idx].tolist()
        )
        
        # Predict on validation fold
        fold_preds = []
        for _, row in df.iloc[val_idx].iterrows():
            pred = fold_agent.predict_with_evidence(row.to_dict())["predicted_era"]
            fold_preds.append(pred)
            
        score = accuracy_score(df["era"].iloc[val_idx].tolist(), fold_preds)
        cv_scores.append(score)
        print(f"  Fold {fold+1}: {score:.3f}")

    print(f"\nCV Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    print()

    # Create official Train/Test split
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["era"]
    )
    print(f"Train: {len(df_train)} | Test: {len(df_test)}\n")

    # Train Final Agent
    agent = AudioAgent()
    agent.fit(df_train, df_train["era"].tolist())

    # Evaluate on Test Set
    y_true, y_pred = [], []
    for _, row in df_test.iterrows():
        result = agent.predict_with_evidence(row.to_dict())
        y_true.append(row["era"])
        y_pred.append(result["predicted_era"])

    print("\n=== Audio Agent Test Set Performance ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=ERA_ORDER_8)
    disp = ConfusionMatrixDisplay(cm, display_labels=ERA_ORDER_8)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Audio Agent — Confusion Matrix (Test Set)")
    plt.tight_layout()
    
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/audio_agent_confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → models/audio_agent_confusion_matrix.png")

    print("\n=== Sample Audio Prediction ===")
    sample = df_test.iloc[random.randint(0, len(df_test)-1)]
    result = agent.predict_with_evidence(sample.to_dict())
    print(f"Song: {sample['track_name']}  |  True: {sample['era']}  |  Predicted: {result['predicted_era']}")
    print(f"Reasoning: {result['reasoning']}")

    # Save the model
    agent.save("models/audio_agent.pkl")
    print("\nSaved → models/audio_agent.pkl")

if __name__ == "__main__":
    main()