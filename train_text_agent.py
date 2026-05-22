import random
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.text_agent import TextAgent

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

    # Drop Taylor's Version re-recordings where the original already exists,
    # but keep any bonus tracks exclusive to the TV release.
    original_titles = set(
        df.loc[~df["album_name"].isin(TAYLORS_VERSION_ALBUMS), "track_name"]
        .str.lower().str.strip()
    )
    df = df[
        ~df["album_name"].isin(TAYLORS_VERSION_ALBUMS) |
        ~df["track_name"].str.lower().str.strip().isin(original_titles)
    ].copy()

    df["era"]    = df["album_name"].map(ALBUM_ERA)
    df["lyrics"] = df["lyric"]
    df = df.dropna(subset=["era", "lyrics"])
    df = df[df["lyrics"].str.strip() != ""]
    return df[["track_name", "era", "lyrics"]].reset_index(drop=True)


def main():
    df = load_dataset()

    print(f"Dataset: {len(df)} songs across {df['era'].nunique()} eras")
    print(df["era"].value_counts().to_string())
    print()

    print("=== 5-Fold Stratified Cross-Validation ===")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df["lyrics"], df["era"])):
        fold_agent = TextAgent()
        fold_agent.fit(
            df["lyrics"].iloc[train_idx].tolist(),
            df["era"].iloc[train_idx].tolist()
        )
        fold_preds = [
            fold_agent.predict_with_evidence(lyr, use_llm=False)["predicted_era"]
            for lyr in df["lyrics"].iloc[val_idx].tolist()
        ]
        score = accuracy_score(df["era"].iloc[val_idx].tolist(), fold_preds)
        cv_scores.append(score)
        print(f"  Fold {fold+1}: {score:.3f}")

    print(f"\nCV Mean: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    print()

    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["era"]
    )
    print(f"Train: {len(df_train)} | Test: {len(df_test)}\n")

    agent = TextAgent()
    agent.fit(df_train["lyrics"].tolist(), df_train["era"].tolist())

    if agent.lda_fitted:
        print("\n=== LDA Topics ===")
        vocab = np.array(agent.lda_vectorizer.get_feature_names_out())
        for i, comp in enumerate(agent.lda.components_):
            top_words = vocab[comp.argsort()[::-1][:8]]
            print(f"  Topic {i:02d}: {', '.join(top_words)}")

    y_true, y_pred = [], []
    for _, row in df_test.iterrows():
        result = agent.predict_with_evidence(row["lyrics"], use_llm=False)
        y_true.append(row["era"])
        y_pred.append(result["predicted_era"])

    print("\n=== Text Agent Test Set Performance ===")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=ERA_ORDER_8)
    disp = ConfusionMatrixDisplay(cm, display_labels=ERA_ORDER_8)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("Text Agent — Confusion Matrix (Test Set)")
    plt.tight_layout()
    os.makedirs("models", exist_ok=True)
    plt.savefig("models/text_agent_confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → models/text_agent_confusion_matrix.png")

    # Ablation: TF-IDF only baseline
    print("\n=== Ablation: TF-IDF only (no char/w2v/lda) ===")
    tfidf_only = TV(max_features=8000, stop_words='english',
                    ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.85)
    X_tr = tfidf_only.fit_transform(df_train["lyrics"])
    X_te = tfidf_only.transform(df_test["lyrics"])
    lr = LR(C=1.5, class_weight='balanced', max_iter=2000,
            random_state=42, solver='lbfgs')
    lr.fit(X_tr, df_train["era"])
    preds_simple = lr.predict(X_te)
    print(f"TF-IDF only accuracy: {accuracy_score(df_test['era'], preds_simple):.3f}")
    print(f"Full pipeline accuracy: {accuracy_score(y_true, y_pred):.3f}")

    print("\n=== Sample Prediction ===")
    sample = df_test.iloc[random.randint(0, len(df_test)-1)]
    result = agent.predict_with_evidence(sample["lyrics"], use_llm=False)
    print(f"Song: {sample['track_name']}  |  True: {sample['era']}  |  Predicted: {result['predicted_era']}")
    print(f"Top keywords: {result['evidence']['top_keywords']}")
    if result['evidence'].get('top_topics'):
        print(f"Top topics: {result['evidence']['top_topics']}")

    agent.save("models/text_agent.pkl")
    print("\nSaved → models/text_agent.pkl")


if __name__ == "__main__":
    main()
