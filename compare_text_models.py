"""
Compare Logistic Regression vs Random Forest for the TextAgent classifier.

Saves:
  models/text_agent_lr_test.pkl   — LR version
  models/text_agent_rf_test.pkl   — RF version

Run: python compare_text_models.py

The best model can then be used in models/text_agent.pkl.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.text_agent import TextAgent

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent

MERGED_CSV = BASE / "TaylorSwiftEras" / "Data" / "Final" / "taylor_merged_df.csv"

ALBUM_ERA = {
    "Taylor Swift":                        "Taylor Swift",
    "Beautiful Eyes":                      "Taylor Swift",
    "The Taylor Swift Holiday Collection": "Taylor Swift",
    "Fearless":                            "Fearless",
    "Fearless (Taylor's Version)":         "Fearless",
    "Speak Now":                           "Speak Now",
    "Red":                                 "Red",
    "Red (Taylor's Version)":              "Red",
    "1989":                                "1989",
    "reputation":                          "Reputation",
    "Lover":                               "Lover",
    "folklore":                            "Folklore",
}

TAYLORS_VERSION_ALBUMS = {"Fearless (Taylor's Version)", "Red (Taylor's Version)"}

ERA_ORDER_8 = [
    "Taylor Swift", "Fearless", "Speak Now", "Red",
    "1989", "Reputation", "Lover", "Folklore"
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(MERGED_CSV)
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


def make_lr_agent():
    return TextAgent(
        classifier=LogisticRegression(
            C=1.5,
            class_weight='balanced',
            max_iter=2000,
            random_state=42,
            solver='lbfgs',
        )
    )


def make_rf_agent():
    return TextAgent(
        classifier=RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
    )


def cross_validate(make_agent_fn, lyrics, eras, label):
    print(f"\n--- 5-Fold CV: {label} ---")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(lyrics, eras)):
        agent = make_agent_fn()
        agent.fit(
            [lyrics[i] for i in train_idx],
            [eras[i]   for i in train_idx],
        )
        preds = [
            agent.predict_with_evidence(lyrics[i], use_llm=False)["predicted_era"]
            for i in val_idx
        ]
        score = accuracy_score([eras[i] for i in val_idx], preds)
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.3f}")
    mean, std = np.mean(scores), np.std(scores)
    print(f"  CV Mean: {mean:.3f} ± {std:.3f}")
    return mean, std


def eval_on_test(agent, df_test, label):
    y_true, y_pred = [], []
    for _, row in df_test.iterrows():
        result = agent.predict_with_evidence(row["lyrics"], use_llm=False)
        y_true.append(row["era"])
        y_pred.append(result["predicted_era"])
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== {label} — Test Set ===")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    return acc, y_true, y_pred


def save_confusion_matrix(y_true, y_pred, label, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=ERA_ORDER_8)
    disp = ConfusionMatrixDisplay(cm, display_labels=ERA_ORDER_8)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(f"{label} — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")


def main():
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset: {len(df)} songs, {df['era'].nunique()} eras")
    print(df["era"].value_counts().to_string(), "\n")

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["era"])
    print(f"Train: {len(df_train)} | Test: {len(df_test)}\n")

    lyrics_all = df["lyrics"].tolist()
    eras_all   = df["era"].tolist()

    # ── Cross-validation ───────────────────────────────────────────────
    lr_cv_mean, lr_cv_std = cross_validate(make_lr_agent, lyrics_all, eras_all, "Logistic Regression")
    rf_cv_mean, rf_cv_std = cross_validate(make_rf_agent, lyrics_all, eras_all, "Random Forest")

    # ── Full train on train split ──────────────────────────────────────
    print("\nTraining final LR model on train split...")
    lr_agent = make_lr_agent()
    lr_agent.fit(df_train["lyrics"].tolist(), df_train["era"].tolist())

    print("Training final RF model on train split (this takes longer)...")
    rf_agent = make_rf_agent()
    rf_agent.fit(df_train["lyrics"].tolist(), df_train["era"].tolist())

    # ── Test set evaluation ────────────────────────────────────────────
    lr_acc, lr_true, lr_pred = eval_on_test(lr_agent, df_test, "Logistic Regression")
    rf_acc, rf_true, rf_pred = eval_on_test(rf_agent, df_test, "Random Forest")

    # ── Confusion matrices ─────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    save_confusion_matrix(lr_true, lr_pred, "Logistic Regression", "models/text_agent_lr_test_cm.png")
    save_confusion_matrix(rf_true, rf_pred, "Random Forest",       "models/text_agent_rf_test_cm.png")

    # ── Save models ────────────────────────────────────────────────────
    lr_agent.save("models/text_agent_lr_test.pkl")
    rf_agent.save("models/text_agent_rf_test.pkl")
    print("\nSaved → models/text_agent_lr_test.pkl")
    print("Saved → models/text_agent_rf_test.pkl")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    print(f"{'Model':<22} {'CV Acc':>8}  {'Test Acc':>10}")
    print(f"{'-'*22} {'-'*8}  {'-'*10}")
    print(f"{'Logistic Regression':<22} {lr_cv_mean:.3f}±{lr_cv_std:.3f}  {lr_acc:.3f}")
    print(f"{'Random Forest':<22} {rf_cv_mean:.3f}±{rf_cv_std:.3f}  {rf_acc:.3f}")
    print()

    winner = "Logistic Regression" if lr_acc >= rf_acc else "Random Forest"
    winner_path = "models/text_agent_lr_test.pkl" if lr_acc >= rf_acc else "models/text_agent_rf_test.pkl"
    diff = abs(lr_acc - rf_acc)
    print(f"Winner: {winner} (+{diff:.3f} test accuracy)")
    print(f"\nTo promote the winner:  copy {winner_path} → models/text_agent.pkl")
    print("To clean up test files: del models/text_agent_lr_test.pkl models/text_agent_rf_test.pkl")
    print("                        del models/text_agent_lr_test_cm.png models/text_agent_rf_test_cm.png")


if __name__ == "__main__":
    main()
