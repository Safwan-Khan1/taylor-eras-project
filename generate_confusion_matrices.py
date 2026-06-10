"""
Quick script to regenerate confusion matrix PNGs from the saved models
without retraining. Uses the same test split (random_state=42) as the
training scripts so results are directly comparable.
"""
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(__file__).parent

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
TV_ALBUMS   = {"Fearless (Taylor's Version)", "Red (Taylor's Version)"}
ERA_ORDER_8 = ["Taylor Swift", "Fearless", "Speak Now", "Red", "1989", "Reputation", "Lover", "Folklore"]
CSV_PATH    = BASE / "TaylorSwiftEras" / "Data" / "Final" / "taylor_merged_df.csv"


def load_csv():
    df = pd.read_csv(CSV_PATH)
    original_titles = set(
        df.loc[~df["album_name"].isin(TV_ALBUMS), "track_name"].str.lower().str.strip()
    )
    df = df[
        ~df["album_name"].isin(TV_ALBUMS) |
        ~df["track_name"].str.lower().str.strip().isin(original_titles)
    ].copy()
    df["era"] = df["album_name"].map(ALBUM_ERA)
    return df


def save_cm_png(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=ERA_ORDER_8)
    disp = ConfusionMatrixDisplay(cm, display_labels=ERA_ORDER_8)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved → {path}")


# ── Text Agent ────────────────────────────────────────────────────────────────
print("=== Text Agent ===")
from src.text_agent import TextAgent

df = load_csv()
df["lyrics"] = df["lyric"]
df = df.dropna(subset=["era", "lyrics"])
df = df[df["lyrics"].str.strip() != ""].reset_index(drop=True)

_, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["era"])
print(f"Test set: {len(df_test)} songs")

agent = TextAgent.load("models/text_agent.pkl")

y_true, y_pred = [], []
for _, row in df_test.iterrows():
    result = agent.predict_with_evidence(row["lyrics"], use_llm=False)
    y_true.append(row["era"])
    y_pred.append(result["predicted_era"])

print(classification_report(y_true, y_pred, zero_division=0))
save_cm_png(y_true, y_pred,
            "Text Agent — Confusion Matrix (Test Set)",
            "models/text_agent_confusion_matrix.png")


# ── Audio Agent ───────────────────────────────────────────────────────────────
print("\n=== Audio Agent ===")
from src.audio_agent import AudioAgent

df2 = load_csv()
df2 = df2.dropna(subset=["era"]).reset_index(drop=True)

_, df2_test = train_test_split(df2, test_size=0.2, random_state=42, stratify=df2["era"])
print(f"Test set: {len(df2_test)} songs")

audio_agent = AudioAgent.load("models/audio_agent.pkl")

y_true2, y_pred2 = [], []
for _, row in df2_test.iterrows():
    result = audio_agent.predict_with_evidence(row.to_dict(), use_llm=False)
    y_true2.append(row["era"])
    y_pred2.append(result["predicted_era"])

print(classification_report(y_true2, y_pred2, zero_division=0))
save_cm_png(y_true2, y_pred2,
            "Audio Agent — Confusion Matrix (Test Set)",
            "models/audio_agent_confusion_matrix.png")

print("\nDone.")
