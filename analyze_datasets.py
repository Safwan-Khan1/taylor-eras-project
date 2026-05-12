import pandas as pd
import numpy as np

# ── Dataset 1 ────────────────────────────────────────────────────────────────
print("=" * 70)
print("DATASET 1: taylor_eras_multimodal_dataset.csv")
print("=" * 70)
df1 = pd.read_csv("data/raw/taylor_eras_multimodal_dataset.csv")
print(f"Shape: {df1.shape}")
print(f"\nColumns ({len(df1.columns)}):\n  {list(df1.columns)}")
print(f"\nDtypes:\n{df1.dtypes.to_string()}")
print(f"\nNull counts:\n{df1.isnull().sum().to_string()}")
print(f"\nNull % per column:\n{(df1.isnull().sum()/len(df1)*100).round(2).to_string()}")

# label column detection
for col in df1.columns:
    if col.lower() in ["era", "album", "label", "class", "target"]:
        print(f"\n[TARGET] Column '{col}' distribution:\n{df1[col].value_counts().to_string()}")

print(f"\nFirst 3 rows:\n{df1.head(3).to_string()}")
print(f"\nNumeric describe:\n{df1.describe().to_string()}")

# Check for text / audio / lyric columns
text_cols = [c for c in df1.columns if any(k in c.lower() for k in ["lyric", "text", "word", "token"])]
audio_cols = [c for c in df1.columns if any(k in c.lower() for k in ["tempo", "energy", "valence", "dance", "acoustic", "speech", "loud", "pitch", "key", "mode", "duration", "beat"])]
print(f"\nText-like columns: {text_cols}")
print(f"Audio-like columns: {audio_cols}")

# ── Dataset 2 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DATASET 2: taylor_lyric_df_processed.csv")
print("=" * 70)
df2 = pd.read_csv("data/raw/taylor_lyric_df_processed.csv")
print(f"Shape: {df2.shape}")
print(f"\nColumns ({len(df2.columns)}):\n  {list(df2.columns)}")
print(f"\nDtypes:\n{df2.dtypes.to_string()}")
print(f"\nNull counts:\n{df2.isnull().sum().to_string()}")
print(f"\nNull % per column:\n{(df2.isnull().sum()/len(df2)*100).round(2).to_string()}")

for col in df2.columns:
    if col.lower() in ["era", "album", "label", "class", "target"]:
        print(f"\n[TARGET] Column '{col}' distribution:\n{df2[col].value_counts().to_string()}")

print(f"\nFirst 3 rows:\n{df2.head(3).to_string()}")
print(f"\nNumeric describe:\n{df2.describe().to_string()}")

text_cols2 = [c for c in df2.columns if any(k in c.lower() for k in ["lyric", "text", "word", "token"])]
audio_cols2 = [c for c in df2.columns if any(k in c.lower() for k in ["tempo", "energy", "valence", "dance", "acoustic", "speech", "loud", "pitch", "key", "mode", "duration", "beat"])]
print(f"\nText-like columns: {text_cols2}")
print(f"Audio-like columns: {audio_cols2}")

# ── Overlap / deduplication ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("OVERLAP / SHARED STRUCTURE")
print("=" * 70)
common_cols = set(df1.columns) & set(df2.columns)
print(f"Common columns: {sorted(common_cols)}")
print(f"Only in df1: {sorted(set(df1.columns) - set(df2.columns))}")
print(f"Only in df2: {sorted(set(df2.columns) - set(df1.columns))}")

# song name overlap if possible
for col in ["track_name", "song", "title", "name", "track"]:
    if col in df1.columns and col in df2.columns:
        songs1 = set(df1[col].dropna().str.lower())
        songs2 = set(df2[col].dropna().str.lower())
        print(f"\nSong overlap ('{col}'): {len(songs1 & songs2)} shared | df1={len(songs1)} | df2={len(songs2)}")
        break

print("\nDone.")
