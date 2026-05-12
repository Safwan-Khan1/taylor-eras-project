import pandas as pd
import numpy as np
import json, sys, io

results = {}

# ── Dataset 1 ────────────────────────────────────────────────────────────────
df1 = pd.read_csv("data/raw/taylor_eras_multimodal_dataset.csv")
r1 = {}
r1["shape"] = list(df1.shape)
r1["columns"] = list(df1.columns)
r1["dtypes"] = {c: str(t) for c, t in df1.dtypes.items()}
r1["null_counts"] = df1.isnull().sum().to_dict()
r1["null_pct"] = (df1.isnull().sum()/len(df1)*100).round(2).to_dict()
# era / label distributions
for col in df1.columns:
    if col.lower() in ["era", "album", "label", "class", "target"]:
        r1[f"label_col_{col}"] = df1[col].value_counts().to_dict()
# text & audio feature lists
r1["text_cols"] = [c for c in df1.columns if any(k in c.lower() for k in ["lyric","text","word","token","clean"])]
r1["audio_cols"] = [c for c in df1.columns if any(k in c.lower() for k in ["tempo","energy","valence","dance","acoustic","speech","loud","pitch","key","mode","duration","beat","danceability","liveness","instrumentalness"])]
r1["sample_row0"] = df1.iloc[0].to_dict() if len(df1) > 0 else {}
# numeric describe serializable
desc1 = df1.describe().to_dict()
r1["describe"] = {col: {k: (float(v) if not np.isnan(v) else None) for k, v in vals.items()} for col, vals in desc1.items()}
results["df1"] = r1

# ── Dataset 2 ────────────────────────────────────────────────────────────────
df2 = pd.read_csv("data/raw/taylor_lyric_df_processed.csv")
r2 = {}
r2["shape"] = list(df2.shape)
r2["columns"] = list(df2.columns)
r2["dtypes"] = {c: str(t) for c, t in df2.dtypes.items()}
r2["null_counts"] = df2.isnull().sum().to_dict()
r2["null_pct"] = (df2.isnull().sum()/len(df2)*100).round(2).to_dict()
for col in df2.columns:
    if col.lower() in ["era", "album", "label", "class", "target"]:
        r2[f"label_col_{col}"] = df2[col].value_counts().to_dict()
r2["text_cols"] = [c for c in df2.columns if any(k in c.lower() for k in ["lyric","text","word","token","clean"])]
r2["audio_cols"] = [c for c in df2.columns if any(k in c.lower() for k in ["tempo","energy","valence","dance","acoustic","speech","loud","pitch","key","mode","duration","beat","danceability","liveness","instrumentalness"])]
r2["sample_row0"] = df2.iloc[0].to_dict() if len(df2) > 0 else {}
desc2 = df2.describe().to_dict()
r2["describe"] = {col: {k: (float(v) if not np.isnan(v) else None) for k, v in vals.items()} for col, vals in desc2.items()}
results["df2"] = r2

# ── Overlap ──────────────────────────────────────────────────────────────────
ov = {}
ov["common_cols"] = sorted(set(df1.columns) & set(df2.columns))
ov["only_df1"] = sorted(set(df1.columns) - set(df2.columns))
ov["only_df2"] = sorted(set(df2.columns) - set(df1.columns))
for col in ["track_name","song","title","name","track","song_title"]:
    if col in df1.columns and col in df2.columns:
        songs1 = set(df1[col].dropna().str.lower())
        songs2 = set(df2[col].dropna().str.lower())
        ov["song_overlap"] = {"column": col, "shared": len(songs1 & songs2), "df1_unique": len(songs1), "df2_unique": len(songs2)}
        break
results["overlap"] = ov

with open("analysis_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, default=str)

print("DONE - check analysis_result.json")
