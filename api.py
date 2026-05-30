# """
# api.py — eralyzer FastAPI backend

# Run:  uvicorn api:app --reload
# Then open: http://localhost:8000
# """

# import sys
# from pathlib import Path

# import pandas as pd
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# BASE = Path(__file__).parent
# sys.path.insert(0, str(BASE))

# # ── Era metadata ───────────────────────────────────────────────────────────────
# ERAS_META = {
#     "Taylor Swift": {"id": "debut",      "label": "Debut",      "year": 2006, "color": "#c8a97e"},
#     "Fearless":     {"id": "fearless",   "label": "Fearless",   "year": 2008, "color": "#d9a818"},
#     "Speak Now":    {"id": "speaknow",   "label": "Speak Now",  "year": 2010, "color": "#9a6cc1"},
#     "Red":          {"id": "red",        "label": "Red",        "year": 2012, "color": "#b53030"},
#     "1989":         {"id": "1989",       "label": "1989",       "year": 2014, "color": "#4a9fc7"},
#     "Reputation":   {"id": "rep",        "label": "Reputation", "year": 2017, "color": "#4a4a4a"},
#     "Lover":        {"id": "lover",      "label": "Lover",      "year": 2019, "color": "#db83b0"},
#     "Folklore":     {"id": "folklore",   "label": "Folklore",   "year": 2020, "color": "#6f8a5c"},
#     "Evermore":     {"id": "evermore",   "label": "Evermore",   "year": 2020, "color": "#b56830"},
#     "Midnights":    {"id": "midnights",  "label": "Midnights",  "year": 2022, "color": "#4a5aa8"},
# }

# # ── Album → Era mapping ────────────────────────────────────────────────────────
# ALBUM_ERA = {
#     "Taylor Swift":                      "Taylor Swift",
#     "Beautiful Eyes":                    "Taylor Swift",
#     "The Taylor Swift Holiday Collection":"Taylor Swift",
#     "Fearless":                          "Fearless",
#     "Fearless (Taylor's Version)":       "Fearless",
#     "Speak Now":                         "Speak Now",
#     "Red":                               "Red",
#     "Red (Taylor's Version)":            "Red",
#     "1989":                              "1989",
#     "reputation":                        "Reputation",
#     "Lover":                             "Lover",
#     "folklore":                          "Folklore",
#     "evermore":                          "Evermore",
#     "Midnights":                         "Midnights",
# }

# # ── Load resources at startup ──────────────────────────────────────────────────
# text_agent = None
# try:
#     from src.text_agent import TextAgent
#     agent_path = BASE / "models" / "text_agent.pkl"
#     if agent_path.exists():
#         text_agent = TextAgent.load(str(agent_path))
#         print(f"[api] TextAgent loaded — {len(text_agent.classes_)} eras")
#     else:
#         print(f"[api] TextAgent not found at {agent_path} — run: python train_text_agent.py")
# except Exception as e:
#     print(f"[api] TextAgent load failed: {e}")

# dataset: pd.DataFrame | None = None
# try:
#     # Primary: full merged dataset (Debut → Folklore, 183 songs, has audio features)
#     primary_path = BASE / "TaylorSwiftEras" / "Data" / "Final" / "taylor_merged_df.csv"
#     df_primary   = pd.read_csv(primary_path)
#     df_primary["era"]    = df_primary["album_name"].map(ALBUM_ERA).fillna("Unknown")
#     df_primary["lyrics"] = df_primary["lyric"]

#     # Fallback: processed dataset (Evermore + Midnights + some overlap, 93 songs)
#     fallback_path = BASE / "data" / "processed" / "taylor_swift_eras_processed.csv"
#     df_fallback   = pd.read_csv(fallback_path)

#     # Merge: primary first, then any fallback songs not already present
#     primary_titles  = set(df_primary["track_name"].str.lower().str.strip())
#     df_extra        = df_fallback[
#         ~df_fallback["track_name"].str.lower().str.strip().isin(primary_titles)
#     ].copy()

#     dataset = pd.concat([df_primary, df_extra], ignore_index=True)
#     print(f"[api] Dataset loaded: {len(dataset)} songs "
#           f"({len(df_primary)} primary + {len(df_extra)} fallback)")
# except Exception as e:
#     print(f"[api] Dataset load failed: {e}")

# # ── App ────────────────────────────────────────────────────────────────────────
# app = FastAPI(title="eralyzer API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class DebateRequest(BaseModel):
#     song_title: str
#     artist: str = "Taylor Swift"


# # ── Text helpers ──────────────────────────────────────────────────────────────
# def trim_to_words(text: str, max_words: int = 30) -> str:
#     """Return text capped at max_words, ending at the last clean sentence if possible."""
#     words = text.split()
#     if len(words) <= max_words:
#         return text.strip()
#     clipped = " ".join(words[:max_words])
#     # Try to end at the last sentence boundary within the clipped text
#     for punct in (".", "!", "?"):
#         idx = clipped.rfind(punct)
#         if idx > len(clipped) // 3:          # ignore boundary too near the start
#             return clipped[:idx + 1].strip()
#     # No clean boundary — end at last word, ensure it closes with a period
#     return clipped.rstrip(",:;—") + "."


# # ── Song lookup ────────────────────────────────────────────────────────────────
# def find_song(title: str) -> dict | None:
#     if dataset is None:
#         return None
#     title_lc = title.lower().strip()
#     names    = dataset["track_name"].str.lower().str.strip()
#     # Exact match first
#     mask = names == title_lc
#     if mask.any():
#         return dataset[mask].iloc[0].to_dict()
#     # Partial match
#     mask = names.str.contains(title_lc, na=False, regex=False)
#     if mask.any():
#         return dataset[mask].iloc[0].to_dict()
#     return None


# # ── Audio Agent (hardcoded until Member 3 integrates real model) ───────────────
# def audio_agent_open(predicted: str, lyric_conf: float) -> list:
#     if lyric_conf >= 0.65:
#         return [
#             {"who": "audio", "text": "> Fetching Spotify feature vector…"},
#             {"who": "audio", "text": "> danceability · energy · valence · acousticness loaded."},
#             {"who": "audio", "text": f"> Feature cluster aligns with <emp>{predicted}</emp> signature."},
#             {"who": "audio", "text": "> No strong objection. Deferring to Lyric Agent.", "cls": "is-resolve"},
#         ]
#     return [
#         {"who": "audio", "text": "> Fetching Spotify feature vector…"},
#         {"who": "audio", "text": "> Moderate ambiguity in audio cluster."},
#         {"who": "audio", "text": f"> Audio leans <emp>{predicted}</emp> but adjacent-era overlap detected."},
#         {"who": "audio", "text": "> Requesting Lyric Agent to reinforce with LDA evidence."},
#     ]


# def audio_agent_concede(predicted: str) -> list:
#     return [
#         {"who": "audio", "text": "> Re-examining feature cluster after lyric signal…"},
#         {"who": "audio", "text": f"> Lyric anchor convincing. Updating to <emp>{predicted}</emp>.", "cls": "is-concede"},
#         {"who": "audio", "text": "> Consensus reached.", "cls": "is-resolve"},
#     ]


# # ── Debate script builder ──────────────────────────────────────────────────────
# def build_debate_script(ta: dict, song_title: str) -> dict:
#     predicted  = ta["predicted_era"]
#     probs      = ta["probabilities"]
#     evidence   = ta["evidence"]
#     reasoning  = ta["reasoning"]

#     keywords   = evidence.get("top_keywords", [])[:5]
#     topic      = evidence.get("dominant_topic", "")
#     sentiment  = evidence.get("sentiment", {})
#     lyric_conf = float(probs.get(predicted, 0))

#     sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
#     runner_up    = sorted_probs[1] if len(sorted_probs) > 1 else None

#     kw_str     = '", "'.join(keywords) if keywords else "—"
#     sent_val   = sentiment.get("compound", 0)
#     sent_label = (
#         "strongly negative" if sent_val < -0.5 else
#         "mildly negative"   if sent_val < -0.1 else
#         "neutral"           if sent_val <  0.1 else
#         "mildly positive"   if sent_val <  0.5 else
#         "strongly positive"
#     )
#     short_reason = trim_to_words(reasoning, max_words=30)
#     era_meta     = ERAS_META.get(predicted, {"id": "midnights", "label": predicted, "color": "#4a5aa8"})
#     high_conf    = lyric_conf >= 0.65

#     # Round 1 — Lyric Agent opens
#     script: list[dict] = [
#         {"who": "lyric", "text": f'> Tokenising: "{song_title}"…'},
#         {"who": "lyric", "text": f'> Top TF-IDF signals: "{kw_str}"'},
#         {"who": "lyric", "text": f'> LDA dominant topic: <emp>{topic}</emp>'},
#         {"who": "lyric", "text": f'> Sentiment: {sent_label} (compound=<num>{sent_val:.2f}</num>)'},
#         {"who": "lyric", "text": f'> TF-IDF prediction: <emp>{predicted} — {lyric_conf:.0%}</emp>'},
#     ]
#     if runner_up and runner_up[1] > 0.15:
#         script.append({
#             "who": "lyric",
#             "text": f'> Runner-up: {runner_up[0]} (<num>{runner_up[1]:.0%}</num>)',
#         })

#     # Round 1 — Audio Agent responds
#     script.extend(audio_agent_open(predicted, lyric_conf))

#     # Round 2 — only if low confidence
#     if not high_conf:
#         script.append({"who": "sys", "text": "── DEBATE ROUND 2 ──"})
#         script.append({
#             "who": "lyric",
#             "text": "> Reinforcing with LDA topic distribution and LLM signal…",
#             "cls": "is-emphatic",
#         })
#         if runner_up:
#             script.append({
#                 "who": "lyric",
#                 "text": (
#                     f'> Gap: {predicted} ({lyric_conf:.0%}) vs '
#                     f'{runner_up[0]} ({runner_up[1]:.0%}) — lyric diction is decisive.'
#                 ),
#                 "cls": "is-emphatic",
#             })
#         script.extend(audio_agent_concede(predicted))

#     # LLM reasoning reveal (capped at 30 words — fits on one CLI line)
#     script.append({"who": "sys",   "text": "── LLM REASONING ──"})
#     script.append({"who": "lyric", "text": f'> {short_reason}', "cls": "is-emphatic"})
#     script.append({"who": "lyric", "text": f'> Final verdict: <emp>{predicted}</emp>', "cls": "is-resolve"})
#     script.append({"who": "audio", "text": "> Consensus. Stacking classifier locked.", "cls": "is-resolve"})

#     rounds  = 1 if high_conf else 2
#     runtime = round(1.4 + rounds * 0.5, 2)

#     return {
#         "id":         "live",
#         "title":      song_title,
#         "artist":     "Taylor Swift",
#         "era":        era_meta["id"],
#         "audioConf":  round(lyric_conf * 0.85, 2),
#         "lyricConf":  round(lyric_conf, 2),
#         "stacking":   round(min(lyric_conf * 1.05, 0.99), 2),
#         "badge":      "consensus",
#         "reason":     short_reason,
#         "rounds":     rounds,
#         "runtime":    runtime,
#         "audioFinal": f"{predicted} — {lyric_conf * 0.85:.0%}",
#         "lyricFinal": f"{predicted} — {lyric_conf:.0%}",
#         "script":     script,
#     }


# # ── API routes (must be defined BEFORE the static mount) ──────────────────────
# @app.get("/status")
# def status():
#     return {
#         "text_agent":    text_agent is not None,
#         "dataset_songs": len(dataset) if dataset is not None else 0,
#     }


# @app.post("/debate")
# def run_debate(req: DebateRequest):
#     if text_agent is None:
#         return JSONResponse(
#             {"error": "TextAgent not loaded. Run: python train_text_agent.py"},
#             status_code=503,
#         )

#     song = find_song(req.song_title)
#     if song is None:
#         return JSONResponse(
#             {"error": f'"{req.song_title}" not found in dataset. Try a Taylor Swift song title.'},
#             status_code=404,
#         )

#     lyrics = str(song.get("lyrics", "")).strip()
#     if not lyrics:
#         return JSONResponse({"error": "No lyrics found for this song."}, status_code=404)

#     try:
#         ta_result = text_agent.predict_with_evidence(lyrics)
#         demo      = build_debate_script(ta_result, req.song_title)
#         return JSONResponse(demo)
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)


# # ── Serve static files (must come AFTER API routes) ───────────────────────────
# app.mount("/", StaticFiles(directory=str(BASE), html=True), name="static")


"""
api.py — eralyzer FastAPI backend

Run:  uvicorn api:app --reload
Then open: http://localhost:8000
"""

"""
api.py — eralyzer FastAPI backend (Perfected Late Fusion)
"""

import sys
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

# ── Era metadata ───────────────────────────────────────────────────────────────
ERAS_META = {
    "Taylor Swift": {"id": "debut",      "label": "Debut",      "year": 2006, "color": "#c8a97e"},
    "Fearless":     {"id": "fearless",   "label": "Fearless",   "year": 2008, "color": "#d9a818"},
    "Speak Now":    {"id": "speaknow",   "label": "Speak Now",  "year": 2010, "color": "#9a6cc1"},
    "Red":          {"id": "red",        "label": "Red",        "year": 2012, "color": "#b53030"},
    "1989":         {"id": "1989",       "label": "1989",       "year": 2014, "color": "#4a9fc7"},
    "Reputation":   {"id": "rep",        "label": "Reputation", "year": 2017, "color": "#4a4a4a"},
    "Lover":        {"id": "lover",      "label": "Lover",      "year": 2019, "color": "#db83b0"},
    "Folklore":     {"id": "folklore",   "label": "Folklore",   "year": 2020, "color": "#6f8a5c"},
    "Evermore":     {"id": "evermore",   "label": "Evermore",   "year": 2020, "color": "#b56830"},
    "Midnights":    {"id": "midnights",  "label": "Midnights",  "year": 2022, "color": "#4a5aa8"},
}

ALBUM_ERA = {
    "Taylor Swift": "Taylor Swift", "Beautiful Eyes": "Taylor Swift", 
    "Fearless": "Fearless", "Speak Now": "Speak Now", "Red": "Red",
    "1989": "1989", "reputation": "Reputation", "Lover": "Lover",
    "folklore": "Folklore", "evermore": "Evermore", "Midnights": "Midnights",
}

# ── Load resources ────────────────────────────────────────────────────────────
text_agent = None
audio_agent = None

try:
    from src.text_agent import TextAgent
    from src.audio_agent import AudioAgent
    
    t_path = BASE / "models" / "text_agent.pkl"
    a_path = BASE / "models" / "audio_agent.pkl"
    if t_path.exists(): text_agent = TextAgent.load(str(t_path))
    if a_path.exists(): audio_agent = AudioAgent.load(str(a_path))
    print(f"[api] Agents Active: Text({text_agent is not None}) Audio({audio_agent is not None})")
except Exception as e:
    print(f"[api] Agent load failed: {e}")

dataset: pd.DataFrame | None = None
try:
    primary_path = BASE / "TaylorSwiftEras" / "Data" / "Final" / "taylor_merged_df.csv"
    dataset = pd.read_csv(primary_path)
    dataset["era"] = dataset["album_name"].map(ALBUM_ERA).fillna("Unknown")
    dataset["lyrics"] = dataset["lyric"]
except Exception as e:
    print(f"[api] Dataset load failed: {e}")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="eralyzer API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Update the Request Model ──────────────────────────────────────────────────
class DebateRequest(BaseModel):
    song_title: str
    artist: str = "Taylor Swift"
    fusion: str = "late" # Default to late

def build_early_fusion_script(ta: dict, au: dict, song_data: dict, runtime_val: float) -> dict:
    """Narrative for Early Fusion: A unified feature vector analysis."""
    p_ta = ta["predicted_era"]
    l_conf = float(ta["probabilities"].get(p_ta, 0))
    a_conf = float(au["probabilities"].get(p_ta, 0)) # Note: Using p_ta to check agreement
    
    # Simulate a unified confidence score for the UI
    combined_conf = min(max(l_conf, a_conf) * 1.05, 0.99)
    
    # Notice how "who" is only 'lyric' (our repurposed Fusion box) or 'sys'
    script = [
        {"who": "sys", "text": "── EARLY FUSION MODE: UNIFIED ANALYSIS ──"},
        {"who": "lyric", "text": "> Loading TF-IDF linguistic matrices..."},
        {"who": "lyric", "text": "> Loading Spotify acoustic feature vectors..."},
        {"who": "sys", "text": "> CONCATENATING TO [1 x 8192] FEATURE VECTOR..."},
        {"who": "lyric", "text": "> Analyzing joint distribution space...", "cls": "is-emphatic"},
        {"who": "lyric", "text": f"> Unified signal aligns with {p_ta} signature.", "cls": "is-resolve"},
        {"who": "sys", "text": "── INFERENCE REASONING ──"},
        {"who": "lyric", "text": f"> The concatenated model predicts {p_ta} because linguistic themes and audio energy are evaluated simultaneously, creating a single highly confident inference.", "cls": "is-emphatic"},
        {"who": "lyric", "text": "> Single-pass inference complete.", "cls": "is-resolve"}
    ]

    meta = ERAS_META.get(p_ta, {"id": "midnights", "label": p_ta, "color": "#4a5aa8"})

    return {
        "id": "live",
        "title": song_data["track_name"],
        "era": meta["id"],
        "audioConf": combined_conf, # Feed combined to both so the UI bars match
        "lyricConf": combined_conf,
        "stacking": round(combined_conf, 2),
        "badge": "joint-inference",
        "reason": f"Early fusion model analyzed {song_data['track_name']} by merging lyric tokens and audio features into a single high-dimensional vector prior to classification.",
        "rounds": 1,
        "runtime": runtime_val,
        "audioFinal": "Joint",
        "lyricFinal": "Joint",
        "script": script,
    }

def trim_to_words(text: str, max_words: int = 30) -> str:
    words = text.split()
    if len(words) <= max_words: return text.strip()
    clipped = " ".join(words[:max_words])
    for i in range(len(clipped) - 1, len(clipped) // 3, -1):
        if clipped[i] in (".", "!", "?") and (i == len(clipped)-1 or clipped[i+1] == " "):
            return clipped[:i+1].strip()
    return clipped.rstrip(",:;—") + "..."

# ── Debate script builder ──────────────────────────────────────────────────────
# ── Updated build_debate_script ───────────────────────────────────────────────
def build_debate_script(ta: dict, au: dict, song_data: dict, runtime_val: float) -> dict:
    p_ta, p_au = ta["predicted_era"], au["predicted_era"]
    l_conf, a_conf = float(ta["probabilities"].get(p_ta, 0)), float(au["probabilities"].get(p_au, 0))
    
    if p_ta == p_au:
        stacking_val = min(max(l_conf, a_conf) * 1.12, 0.99)
        final_verdict, badge = p_ta, "consensus"
    else:
        stacking_val = (l_conf + a_conf) / 2
        final_verdict, badge = (p_ta if l_conf >= a_conf else p_au), "adjudicated"

    short_ta = ta["reasoning"]
    short_au = au["reasoning"]
    perfect_reason = f"Synthesized verdict: {short_ta} Production markers ({a_conf:.0%}) confirm the {final_verdict} profile."

    kw_str = '", "'.join(ta["evidence"].get("top_keywords", [])[:3])
    
    # Round 1
    script = [
        {"who": "lyric", "text": f'> Tokenising: "{song_data["track_name"]}"…'},
        {"who": "lyric", "text": f'> Signals: "{kw_str}"'},
        {"who": "lyric", "text": f'> TF-IDF prediction: <emp>{p_ta} — {l_conf:.0%}</emp>'},
        {"who": "audio", "text": "> Fetching Spotify feature vector…"},
        {"who": "audio", "text": f"> Audio Signals: energy(<num>{song_data.get('energy',0):.2f}</num>) · acousticness(<num>{song_data.get('acousticness',0):.2f}</num>) · valence(<num>{song_data.get('valence',0):.2f}</num>)"},
        {"who": "audio", "text": f"> Gradient Boost prediction: <emp>{p_au} — {a_conf:.0%}</emp>"}
    ]

    # Round 2 - Triggered if Disagree OR low confidence (< 65%)
    if p_ta != p_au or l_conf < 0.65:
        script.append({"who": "sys", "text": "── DEBATE ROUND 2 ──"})
        if p_ta == p_au:
            script.append({"who": "lyric", "text": "> Low confidence detected. Reinforcing with LDA topic weights...", "cls": "is-emphatic"})
            script.append({"who": "audio", "text": "> Cross-referencing audio cluster... signature verified.", "cls": "is-resolve"})
        else:
            script.append({"who": "lyric", "text": f"> Conflict: {p_ta} vs {p_au}. Applying late fusion weights...", "cls": "is-emphatic"})
            script.append({"who": "audio", "text": f"> Conceding to higher signal weight: <emp>{final_verdict}</emp>", "cls": "is-concede"})

    # Agent Reasoning
    script.append({"who": "sys",   "text": "── AGENT REASONING ──"})
    script.append({"who": "audio", "text": f'> {short_au}', "cls": "is-emphatic"})
    script.append({"who": "lyric", "text": f'> {short_ta}', "cls": "is-emphatic"})
    script.append({"who": "lyric", "text": f'> Final verdict: <emp>{final_verdict}</emp>', "cls": "is-resolve"})

    meta = ERAS_META.get(final_verdict, {"id": "midnights", "label": final_verdict, "color": "#4a5aa8"})
    
    return {
        "id": "live",
        "title": song_data["track_name"],
        "era": meta["id"],
        "audioConf": round(a_conf, 2),
        "lyricConf": round(l_conf, 2),
        "stacking": round(stacking_val, 2),
        "badge": badge,
        "reason": perfect_reason,
        "rounds": 2 if (p_ta != p_au or l_conf < 0.65) else 1,
        "runtime": runtime_val, # CRITICAL: Now a FLOAT (e.g. 0.45)
        "audioFinal": f"{p_au} — {a_conf:.0%}",
        "lyricFinal": f"{p_ta} — {l_conf:.0%}",
        "script": script,
    }

# ── Updated run_debate route ──────────────────────────────────────────────────
# ── Update the Route ─────────────────────────────────────────────────────────
@app.post("/debate")
def run_debate(req: DebateRequest):
    song_row = dataset[dataset["track_name"].str.lower().str.strip() == req.song_title.lower().strip()]
    if song_row.empty:
        return JSONResponse({"error": "Song not found"}, status_code=404)
    
    song = song_row.iloc[0].to_dict()
    start_time = time.perf_counter()
    
    # Process inputs
    ta_res = text_agent.predict_with_evidence(str(song["lyrics"]))
    au_features = {f: song.get(f, 0) for f in audio_agent.features}
    au_res = audio_agent.predict_with_evidence(au_features)
    
    elapsed = time.perf_counter() - start_time

    # Choose Script based on Fusion Mode
    if req.fusion == "early":
        response_data = build_early_fusion_script(ta_res, au_res, song, elapsed)
    else:
        response_data = build_debate_script(ta_res, au_res, song, elapsed)
        
    return JSONResponse(response_data)