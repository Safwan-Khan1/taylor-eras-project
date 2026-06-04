import pandas as pd
from pathlib import Path

from debate_orchestrator.orchestrator import run
from debate_orchestrator.dashboard import print_dashboard

from src.text_agent import TextAgent
from src.audio_agent import AudioAgent


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    BASE_DIR /
    "TaylorSwiftEras" /
    "Data" /
    "Final"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

lyrics_df = pd.read_csv(
    DATA_DIR / "taylor_lyric_df.csv"
)

audio_df = pd.read_csv(
    DATA_DIR / "taylor_audio_df.csv"
)

# --------------------------------------------------
# MAP ALBUM -> ERA
# --------------------------------------------------

def assign_era(album_name):

    album_name = str(album_name).lower()

    if "1989" in album_name:
        return "1989"

    if "reputation" in album_name:
        return "Reputation"

    if "lover" in album_name:
        return "Lover"

    if "folklore" in album_name:
        return "Folklore"

    if "evermore" in album_name:
        return "Evermore"

    if "midnights" in album_name:
        return "Midnights"

    if "fearless" in album_name:
        return "Fearless"

    if "speak now" in album_name:
        return "Speak Now"

    if "red" in album_name:
        return "Red"

    if "taylor swift" in album_name:
        return "Taylor Swift"

    return None


lyrics_df["era"] = lyrics_df["album_name"].apply(assign_era)
audio_df["era"] = audio_df["album_name"].apply(assign_era)

lyrics_df = lyrics_df.dropna(subset=["era"])
audio_df = audio_df.dropna(subset=["era"])

print("Training samples:", len(lyrics_df))

# --------------------------------------------------
# TRAIN AGENTS
# --------------------------------------------------

text_agent = TextAgent()
audio_agent = AudioAgent()

print("Training TextAgent...")
text_agent.fit(
    lyrics_df["lyric"],
    lyrics_df["era"]
)

print("Training AudioAgent...")
audio_agent.fit(
    audio_df,
    audio_df["era"]
)

# --------------------------------------------------
# SAMPLE INPUT
# --------------------------------------------------

sample_lyrics = (
    "We are never ever getting back together"
)

sample_audio = {
    "danceability": 0.65,
    "energy": 0.80,
    "key": 1,
    "loudness": -5.0,
    "mode": 1,
    "speechiness": 0.05,
    "acousticness": 0.10,
    "instrumentalness": 0.0,
    "liveness": 0.12,
    "valence": 0.55,
    "tempo": 128,
    "duration_ms": 210000
}

# --------------------------------------------------
# RUN DEBATE
# --------------------------------------------------

result = run(
    lyrics=sample_lyrics,
    audio_features=sample_audio,
    text_agent=text_agent,
    audio_agent=audio_agent
)

# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print_dashboard(result)