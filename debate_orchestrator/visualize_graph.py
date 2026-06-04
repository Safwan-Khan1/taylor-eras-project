import matplotlib.pyplot as plt
from debate_orchestrator.orchestrator import build_graph
from src.text_agent import TextAgent
from src.audio_agent import AudioAgent
import pandas as pd
from pathlib import Path

def main():

    # -----------------------------
    # LOAD DATA (for training)
    # -----------------------------
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "TaylorSwiftEras" / "Data" / "Final"

    lyrics_df = pd.read_csv(DATA_DIR / "taylor_lyric_df.csv")
    audio_df = pd.read_csv(DATA_DIR / "taylor_audio_df.csv")

    # simple era mapping (same as test_run)
    def assign_era(name):
        name = str(name).lower()
        if "red" in name: return "Red"
        if "1989" in name: return "1989"
        if "lover" in name: return "Lover"
        if "fearless" in name: return "Fearless"
        if "speak now" in name: return "Speak Now"
        return None

    lyrics_df["era"] = lyrics_df["album_name"].apply(assign_era)
    audio_df["era"] = audio_df["album_name"].apply(assign_era)

    lyrics_df = lyrics_df.dropna(subset=["era"])
    audio_df = audio_df.dropna(subset=["era"])

    # -----------------------------
    # TRAIN REAL AGENTS
    # -----------------------------
    text_agent = TextAgent()
    audio_agent = AudioAgent()

    text_agent.fit(lyrics_df["lyric"], lyrics_df["era"])
    audio_agent.fit(audio_df, audio_df["era"])

    # -----------------------------
    # BUILD GRAPH WITH REAL MODELS
    # -----------------------------
    graph = build_graph(text_agent, audio_agent)

    # -----------------------------
    # EXPORT VISUALIZATION
    # -----------------------------
    try:
        png = graph.get_graph().draw_mermaid_png()

        with open("debate_graph.png", "wb") as f:
            f.write(png)

        print("✅ Graph saved as debate_graph.png")

    except Exception as e:
        print("Graph visualization failed:", e)


if __name__ == "__main__":
    main()