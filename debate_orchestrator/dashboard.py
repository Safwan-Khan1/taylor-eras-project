def print_dashboard(result):

    print("\n" + "="*60)
    print("🎵 TAYLOR ERAS DEBATE SYSTEM (LANGGRAPH)")
    print("="*60)

    final = result.get("final", {})
    text = result.get("text_out", {})
    audio = result.get("audio_out", {})
    early = result.get("early", {})
    late = result.get("late", {})
    critique = result.get("round_2", {})

    # ---------------- FINAL ----------------
    print("\n🏁 FINAL VERDICT")
    print("-"*60)
    print("Prediction        :", final.get("prediction"))
    print("Agreement         :", final.get("agreement"))
    print("Fusion Agreement  :", final.get("fusion_agreement"))

    # ---------------- TEXT ----------------
    print("\n🤖 TEXT AGENT")
    print("-"*60)
    print("Prediction  :", text.get("predicted_era"))
    print("Confidence  :", round(max(text.get("probabilities", {}).values()), 3))
    print("Reasoning   :", text.get("reasoning", "")[:120], "...")

    # ---------------- AUDIO ----------------
    print("\n🎧 AUDIO AGENT")
    print("-"*60)
    print("Prediction  :", audio.get("predicted_era"))
    print("Confidence  :", round(max(audio.get("probabilities", {}).values()), 3))
    print("Reasoning   :", audio.get("reasoning", "")[:120], "...")

    # ---------------- FUSION ----------------
    print("\n⚖️ FUSION STAGE")
    print("-"*60)
    print("Early Fusion:", early.get("prediction"))
    print("Late Fusion :", late.get("prediction"))

    # ---------------- DEBATE ----------------
    print("\n🧠 ROUND 2 DEBATE")
    print("-"*60)
    print("Conflict    :", critique.get("conflict"))
    print("Text stance :", critique.get("text_critique", {}).get("statement"))
    print("Audio stance:", critique.get("audio_critique", {}).get("statement"))

    print("\n" + "="*60)