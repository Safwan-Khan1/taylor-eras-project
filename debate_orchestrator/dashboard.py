import sys
import io

def print_dashboard(result):

    out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    out.write("\n" + "="*60 + "\n")
    out.write("TAYLOR ERAS DEBATE SYSTEM (LANGGRAPH)\n")
    out.write("="*60 + "\n")

    final = result.get("final", {})
    text = result.get("text_out", {})
    audio = result.get("audio_out", {})
    early = result.get("early", {})
    late = result.get("late", {})
    critique = result.get("round_2", {})

    # ---------------- FINAL ----------------
    out.write("\nFINAL VERDICT\n")
    out.write("-"*60 + "\n")
    out.write(f"Prediction        : {final.get('prediction')}\n")
    out.write(f"Agreement         : {final.get('agreement')}\n")
    out.write(f"Fusion Agreement  : {final.get('fusion_agreement')}\n")

    # ---------------- TEXT ----------------
    out.write("\nTEXT AGENT\n")
    out.write("-"*60 + "\n")
    out.write(f"Prediction  : {text.get('predicted_era')}\n")
    out.write(f"Confidence  : {round(max(text.get('probabilities', {}).values()), 3)}\n")
    out.write(f"Reasoning   : {text.get('reasoning', '')[:120]} ...\n")

    # ---------------- AUDIO ----------------
    out.write("\nAUDIO AGENT\n")
    out.write("-"*60 + "\n")
    out.write(f"Prediction  : {audio.get('predicted_era')}\n")
    out.write(f"Confidence  : {round(max(audio.get('probabilities', {}).values()), 3)}\n")
    out.write(f"Reasoning   : {audio.get('reasoning', '')[:120]} ...\n")

    # ---------------- FUSION ----------------
    out.write("\nFUSION STAGE\n")
    out.write("-"*60 + "\n")
    out.write(f"Early Fusion: {early.get('prediction')}\n")
    out.write(f"Late Fusion : {late.get('prediction')}\n")

    # ---------------- DEBATE ----------------
    out.write("\nROUND 2 DEBATE\n")
    out.write("-"*60 + "\n")
    out.write(f"Conflict    : {critique.get('conflict')}\n")
    out.write(f"Text stance : {critique.get('text_critique', {}).get('statement')}\n")
    out.write(f"Audio stance: {critique.get('audio_critique', {}).get('statement')}\n")

    out.write("\n" + "="*60 + "\n")
    out.flush()