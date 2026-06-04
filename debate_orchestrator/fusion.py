import numpy as np


TEXT_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4


def late_fusion(text_probs, audio_probs):
    eras = set(text_probs.keys()) | set(audio_probs.keys())

    final_scores = {}

    for era in eras:
        final_scores[era] = (
            TEXT_WEIGHT * text_probs.get(era, 0.0)
            + AUDIO_WEIGHT * audio_probs.get(era, 0.0)
        )

    final_prediction = max(final_scores, key=final_scores.get)

    return {
        "prediction": final_prediction,
        "scores": final_scores
    }

def early_fusion(text_probs, audio_probs):
    # simple stacking-style average baseline
    eras = set(text_probs.keys()) | set(audio_probs.keys())

    scores = {}

    for era in eras:
        scores[era] = (text_probs.get(era, 0) + audio_probs.get(era, 0)) / 2

    return {
        "prediction": max(scores, key=scores.get),
        "scores": scores
    }