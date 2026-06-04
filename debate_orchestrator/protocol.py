def round_1(text_out, audio_out):
    return {
        "text_agent": text_out,
        "audio_agent": audio_out
    }

def round_2(text_out, audio_out):

    disagreement = (
        text_out["predicted_era"] !=
        audio_out["predicted_era"]
    )

    return {
        "conflict": disagreement,
        "text_critique": {
            "statement": f"My lyrical analysis strongly supports {text_out['predicted_era']}",
            "reason": text_out["evidence"]
        },
        "audio_critique": {
            "statement": f"Audio features indicate {audio_out['predicted_era']} is more likely",
            "reason": audio_out["evidence"]
        }
    }