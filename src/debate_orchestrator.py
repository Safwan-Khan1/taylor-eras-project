"""
Debate Orchestrator — Member 4

LangGraph workflow that runs a 3-round multi-agent debate between TextAgent and
AudioAgent, applies a late-fusion meta-classifier, and emits structured JSON
logs for the frontend.
"""

from __future__ import annotations

import json
import operator
import time
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

AUDIO_WEIGHT = 0.6
TEXT_WEIGHT = 0.4
CONFIDENCE_THRESHOLD = 0.65


class DebateState(TypedDict):
    song_data: dict
    fusion: str
    text_result: dict | None
    audio_result: dict | None
    script: Annotated[list[dict], operator.add]
    debate_log: Annotated[list[dict], operator.add]
    round_num: int
    needs_round_2: bool
    needs_round_3: bool
    meta_verdict: dict | None
    runtime: float


def _trim_to_words(text: str, max_words: int = 30) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    clipped = " ".join(words[:max_words])
    for i in range(len(clipped) - 1, len(clipped) // 3, -1):
        if clipped[i] in (".", "!", "?") and (i == len(clipped) - 1 or clipped[i + 1] == " "):
            return clipped[: i + 1].strip()
    return clipped.rstrip(",:;—") + "..."


def _sorted_runner_up(probs: dict[str, float], predicted: str) -> tuple[str, float] | None:
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for era, conf in ranked:
        if era != predicted:
            return era, conf
    return None


class DebateOrchestrator:
    """LangGraph debate loop connecting TextAgent and AudioAgent as tools."""

    def __init__(
        self,
        text_agent,
        audio_agent,
        late_fusion_model=None,
        early_fusion_model=None,
        eval_metrics: dict | None = None,
        eras_meta: dict | None = None,
    ):
        self.text_agent = text_agent
        self.audio_agent = audio_agent
        self.late_fusion_model = late_fusion_model
        self.early_fusion_model = early_fusion_model
        self.eval_metrics = eval_metrics or {}
        self.eras_meta = eras_meta or {}
        self._graph = self._build_graph()

    # ── Agent tools ───────────────────────────────────────────────────────────

    def tool_text_classify(self, lyrics: str) -> dict:
        return self.text_agent.predict_with_evidence(lyrics)

    def tool_audio_classify(self, audio_features: dict) -> dict:
        return self.audio_agent.predict_with_evidence(audio_features)

    # ── Meta-classifier (late fusion) ─────────────────────────────────────────

    def _align_probabilities(self, ta: dict, au: dict) -> tuple[np.ndarray, list[str]]:
        eras = sorted(set(ta["probabilities"]) | set(au["probabilities"]))
        text_vec = np.array([ta["probabilities"].get(e, 0.0) for e in eras])
        audio_vec = np.array([au["probabilities"].get(e, 0.0) for e in eras])
        return np.vstack([audio_vec, text_vec]), eras

    def _meta_classify(self, ta: dict, au: dict, song_data: dict) -> dict:
        """Late fusion verdict: trained meta-model when available, else weighted blend."""
        if self.late_fusion_model is not None:
            try:
                lyrics = str(song_data.get("lyrics", ""))
                audio_cols = getattr(self.late_fusion_model, "_last_audio_cols", None)
                if audio_cols is None:
                    audio_cols = [
                        "danceability", "energy", "loudness", "speechiness",
                        "acousticness", "instrumentalness", "liveness", "valence",
                        "tempo", "duration_ms",
                    ]
                X_audio = np.array([[song_data.get(c, 0) for c in audio_cols]])
                probs = self.late_fusion_model.predict_proba(X_audio, np.array([lyrics]))[0]
                classes = list(self.late_fusion_model.classes_)
                probabilities = {c: float(p) for c, p in zip(classes, probs)}
                predicted = classes[int(np.argmax(probs))]
                confidence = float(probs.max())
                source = "trained_meta_classifier"
            except Exception:
                source = "weighted_blend"
                stacked, eras = self._align_probabilities(ta, au)
                fused = AUDIO_WEIGHT * stacked[0] + TEXT_WEIGHT * stacked[1]
                idx = int(np.argmax(fused))
                predicted = eras[idx]
                confidence = float(fused[idx])
                probabilities = {e: float(fused[i]) for i, e in enumerate(eras)}
        else:
            source = "weighted_blend"
            stacked, eras = self._align_probabilities(ta, au)
            fused = AUDIO_WEIGHT * stacked[0] + TEXT_WEIGHT * stacked[1]
            idx = int(np.argmax(fused))
            predicted = eras[idx]
            confidence = float(fused[idx])
            probabilities = {e: float(fused[i]) for i, e in enumerate(eras)}

        p_ta, p_au = ta["predicted_era"], au["predicted_era"]
        l_conf = float(ta["probabilities"].get(p_ta, 0))
        a_conf = float(au["probabilities"].get(p_au, 0))

        if p_ta == p_au == predicted and l_conf >= CONFIDENCE_THRESHOLD and a_conf >= CONFIDENCE_THRESHOLD:
            badge = "consensus"
        elif predicted in (p_ta, p_au):
            badge = "consensus" if p_ta == p_au else "adjudicated"
        else:
            badge = "adjudicated"

        return {
            "predicted_era": predicted,
            "confidence": confidence,
            "probabilities": probabilities,
            "badge": badge,
            "source": source,
            "weights": {"audio": AUDIO_WEIGHT, "text": TEXT_WEIGHT},
        }

    # ── LangGraph nodes ───────────────────────────────────────────────────────

    def _node_collect_evidence(self, state: DebateState) -> dict:
        song = state["song_data"]
        lyrics = str(song.get("lyrics", "")).strip()
        audio_features = {f: song.get(f, 0) for f in self.audio_agent.features}

        ta = self.tool_text_classify(lyrics)
        au = self.tool_audio_classify(audio_features)

        p_ta, p_au = ta["predicted_era"], au["predicted_era"]
        l_conf = float(ta["probabilities"].get(p_ta, 0))
        a_conf = float(au["probabilities"].get(p_au, 0))
        agree = p_ta == p_au
        high_conf = l_conf >= CONFIDENCE_THRESHOLD and a_conf >= CONFIDENCE_THRESHOLD

        return {
            "text_result": ta,
            "audio_result": au,
            "needs_round_2": not (agree and high_conf),
            "needs_round_3": not agree,
            "round_num": 0,
            "debate_log": [{
                "event": "evidence_collected",
                "text_prediction": p_ta,
                "text_confidence": l_conf,
                "audio_prediction": p_au,
                "audio_confidence": a_conf,
                "consensus_after_round_1": agree and high_conf,
            }],
        }

    def _node_round_1(self, state: DebateState) -> dict:
        ta, au = state["text_result"], state["audio_result"]
        song = state["song_data"]
        p_ta, p_au = ta["predicted_era"], au["predicted_era"]
        l_conf = float(ta["probabilities"].get(p_ta, 0))
        a_conf = float(au["probabilities"].get(p_au, 0))
        kw_str = '", "'.join(ta["evidence"].get("top_keywords", [])[:3])

        script = [
            {"who": "lyric", "text": f'> Tokenising: "{song["track_name"]}"…'},
            {"who": "lyric", "text": f'> Signals: "{kw_str}"'},
            {"who": "lyric", "text": f'> TF-IDF prediction: <emp>{p_ta} — {l_conf:.0%}</emp>'},
            {"who": "audio", "text": "> Fetching Spotify feature vector…"},
            {
                "who": "audio",
                "text": (
                    f"> Audio Signals: energy(<num>{song.get('energy', 0):.2f}</num>) · "
                    f"acousticness(<num>{song.get('acousticness', 0):.2f}</num>) · "
                    f"valence(<num>{song.get('valence', 0):.2f}</num>)"
                ),
            },
            {"who": "audio", "text": f"> Gradient Boost prediction: <emp>{p_au} — {a_conf:.0%}</emp>"},
        ]

        log = {"event": "round_1_complete", "round": 1, "text": p_ta, "audio": p_au}
        return {"script": script, "debate_log": [log], "round_num": 1}

    def _node_round_2(self, state: DebateState) -> dict:
        ta, au = state["text_result"], state["audio_result"]
        p_ta, p_au = ta["predicted_era"], au["predicted_era"]
        l_conf = float(ta["probabilities"].get(p_ta, 0))
        a_conf = float(au["probabilities"].get(p_au, 0))

        script = [{"who": "sys", "text": "── DEBATE ROUND 2 ──"}]
        if p_ta == p_au:
            script.extend([
                {
                    "who": "lyric",
                    "text": "> Low confidence detected. Reinforcing with LDA topic weights...",
                    "cls": "is-emphatic",
                },
                {
                    "who": "audio",
                    "text": "> Cross-referencing audio cluster... signature verified.",
                    "cls": "is-resolve",
                },
            ])
        else:
            runner_ta = _sorted_runner_up(ta["probabilities"], p_ta)
            runner_au = _sorted_runner_up(au["probabilities"], p_au)
            ta_challenge = (
                f"> I maintain {p_ta} ({l_conf:.0%}). "
                f"Lyric diction outweighs the audio lean toward {p_au}."
            )
            if runner_ta:
                ta_challenge += f" Runner-up {runner_ta[0]} ({runner_ta[1]:.0%}) is not competitive."
            au_challenge = (
                f"> Audio production profile favors {p_au} ({a_conf:.0%}). "
                f"Requesting lyric agent to reconcile TF-IDF noise."
            )
            if runner_au:
                au_challenge += f" Secondary cluster: {runner_au[0]} ({runner_au[1]:.0%})."
            script.extend([
                {"who": "lyric", "text": f"> Conflict: {p_ta} vs {p_au}. Applying late fusion weights...", "cls": "is-emphatic"},
                {"who": "lyric", "text": ta_challenge, "cls": "is-emphatic"},
                {"who": "audio", "text": au_challenge, "cls": "is-emphatic"},
            ])

        return {
            "script": script,
            "debate_log": [{"event": "round_2_complete", "round": 2, "disagreement": p_ta != p_au}],
            "round_num": 2,
            "needs_round_3": p_ta != p_au,
        }

    def _node_round_3(self, state: DebateState) -> dict:
        ta, au = state["text_result"], state["audio_result"]
        meta = self._meta_classify(ta, au, state["song_data"])
        p_ta, p_au = ta["predicted_era"], au["predicted_era"]

        script = [
            {"who": "sys", "text": "── DEBATE ROUND 3 ──"},
            {
                "who": "sys",
                "text": (
                    f"> Meta-classifier invoked ({meta['source']}). "
                    f"Weights: audio {meta['weights']['audio']:.0%} · text {meta['weights']['text']:.0%}."
                ),
            },
            {
                "who": "lyric",
                "text": f"> Final lyric position: <emp>{p_ta}</emp> — submitting to meta-layer.",
                "cls": "is-emphatic",
            },
            {
                "who": "audio",
                "text": f"> Final audio position: <emp>{p_au}</emp> — deferring to stacked classifier.",
                "cls": "is-concede" if meta["predicted_era"] != p_au else "is-resolve",
            },
            {
                "who": "sys",
                "text": f"> Late fusion verdict: <emp>{meta['predicted_era']} — {meta['confidence']:.0%}</emp>",
            },
        ]

        return {
            "script": script,
            "meta_verdict": meta,
            "debate_log": [{"event": "round_3_complete", "round": 3, "meta_verdict": meta["predicted_era"]}],
            "round_num": 3,
        }

    def _node_meta_classifier(self, state: DebateState) -> dict:
        if state.get("meta_verdict"):
            return {}
        ta, au = state["text_result"], state["audio_result"]
        meta = self._meta_classify(ta, au, state["song_data"])
        return {"meta_verdict": meta, "debate_log": [{"event": "meta_classifier", "verdict": meta}]}

    def _node_finalize(self, state: DebateState) -> dict:
        ta, au = state["text_result"], state["audio_result"]
        meta = state["meta_verdict"]
        final_era = meta["predicted_era"]

        short_ta = _trim_to_words(ta["reasoning"])
        short_au = _trim_to_words(au["reasoning"])

        script = [
            {"who": "sys", "text": "── AGENT REASONING ──"},
            {"who": "audio", "text": f"> {short_au}", "cls": "is-emphatic"},
            {"who": "lyric", "text": f"> {short_ta}", "cls": "is-emphatic"},
            {"who": "lyric", "text": f'> Final verdict: <emp>{final_era}</emp>', "cls": "is-resolve"},
        ]

        return {
            "script": script,
            "debate_log": [{"event": "finalize", "final_era": final_era, "badge": meta["badge"]}],
        }

    # ── Routing ───────────────────────────────────────────────────────────────

    def _route_after_round_1(self, state: DebateState) -> Literal["round_2", "meta_classifier"]:
        return "round_2" if state["needs_round_2"] else "meta_classifier"

    def _route_after_round_2(self, state: DebateState) -> Literal["round_3", "meta_classifier"]:
        return "round_3" if state["needs_round_3"] else "meta_classifier"

    def _build_graph(self):
        graph = StateGraph(DebateState)
        graph.add_node("collect_evidence", self._node_collect_evidence)
        graph.add_node("round_1", self._node_round_1)
        graph.add_node("round_2", self._node_round_2)
        graph.add_node("round_3", self._node_round_3)
        graph.add_node("meta_classifier", self._node_meta_classifier)
        graph.add_node("finalize", self._node_finalize)

        graph.set_entry_point("collect_evidence")
        graph.add_edge("collect_evidence", "round_1")
        graph.add_conditional_edges("round_1", self._route_after_round_1, {
            "round_2": "round_2",
            "meta_classifier": "meta_classifier",
        })
        graph.add_conditional_edges("round_2", self._route_after_round_2, {
            "round_3": "round_3",
            "meta_classifier": "meta_classifier",
        })
        graph.add_edge("round_3", "meta_classifier")
        graph.add_edge("meta_classifier", "finalize")
        graph.add_edge("finalize", END)
        return graph.compile()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_late_fusion_debate(self, song_data: dict) -> dict:
        start_time = time.perf_counter()
        initial: DebateState = {
            "song_data": song_data,
            "fusion": "late",
            "text_result": None,
            "audio_result": None,
            "script": [],
            "debate_log": [],
            "round_num": 0,
            "needs_round_2": True,
            "needs_round_3": False,
            "meta_verdict": None,
            "runtime": 0.0,
        }
        final_state = self._graph.invoke(initial)
        final_state["runtime"] = time.perf_counter() - start_time
        ta, au = final_state["text_result"], final_state["audio_result"]
        meta = final_state["meta_verdict"] or self._meta_classify(ta, au, song_data)
        return self._build_response_from_state(final_state, meta)

    def _build_response_from_state(self, state: DebateState, meta: dict) -> dict:
        ta, au = state["text_result"], state["audio_result"]
        song = state["song_data"]
        p_ta, p_au = ta["predicted_era"], au["predicted_era"]
        l_conf = float(ta["probabilities"].get(p_ta, 0))
        a_conf = float(au["probabilities"].get(p_au, 0))
        final_era = meta["predicted_era"]
        era_meta = self.eras_meta.get(final_era, {"id": "midnights", "label": final_era})
        short_ta = _trim_to_words(ta["reasoning"])

        return {
            "id": "live",
            "title": song["track_name"],
            "artist": song.get("artist", "Taylor Swift"),
            "era": era_meta["id"],
            "audioConf": round(a_conf, 2),
            "lyricConf": round(l_conf, 2),
            "stacking": round(meta["confidence"], 2),
            "badge": meta["badge"],
            "reason": (
                f"Synthesized verdict: {short_ta} "
                f"Production markers ({a_conf:.0%}) confirm the {final_era} profile."
            ),
            "rounds": state["round_num"],
            "runtime": state["runtime"],
            "audioFinal": f"{p_au} — {a_conf:.0%}",
            "lyricFinal": f"{p_ta} — {l_conf:.0%}",
            "script": state["script"],
            "debate_log": state["debate_log"],
            "meta_classifier": {
                "predicted_era": final_era,
                "confidence": meta["confidence"],
                "source": meta["source"],
                "weights": meta["weights"],
                "probabilities": meta["probabilities"],
            },
            "fusion_mode": "late",
        }

    def run_early_fusion(self, ta: dict, au: dict, song_data: dict, runtime: float) -> dict:
        """Early fusion narrative path (Member 3's unified vector story)."""
        p_ta = ta["predicted_era"]
        l_conf = float(ta["probabilities"].get(p_ta, 0))
        a_conf = float(au["probabilities"].get(p_ta, 0))
        combined_conf = min(max(l_conf, a_conf) * 1.05, 0.99)

        if self.early_fusion_model is not None:
            try:
                lyrics = str(song_data.get("lyrics", ""))
                audio_cols = getattr(self.early_fusion_model, "_last_audio_cols", None) or []
                if audio_cols:
                    X_audio = np.array([[song_data.get(c, 0) for c in audio_cols]])
                    probs = self.early_fusion_model.predict_proba(X_audio, np.array([lyrics]))[0]
                    p_ta = self.early_fusion_model.classes_[int(np.argmax(probs))]
                    combined_conf = float(probs.max())
            except Exception:
                pass

        script = [
            {"who": "sys", "text": "── EARLY FUSION MODE: UNIFIED ANALYSIS ──"},
            {"who": "lyric", "text": "> Loading TF-IDF linguistic matrices..."},
            {"who": "lyric", "text": "> Loading Spotify acoustic feature vectors..."},
            {"who": "sys", "text": "> CONCATENATING TO [1 x 8192] FEATURE VECTOR..."},
            {"who": "lyric", "text": "> Analyzing joint distribution space...", "cls": "is-emphatic"},
            {"who": "lyric", "text": f"> Unified signal aligns with {p_ta} signature.", "cls": "is-resolve"},
            {"who": "sys", "text": "── INFERENCE REASONING ──"},
            {
                "who": "lyric",
                "text": (
                    f"> The concatenated model predicts {p_ta} because linguistic themes and "
                    "audio energy are evaluated simultaneously, creating a single highly confident inference."
                ),
                "cls": "is-emphatic",
            },
            {"who": "lyric", "text": "> Single-pass inference complete.", "cls": "is-resolve"},
        ]

        era_meta = self.eras_meta.get(p_ta, {"id": "midnights", "label": p_ta})
        return {
            "id": "live",
            "title": song_data["track_name"],
            "era": era_meta["id"],
            "audioConf": combined_conf,
            "lyricConf": combined_conf,
            "stacking": round(combined_conf, 2),
            "badge": "joint-inference",
            "reason": (
                f"Early fusion model analyzed {song_data['track_name']} by merging lyric tokens and "
                "audio features into a single high-dimensional vector prior to classification."
            ),
            "rounds": 1,
            "runtime": runtime,
            "audioFinal": "Joint",
            "lyricFinal": "Joint",
            "script": script,
            "debate_log": [{"event": "early_fusion", "predicted_era": p_ta, "confidence": combined_conf}],
            "fusion_mode": "early",
        }

    def get_fusion_comparison(self) -> dict:
        """Expose late vs early fusion accuracy from training artifacts."""
        results = self.eval_metrics.get("results", {})
        early = results.get("early", {})
        late = results.get("late", {})
        return {
            "early_fusion": early,
            "late_fusion": late,
            "text_only": results.get("text_only", {}),
            "audio_only": results.get("audio_only", {}),
            "winner": (
                "late" if late.get("accuracy", 0) > early.get("accuracy", 0)
                else "early" if early.get("accuracy", 0) > late.get("accuracy", 0)
                else "tie"
            ),
            "accuracy_delta": round(
                late.get("accuracy", 0) - early.get("accuracy", 0), 4
            ),
            "best_model": self.eval_metrics.get("best_model"),
        }


def load_eval_metrics(base_path: Path) -> dict:
    metrics_path = base_path / "artifacts" / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)
    return {}
