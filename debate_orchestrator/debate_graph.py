from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from src.text_agent import TextAgent
from src.audio_agent import AudioAgent
from debate_orchestrator.fusion import early_fusion, late_fusion
from debate_orchestrator.protocol import round_1, round_2


# -------------------------------------------------
# STATE DEFINITION
# -------------------------------------------------
class DebateState(TypedDict):
    lyrics: str
    audio_features: Dict[str, float]

    text_out: Dict[str, Any]
    audio_out: Dict[str, Any]

    round_1: Dict[str, Any]
    round_2: Dict[str, Any]

    early_fusion: Dict[str, Any]
    late_fusion: Dict[str, Any]

    final_result: Dict[str, Any]


# -------------------------------------------------
# AGENTS (loaded once)
# -------------------------------------------------
text_agent = TextAgent()
audio_agent = AudioAgent()


# -------------------------------------------------
# NODES
# -------------------------------------------------
def text_node(state: DebateState):
    state["text_out"] = text_agent.predict_with_evidence(state["lyrics"])
    return state


def audio_node(state: DebateState):
    state["audio_out"] = audio_agent.predict_with_evidence(state["audio_features"])
    return state


def round1_node(state: DebateState):
    state["round_1"] = round_1(state["text_out"], state["audio_out"])
    return state


def round2_node(state: DebateState):
    state["round_2"] = round_2(state["text_out"], state["audio_out"])
    return state


def fusion_node(state: DebateState):
    state["early_fusion"] = early_fusion(
        state["text_out"]["probabilities"],
        state["audio_out"]["probabilities"]
    )

    state["late_fusion"] = late_fusion(
        state["text_out"]["probabilities"],
        state["audio_out"]["probabilities"]
    )

    return state


def final_node(state: DebateState):
    state["final_result"] = {
        "prediction": state["late_fusion"]["prediction"],
        "agreement": state["text_out"]["predicted_era"] == state["audio_out"]["predicted_era"],
        "fusion_agreement": state["early_fusion"]["prediction"] == state["late_fusion"]["prediction"]
    }
    return state


# -------------------------------------------------
# BUILD GRAPH
# -------------------------------------------------
def build_graph():
    graph = StateGraph(DebateState)

    graph.add_node("text", text_node)
    graph.add_node("audio", audio_node)
    graph.add_node("round1", round1_node)
    graph.add_node("round2", round2_node)
    graph.add_node("fusion", fusion_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("text")

    graph.add_edge("text", "audio")
    graph.add_edge("audio", "round1")
    graph.add_edge("round1", "round2")
    graph.add_edge("round2", "fusion")
    graph.add_edge("fusion", "final")
    graph.add_edge("final", END)

    return graph.compile()


# -------------------------------------------------
# RUN FUNCTION
# -------------------------------------------------
def run_debate(lyrics, audio_features):
    app = build_graph()

    state = {
        "lyrics": lyrics,
        "audio_features": audio_features
    }

    result = app.invoke(state)
    return result