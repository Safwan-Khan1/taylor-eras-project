from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from debate_orchestrator.fusion import early_fusion, late_fusion
from debate_orchestrator.protocol import round_2


# -----------------------------
# STATE
# -----------------------------
class DebateState(TypedDict):
    lyrics: str
    audio_features: dict

    text_out: Dict[str, Any]
    audio_out: Dict[str, Any]

    round_2: Dict[str, Any]
    early: Dict[str, Any]
    late: Dict[str, Any]
    final: Dict[str, Any]


# -----------------------------
# BUILD GRAPH
# -----------------------------
def build_graph(text_agent, audio_agent):

    graph = StateGraph(DebateState)

    # -------------------------
    # NODES (USE TRAINED MODELS)
    # -------------------------
    def text_node(state: DebateState):
        state["text_out"] = text_agent.predict_with_evidence(state["lyrics"])
        return state

    def audio_node(state: DebateState):
        state["audio_out"] = audio_agent.predict_with_evidence(state["audio_features"])
        return state

    def critique_node(state: DebateState):
        state["round_2"] = round_2(state["text_out"], state["audio_out"])
        return state

    def early_fusion_node(state: DebateState):
        state["early"] = early_fusion(
            state["text_out"]["probabilities"],
            state["audio_out"]["probabilities"]
        )
        return state

    def late_fusion_node(state: DebateState):
        state["late"] = late_fusion(
            state["text_out"]["probabilities"],
            state["audio_out"]["probabilities"]
        )
        return state

    def final_node(state: DebateState):
        state["final"] = {
            "prediction": state["late"]["prediction"],
            "agreement": state["text_out"]["predicted_era"]
                          == state["audio_out"]["predicted_era"],
            "fusion_agreement": state["early"]["prediction"]
                                == state["late"]["prediction"]
        }
        return state

    # -------------------------
    # GRAPH STRUCTURE
    # -------------------------
    graph.add_node("text", text_node)
    graph.add_node("audio", audio_node)
    graph.add_node("critique", critique_node)
    graph.add_node("early_fusion", early_fusion_node)
    graph.add_node("late_fusion", late_fusion_node)
    graph.add_node("final", final_node)

    graph.set_entry_point("text")

    graph.add_edge("text", "audio")
    graph.add_edge("audio", "critique")
    graph.add_edge("critique", "early_fusion")
    graph.add_edge("early_fusion", "late_fusion")
    graph.add_edge("late_fusion", "final")
    graph.add_edge("final", END)

    return graph.compile()


# -----------------------------
# PUBLIC RUN FUNCTION
# -----------------------------
def run(lyrics: str, audio_features: dict, text_agent, audio_agent):

    app = build_graph(text_agent, audio_agent)

    initial_state = {
        "lyrics": lyrics,
        "audio_features": audio_features
    }

    return app.invoke(initial_state)