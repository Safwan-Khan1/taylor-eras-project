from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AgentOutput:
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    evidence: str