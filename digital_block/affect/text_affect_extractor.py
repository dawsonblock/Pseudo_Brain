# digital_block/affect/text_affect_extractor.py

from typing import Dict


class TextAffectExtractor:
    """
    Text → generic affect vector (heuristic stub).

    Replace with a proper classifier later (e.g., small transformer).
    """

    def __init__(self) -> None:
        pass

    def analyze(self, text: str) -> Dict[str, float]:
        t = text.lower()
        affect = {
            "valence": 0.0,    # [-1,1]
            "arousal": 0.3,    # [0,1]
            "anxiety": 0.0,
            "anger": 0.0,
            "sadness": 0.0,
            "curiosity": 0.3,
            "confidence": 0.5,
            "tension": 0.0,
        }

        neg = ["worried", "afraid", "scared", "anxious", "angry", "upset", "bad", "hate"]
        pos = ["good", "great", "excited", "happy", "love", "nice", "cool", "awesome"]
        cur = ["wonder", "curious", "why", "how", "what if", "explore"]

        if any(w in t for w in neg):
            affect["valence"] -= 0.3
            affect["anxiety"] += 0.4
            affect["arousal"] += 0.2
            affect["tension"] += 0.2
        if any(w in t for w in pos):
            affect["valence"] += 0.4
            affect["arousal"] += 0.2
            affect["confidence"] += 0.2
        if any(w in t for w in cur):
            affect["curiosity"] += 0.4
            affect["arousal"] += 0.2

        affect["valence"] = max(-1.0, min(1.0, affect["valence"]))
        for k in ["arousal", "anxiety", "anger", "sadness", "curiosity", "confidence", "tension"]:
            affect[k] = max(0.0, min(1.0, affect[k]))

        return affect
