# digital_block/affect/fusion_affect.py

from typing import Dict


def fuse_affect(text_affect: Dict[str, float], voice_affect: Dict[str, float]) -> Dict[str, float]:
    """
    Fuse text-based and voice-based affect into a single vector.
    """

    valence = text_affect.get("valence", 0.0) + 0.5 * voice_affect.get("valence_audio", 0.0)
    arousal = (
        0.7 * text_affect.get("arousal", 0.3)
        + 0.3 * voice_affect.get("arousal_audio", 0.3)
    )
    tension = max(0.0, min(1.0, voice_affect.get("tension_audio", 0.0)))

    fused = dict(text_affect)
    fused["valence"] = max(-1.0, min(1.0, valence))
    fused["arousal"] = max(0.0, min(1.0, arousal))
    fused["tension"] = tension
    return fused
