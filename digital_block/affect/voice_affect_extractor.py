# digital_block/affect/voice_affect_extractor.py

from typing import Dict, Optional


class VoiceAffectExtractor:
    """
    Audio → affect-like features (stub).

    Replace with a real prosody/emotion model later.
    """

    def __init__(self) -> None:
        pass

    def analyze(self, audio_path: Optional[str]) -> Dict[str, float]:
        if audio_path is None:
            return {
                "valence_audio": 0.0,
                "arousal_audio": 0.3,
                "tension_audio": 0.0,
            }

        # TODO: implement with torchaudio / pretrained model.
        return {
            "valence_audio": 0.0,
            "arousal_audio": 0.3,
            "tension_audio": 0.0,
        }
