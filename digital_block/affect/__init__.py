# digital_block/affect/__init__.py

from .text_affect_extractor import TextAffectExtractor
from .voice_affect_extractor import VoiceAffectExtractor
from .fusion_affect import fuse_affect

__all__ = [
    "TextAffectExtractor",
    "VoiceAffectExtractor",
    "fuse_affect",
]
