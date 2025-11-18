# digital_block/distillation/conversational_distiller.py

from typing import Optional

import torch

from digital_block.conversation_event import ConversationEvent
from digital_block.affect import TextAffectExtractor, VoiceAffectExtractor, fuse_affect
from digital_block.block_style import BlockStyleMapper


class ConversationalDistiller:
    """
    Orchestrates:
      - text affect extraction
      - voice affect extraction
      - fusion
      - Block-style slider inference
    """

    def __init__(self, mapper_weights_path: Optional[str] = None) -> None:
        self.text_affect = TextAffectExtractor()
        self.voice_affect = VoiceAffectExtractor()
        self.block_mapper = BlockStyleMapper()
        if mapper_weights_path:
            self.block_mapper.load_state_dict(
                torch.load(mapper_weights_path, map_location="cpu")
            )
        self.block_mapper.eval()

    def process_event(self, ev: ConversationEvent, infer_block: bool = True) -> ConversationEvent:
        text_aff = self.text_affect.analyze(ev.text)
        voice_aff = self.voice_affect.analyze(ev.audio_path)
        fused = fuse_affect(text_aff, voice_aff)

        ev.affect_generic = fused

        if infer_block:
            ev.block_labels = self.block_mapper.forward_from_dict(fused)

        return ev
