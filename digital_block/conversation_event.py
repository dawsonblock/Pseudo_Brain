# digital_block/conversation_event.py

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import uuid
import time


@dataclass
class ConversationEvent:
    event_id: str
    timestamp: float
    source: str                 # "user" or "ai"
    text: str
    audio_path: Optional[str] = None

    model_signals: Dict[str, Any] = field(default_factory=dict)
    affect_generic: Optional[Dict[str, float]] = None
    block_labels: Optional[Dict[str, float]] = None
    emotion_state: Optional[Dict[str, float]] = None


def new_event(
    source: str,
    text: str,
    audio_path: Optional[str] = None,
    model_signals: Optional[Dict[str, Any]] = None,
) -> ConversationEvent:
    return ConversationEvent(
        event_id=str(uuid.uuid4()),
        timestamp=time.time(),
        source=source,
        text=text,
        audio_path=audio_path,
        model_signals=model_signals or {},
    )
