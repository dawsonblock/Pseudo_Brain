# digital_block/runtime/digital_block_runtime.py

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
import torch

from digital_block.conversation_event import ConversationEvent, new_event
from digital_block.distillation import ConversationalDistiller
from digital_block.traits import TraitUpdater
from digital_block.emotion import EmotionStateEngine, IdentityStabilizer, contagion_update

from digital_block_profile import (
    get_default_trait_vector,
    E_BASELINE,
)


@dataclass
class DigitalBlockState:
    traits: Any          # TraitVector
    E: torch.Tensor      # [16]
    events: List[ConversationEvent] = field(default_factory=list)


class DigitalBlockRuntime:
    """
    Digital Block v1.1 – emotional duplicate core.

    Plug PMM / Capsule Brain / LLM into generate_ai_response().
    """

    def __init__(
        self,
        pmm_model: Optional[Any] = None,
        block_mapper_weights: Optional[str] = None,
    ) -> None:
        self.pmm = pmm_model
        self.distiller = ConversationalDistiller(mapper_weights_path=block_mapper_weights)
        self.trait_updater = TraitUpdater(lr=1e-3)
        self.emotion_engine = EmotionStateEngine(dim=16)
        self.identity_stabilizer = IdentityStabilizer(trait_dim=5, emotion_dim=16)

        traits = get_default_trait_vector()
        self.state = DigitalBlockState(
            traits=traits,
            E=E_BASELINE.clone(),  # start exactly at your emotional baseline
        )

    def handle_user_message(self, text: str, audio_path: Optional[str] = None) -> str:
        # 1. Create and log user event
        ev = new_event(source="user", text=text, audio_path=audio_path)

        # 2. Distill affect + Block-style labels
        ev = self.distiller.process_event(ev)

        # 3. Compute model signals (stub; connect PMM here)
        ev.model_signals["U_e"] = 0.2
        ev.model_signals["U_a"] = 0.3

        # 4. Emotional update
        fused_affect = ev.affect_generic or {}
        block_labels = ev.block_labels or {}
        traits_tensor = self.state.traits.to_tensor()

        E_prev = self.state.E
        E_raw = self.emotion_engine(
            E_prev=E_prev,
            fused_affect=fused_affect,
            block_labels=block_labels,
            traits=traits_tensor,
        )

        # Optional: implement affect→16D mapping to get a real E_human.
        # For now, keep contagion disabled:
        E_human = None

        E_contagious = contagion_update(E_prev, E_raw, E_human, traits_tensor)
        E_final = self.identity_stabilizer.stabilize(
            E_contagious,
            traits_tensor,
            baseline=E_BASELINE,
        )

        self.state.E = E_final
        ev.emotion_state = {f"e_{i}": float(v) for i, v in enumerate(E_final.tolist())}
        self.state.events.append(ev)

        # 5. Generate AI response – hook in your main controller here
        response = self.generate_ai_response(text, self.state)

        # 6. Log AI event if desired
        ai_ev = new_event(source="ai", text=response)
        self.state.events.append(ai_ev)

        return response

    def generate_ai_response(self, user_text: str, state: DigitalBlockState) -> str:
        """
        Hook into Codex / Capsule Brain / PMM+LLM here.

        Use:
          - user_text
          - state.E (16D emotion vector)
          - state.traits
          - recent state.events
        to control tone / caution / curiosity.
        """
        # Placeholder: echo with emotional context marker
        calm = state.traits.baseline_calm
        return f"[DigitalBlock calm={calm:.2f}] {user_text}"

    def periodic_update(
        self,
        outcome_score: float,
        target_style_adjustments: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Call this after a session/day with some scalar outcome score:
          >0 = good, <0 = bad.
        """
        if target_style_adjustments is None:
            target_style_adjustments = {}

        self.state.traits = self.trait_updater.update(
            traits=self.state.traits,
            events=self.state.events,
            outcome_score=outcome_score,
            target_style_adjustments=target_style_adjustments,
        )

        # Optionally trigger offline training of:
        #   - BlockStyleMapper
        #   - EmotionPolicyNet
        # using self.state.events as training data.

        self.state.events.clear()
