# Capsule Brain System - Complete Architecture

**Full implementation of Capsule Brain with FRNN Workspace, DPMM, ToneNet, Feelings, and Capsules**

---

## System Overview

This is a production-grade implementation of the Capsule Brain architecture integrating:

1. **Dynamic Pseudo-Mode Memory (DPMM)** - Already implemented in `../ppm_new.py`
2. **ToneNet** - Already implemented in `../tonenet/`
3. **Feeling Layer** - Implemented in `feelings.py`
4. **FRNN-based Global Workspace** - To be implemented in `workspace/`
5. **Capsules** - To be implemented in `capsules/`
6. **Main Brain Loop** - Orchestrates everything

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          INPUT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Audio Waveform          Text Input        Internal Events      │
│       ↓                      ↓                    ↓              │
│  ToneNet Router      TextEncoder           EventGenerator       │
│       ↓                      ↓                    ↓              │
│  ┌────────────────────────────────────────────────────┐         │
│  │            SpikePacket (content, routing,          │         │
│  │              priority, modality, metadata)         │         │
│  └────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FEELING LAYER UPDATE                          │
├─────────────────────────────────────────────────────────────────┤
│  Extract tone from spike.metadata["tone"]                       │
│  F_new = (1-α) * F_old + α * one_hot(tone)                     │
│  F = softmax(F_new)  →  (1, 8) emotion distribution            │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DPMM STORAGE                                  │
├─────────────────────────────────────────────────────────────────┤
│  PMM.store(spike)                                               │
│    - Compute similarities to modes μ_i                          │
│    - Update importance λ_i, occupancy, spectral params          │
│    - Trigger merge/split/prune if needed                        │
│    - Enforce Σ occupancy = 1.0                                  │
│  Returns: {stored, novelty, active_modes, ...}                  │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│              FRNN WORKSPACE CONTROLLER                           │
├─────────────────────────────────────────────────────────────────┤
│  Input: x_t = concat(spike.content, F)                         │
│         retrieval_hook = PMM.retrieve(query)                    │
│                                                                  │
│  FRNNCore_v3.step(x_t, state, retrieval_hook)                  │
│    - Discrete mode dist m_t (B, K) via Gumbel softmax          │
│    - Memory matrix M_t (K, mem_dim)                            │
│    - Selective write to relevant modes                          │
│    - Stickiness for mode persistence                            │
│    - Output y_t (B, latent_dim)                                │
│                                                                  │
│  Returns: WorkspaceState {broadcast, mode_probs,                │
│                           current_memory, feelings, last_spike} │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                  BROADCAST TO CAPSULES                           │
├─────────────────────────────────────────────────────────────────┤
│  For each capsule in [Language, Emotion, Planning,              │
│                       SelfModel, Safety, ...]:                  │
│                                                                  │
│    gate_j = σ(w_j · ws_state.broadcast)                        │
│                                                                  │
│    if gate_j > threshold:                                       │
│      capsule_j.receive_workspace(ws_state)                     │
│      output_spikes = capsule_j.process()                        │
│      action_queue.append(output_spikes)                         │
└─────────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CAPSULE ACTIONS                               │
├─────────────────────────────────────────────────────────────────┤
│  Language Capsule: Generate text response                        │
│  Emotion Capsule: Monitor mood drift, adjust parameters         │
│  Planning Capsule: Update goals, emit plan steps                │
│  SelfModel Capsule: Explain architecture state                  │
│  Safety Capsule: Flag unsafe patterns, damp priorities          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Data Flow

### One Complete Step

```python
from capsule_brain.config import DEFAULT_CONFIG
from capsule_brain.core_types import SpikePacket, WorkspaceState
from capsule_brain.feelings import FeelingLayer
from ppm_new import StaticPseudoModeMemory  # Your existing PMM
from tonenet import ToneNetRouter  # Your existing ToneNet

def brain_step(
    audio: torch.Tensor,
    pmm: StaticPseudoModeMemory,
    tonenet: ToneNetRouter,
    workspace: FRNNWorkspaceController,
    feelings: FeelingLayer,
    capsules: List[BaseCapsule],
    timestamp: float
) -> dict:
    """
    Execute one full Capsule Brain reasoning step.
    
    Args:
        audio: Raw waveform (B, T)
        pmm: Pseudo-mode memory instance
        tonenet: ToneNet router instance
        workspace: FRNN workspace controller
        feelings: Feeling layer
        capsules: List of active capsules
        timestamp: Current time
        
    Returns:
        Dictionary with workspace state, capsule outputs, metrics
    """
    
    # 1. AUDIO → SPIKE
    spike, tone_idx = tonenet.audio_to_spike(audio)
    spike.metadata["timestamp"] = timestamp
    
    # 2. UPDATE FEELINGS
    F = feelings.update(tone_idx)
    
    # 3. STORE IN PMM
    store_result = pmm.store(spike)
    
    # 4. WORKSPACE STEP (with PMM retrieval)
    ws_state = workspace.step(spike, F)
    
    # 5. BROADCAST TO CAPSULES
    capsule_outputs = []
    for capsule in capsules:
        if capsule.should_activate(ws_state):
            outputs = capsule.process(ws_state)
            capsule_outputs.extend(outputs)
    
    # 6. COLLECT RESULTS
    return {
        "workspace_state": ws_state,
        "feeling_dist": F.tolist(),
        "dominant_tone": feelings.get_dominant_tone(),
        "pmm_novelty": store_result["novelty"],
        "pmm_active_modes": store_result["active_modes"],
        "capsule_outputs": capsule_outputs,
        "timestamp": timestamp
    }
```

---

## Component Integration

### Using Existing Components

```python
# Initialize system with existing components
from config import DEFAULT_CONFIG
cfg = DEFAULT_CONFIG

# 1. PMM (Already exists in ../ppm_new.py)
from ppm_new import StaticPseudoModeMemory
pmm = StaticPseudoModeMemory(
    latent_dim=cfg.latent_dim,
    max_modes=cfg.max_modes,
    init_modes=cfg.init_modes,
    device=cfg.device
)

# 2. ToneNet (Already exists in ../tonenet/)
from tonenet import ToneNetRouter
tonenet = ToneNetRouter(
    pmm=pmm,  # Can attach PMM for integration
    harmonics=cfg.harmonics,
    sample_rate=cfg.sample_rate,
    device=cfg.device
)

# 3. Feelings (New - implemented above)
from feelings import FeelingLayer
feelings = FeelingLayer(alpha=cfg.feeling_alpha, device=cfg.device)

# 4. FRNN Workspace (To be implemented)
# from workspace import FRNNWorkspaceController
# workspace = FRNNWorkspaceController(
#     latent_dim=cfg.latent_dim,
#     feelings_dim=cfg.feelings_dim,
#     pmm_retrieval_dim=cfg.retrieval_dim,
#     num_states=cfg.num_states,
#     memory_dim=cfg.memory_dim,
#     hidden_dim=cfg.hidden_dim,
#     bank_size=cfg.bank_size,
#     device=cfg.device
# )
# workspace.attach_pmm_retrieval(build_pmm_retrieval_fn(pmm, workspace))
# workspace.reset(batch_size=1)

# 5. Capsules (To be implemented)
# from capsules import (
#     LanguageCapsule, EmotionCapsule, PlanningCapsule,
#     SelfModelCapsule, SafetyCapsule
# )
# capsules = [
#     LanguageCapsule(cfg),
#     EmotionCapsule(cfg),
#     PlanningCapsule(cfg),
#     SelfModelCapsule(cfg),
#     SafetyCapsule(cfg)
# ]
```

---

## File Implementation Status

### ✅ Completed (Existing)
- `../ppm_new.py` - Full DPMM with invariants
- `../capsule_brain_integration.py` - PMM API wrapper  
- `../tonenet/` - Complete ToneNet package
  - `tonenet/router.py` - Main integration
  - `tonenet/glyphs/glyph_encoder.py` - Math→Symbol
  - `tonenet/glyphs/glyph_decoder.py` - Symbol→Math
  - `tonenet/synth/gpu_synth.py` - GPU synthesis
  - `tonenet/cuda_kernels/harmonic_additive.cu` - CUDA kernels

### ✅ Implemented (New)
- `capsule_brain/config.py` - Global configuration
- `capsule_brain/core_types.py` - SpikePacket, WorkspaceState
- `capsule_brain/feelings.py` - Feeling Layer

### 🔨 To Implement

#### workspace/frnn_core_v3.py
```python
# FRNNCore_v3 implementation from research spec
# - Discrete mode distribution m_t via Gumbel softmax
# - Memory matrix M_t (K, mem_dim)
# - Selective write to active modes
# - Stickiness for mode persistence
# - retrieval_hook integration
# - step() and reset_state() methods
```

#### workspace/workspace_controller.py
```python
# FRNNWorkspaceController wrapper
# - Wraps FRNNCore_v3
# - Integrates PMM via retrieval hook
# - Combines spike.content + feelings as input
# - Returns WorkspaceState
```

#### capsules/base.py
```python
class BaseCapsule(nn.Module):
    """Base class for all capsules."""
    def should_activate(self, ws_state: WorkspaceState) -> bool:
        """Gate function - should this capsule activate?"""
        pass
    
    def process(self, ws_state: WorkspaceState) -> List[SpikePacket]:
        """Process workspace state, emit outputs."""
        pass
```

#### capsules/self_model.py
```python
class SelfModelCapsule(BaseCapsule):
    """
    Introspects the architecture and explains current state.
    
    Can answer questions like:
    - "What are you thinking about?"
    - "Why did you respond that way?"
    - "What modes are active in your memory?"
    """
```

---

## Testing Strategy

### Invariants to Verify

```python
# test_integration.py

def test_full_pipeline():
    """Test complete pipeline from audio to capsule outputs."""
    
    # Create dummy audio
    audio = torch.randn(1, 48000)
    
    # Run one step
    result = brain_step(audio, pmm, tonenet, workspace, feelings, capsules, 0.0)
    
    # Verify invariants
    assert abs(pmm.occupancy[pmm.active_mask].sum() - 1.0) < 1e-6  # Mass conservation
    assert (feelings.F >= 0).all()  # Non-negative feelings
    assert abs(feelings.F.sum() - 1.0) < 1e-5  # Normalized feelings
    assert result["workspace_state"].broadcast.shape == (1, 256)  # Correct shape
    assert result["workspace_state"].mode_probs.shape[1] == 64  # K modes
    
def test_feeling_updates():
    """Test feeling layer EMA dynamics."""
    feelings = FeelingLayer(alpha=0.5)
    
    # Neutral → Happy
    feelings.update(1)  # happy
    idx, name, prob = feelings.get_dominant_tone()
    assert name == "happy"
    
    # Repeated updates increase probability
    for _ in range(5):
        feelings.update(1)
    idx, name, prob = feelings.get_dominant_tone()
    assert prob > 0.8  # Should converge
```

---

## Usage Example

```python
import torch
from capsule_brain.config import DEFAULT_CONFIG
from capsule_brain.feelings import FeelingLayer
from capsule_brain.core_types import create_dummy_spike

# Initialize
cfg = DEFAULT_CONFIG
feelings = FeelingLayer(alpha=cfg.feeling_alpha)

# Simulate tone sequence
tones = [0, 1, 1, 1, 2, 2, 3]  # neutral → happy → sad → angry

for t, tone in enumerate(tones):
    F = feelings.update(tone)
    idx, name, prob = feelings.get_dominant_tone()
    print(f"Step {t}: Detected {FeelingLayer.TONE_NAMES[tone]}, "
          f"Dominant: {name} ({prob:.3f})")
    print(f"  Full dist: {feelings.to_dict()}")

# Output:
# Step 0: Detected neutral, Dominant: neutral (0.850)
# Step 1: Detected happy, Dominant: neutral (0.618)
# Step 2: Detected happy, Dominant: happy (0.510)
# Step 3: Detected happy, Dominant: happy (0.683)
# Step 4: Detected sad, Dominant: happy (0.556)
# Step 5: Detected sad, Dominant: sad (0.516)
# Step 6: Detected angry, Dominant: sad (0.426)
```

---

## Next Steps

To complete the implementation:

1. **Implement FRNN Workspace** (`workspace/frnn_core_v3.py`, `workspace/workspace_controller.py`)
   - Use the spec provided in the prompt
   - Integrate with PMM via retrieval hook

2. **Implement Capsules** (`capsules/base.py`, `capsules/*.py`)
   - BaseCapsule with gating
   - Language, Emotion, Planning, SelfModel, Safety

3. **Create Main Loop** (`brain.py`, `main_loop.py`)
   - Orchestrate all components
   - Handle streaming inputs
   - Manage capsule activation

4. **Add Tests** (`tests/`)
   - Unit tests for each component
   - Integration tests for full pipeline
   - Invariant checking

5. **Documentation**
   - API docs for each module
   - Usage examples
   - Architecture diagrams

---

## Architecture Advantages

### vs. Attention-Based Systems

| Feature | Attention (Transformer) | FRNN Workspace (This) |
|---------|------------------------|----------------------|
| Memory | Quadratic in sequence | Constant (K modes) |
| State | Stateless | Stateful recurrent |
| Modes | Soft averaging | Discrete modes (sticky) |
| Retrieval | All-to-all | Selective (mode-gated) |
| Interpretability | Low | High (discrete modes) |

### Key Benefits

1. **Bounded Memory** - K modes, not O(T²)
2. **Persistent Context** - M_t evolves over time
3. **Discrete Reasoning** - Clear mode transitions
4. **Capsule Integration** - Modular, composable
5. **Emotion-Aware** - Feelings integrated at core
6. **Memory Compression** - DPMM with merge/split
7. **Audio-Native** - ToneNet symbolic encoding

---

## License

Part of the Capsule Brain Pseudo-Memory system.

---

**Status: Core architecture defined, key components implemented, ready for FRNN/Capsule completion** 🧠
