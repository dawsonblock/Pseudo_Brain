# Capsule Brain - Build Instructions

## Status: Core Components Complete ✅

### Completed Files
1. ✅ `config.py` - Clean CapsuleBrainConfig with all hyperparameters
2. ✅ `core_types.py` - SpikePacket and WorkspaceState dataclasses
3. ✅ `feelings.py` - FeelingLayer with dataclass approach
4. ✅ `workspace/pmm_integration.py` - PMM retrieval hook builder

### Files Needing Updates

#### 5. `workspace/frnn_workspace.py` (rename/rewrite workspace_controller.py)

**Action**: Rewrite `workspace_controller.py` as clean `frnn_workspace.py`

**Requirements**:
- Import from `capsule_brain.config` (not sys.path hacks)
- Import from `capsule_brain.core_types`
- Reference existing `FRNNCore_v3` from `workspace.frnn_core_v3`
- Use ONLY fields from CapsuleBrainConfig (no gumbel_temp, etc.)
- Implement `attach_pmm_retrieval()`, `reset()`, `step()`, `get_state_summary()`

**Template**:
```python
# -*- coding: utf-8 -*-
from typing import Optional, Callable, Dict, Any
import torch
import torch.nn as nn

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import SpikePacket, WorkspaceState
from capsule_brain.workspace.frnn_core_v3 import FRNNCore_v3, FRNNConfig_v3


class FRNNWorkspaceController(nn.Module):
    def __init__(self, cfg: CapsuleBrainConfig):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        frnn_input_dim = cfg.latent_dim + cfg.feelings_dim
        
        frnn_cfg = FRNNConfig_v3(
            input_dim=frnn_input_dim,
            output_dim=cfg.latent_dim,
            num_states=cfg.num_states,
            memory_dim=cfg.memory_dim,
            hidden_dim=cfg.hidden_dim,
            gumbel_temp=1.0,  # Hardcode defaults not in config
            gumbel_hard=True,
            stickiness=0.1,
            selective_write=True,
            mlp_dropout=0.1,
            attention_bank_in_readout=(cfg.bank_size > 0),
            bank_size=cfg.bank_size,
            ema_decay=0.99,
            retrieval_dim=cfg.retrieval_dim,
        )
        
        self.frnn = FRNNCore_v3(frnn_cfg).to(self.device)
        self._state = None
        self._pmm_retrieval_fn: Optional[Callable] = None
    
    def attach_pmm_retrieval(self, fn: Callable):
        self._pmm_retrieval_fn = fn
    
    def reset(self, batch_size: int = 1):
        self._state = self.frnn.reset_state(batch_size, self.device)
    
    @torch.no_grad()
    def step(self, spike: SpikePacket, feelings: torch.Tensor) -> WorkspaceState:
        x = spike.content.to(self.device)
        F = feelings.to(self.device)
        x_t = torch.cat([x, F], dim=1)
        
        retrieval_hook = None
        if self._pmm_retrieval_fn:
            def _hook(x_local):
                return self._pmm_retrieval_fn(x_local)
            retrieval_hook = _hook
        
        y_t, self._state = self.frnn.step(x_t, self._state, retrieval_hook=retrieval_hook)
        
        probes = self.frnn.get_probes()
        m_t = probes.get("m_t", torch.full((x.shape[0], self.cfg.num_states), 
                                            1.0/self.cfg.num_states, device=self.device))
        current_memory = probes.get("current_memory", torch.zeros(x.shape[0], self.cfg.memory_dim, device=self.device))
        
        return WorkspaceState(
            broadcast=y_t,
            mode_probs=m_t,
            current_memory=current_memory,
            feelings=F,
            last_spike=spike,
        )
    
    def get_state_summary(self) -> Dict[str, Any]:
        if self._state is None:
            return {"initialized": False}
        probes = self.frnn.get_probes()
        m_t = probes.get("m_t")
        if m_t is None:
            return {"initialized": True, "mode_entropy": None}
        probs = m_t[0]
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        return {"initialized": True, "mode_entropy": entropy}
```

#### 6. `capsules/base.py`, `capsules/self_model.py`, `capsules/safety.py`

**Action**: Remove sys.path hacks, use clean imports

**Fix all capsule files**:
```python
# Remove:
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Replace with:
from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import WorkspaceState, SpikePacket
```

#### 7. `brain.py` - Main Orchestrator

**Create**: `/Users/dawsonblock/Pseudo_Brain/capsule_brain/brain.py`

**Use spec from corrected prompt - imports**:
```python
from typing import List, Dict, Any
import torch

from capsule_brain.config import CapsuleBrainConfig
from capsule_brain.core_types import SpikePacket, WorkspaceState
from capsule_brain.feelings import FeelingLayer
from capsule_brain.workspace.frnn_workspace import FRNNWorkspaceController
from capsule_brain.workspace.pmm_integration import build_pmm_retrieval_fn
from capsule_brain.capsules import BaseCapsule, SelfModelCapsule, SafetyCapsule

from ppm_new import StaticPseudoModeMemory
from tonenet import ToneNetRouter
```

Follow the CapsuleBrain class implementation from the corrected prompt exactly.

#### 8. `tests/test_integration.py` and `demo.py`

**Create** both files following the spec templates.

**Key**: Use clean imports everywhere:
```python
from capsule_brain.config import DEFAULT_CONFIG
from capsule_brain.brain import CapsuleBrain
```

---

## Quick Reference: Clean Import Patterns

### ✅ CORRECT
```python
from capsule_brain.config import CapsuleBrainConfig, DEFAULT_CONFIG
from capsule_brain.core_types import SpikePacket, WorkspaceState
from capsule_brain.feelings import FeelingLayer
from capsule_brain.workspace.frnn_workspace import FRNNWorkspaceController
from capsule_brain.workspace.pmm_integration import build_pmm_retrieval_fn
from capsule_brain.capsules import BaseCapsule
from ppm_new import StaticPseudoModeMemory
from tonenet import ToneNetRouter
```

### ❌ WRONG
```python
import sys
sys.path.insert(0, ...)  # NO!
from config import ...   # NO! (missing capsule_brain prefix)
```

---

## Testing Order

1. **Import test**: `python3 -c "from capsule_brain.config import DEFAULT_CONFIG; print(DEFAULT_CONFIG)"`
2. **Feelings test**: `python3 -c "from capsule_brain.feelings import FeelingLayer; f = FeelingLayer(); print(f.update(1))"`
3. **Full test**: `python3 -m capsule_brain.tests.test_integration`
4. **Demo**: `python3 -m capsule_brain.demo`

---

## Summary

**Completed**: config, core_types, feelings, pmm_integration  
**Need fixing**: workspace controller (rename to frnn_workspace.py)  
**Need clean imports**: all capsules files  
**Need creation**: brain.py, test, demo

**Total remaining effort**: ~2 hours to clean up existing files and add the 3 new ones.

All specifications are in the corrected prompt Mr Block provided.
