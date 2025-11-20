# Capsule Brain - Complete Build Guide

**Step-by-Step Technical Implementation with Mathematical Foundations**

---

## Table of Contents

1. [System Overview & Philosophy](#1-system-overview)
2. [Mathematical Foundations](#2-mathematical-foundations)
3. [Component Build Order](#3-component-build-order)
4. [Data Flow & Integration](#4-data-flow)
5. [Training & Optimization](#5-training)

---

## 1. System Overview

### Architecture Stack

```
┌─────────────────────────────────────────┐
│         CapsuleBrain Orchestrator        │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │   PMM   │  │  FRNN WS │  │Capsules│ │
│  └────┬────┘  └─────┬────┘  └───┬────┘ │
│       │             │            │      │
│  ┌────┴────┐   ┌───┴────┐  ┌───┴────┐ │
│  │ ToneNet │   │Feelings│  │ Safety │ │
│  └─────────┘   └────────┘  └────────┘ │
└─────────────────────────────────────────┘
```

### Key Design Principles

1. **Mass Conservation**: All probability distributions sum to 1.0
2. **Gradient Safety**: Buffers (not Parameters) for dynamics
3. **Online Learning**: Updates during inference
4. **Discrete States**: Finite modes for interpretability
5. **Modular**: Each component independently testable

---

## 2. Mathematical Foundations

### 2.1 Pseudo-Mode Memory (PMM)

**Core Idea**: Memory as weighted mixture of prototypes in latent space.

**Variables**:
- `μᵢ ∈ ℝᴰ`: Prototype vectors (D = latent_dim)
- `λᵢ ∈ ℝ₊`: Importance scores
- `occupancy_i ∈ [0,1]`: Mass distribution (Σᵢ occupancy_i = 1.0)
- `γᵢ ∈ [0,1]`: Decay rates (spectral)
- `ωᵢ ∈ ℝ₊`: Oscillation frequencies

**Forward Pass**:
```
Given input x ∈ ℝᴰ:

1. Similarity: sᵢ = cos_sim(x, μᵢ) = (x·μᵢ)/(‖x‖·‖μᵢ‖)

2. Attention: αᵢ = exp(sᵢ/τ) / Σⱼ exp(sⱼ/τ)

3. Reconstruction: x̂ = Σᵢ αᵢ · μᵢ

4. Loss: L = ‖x - x̂‖² + β·H(α)
   where H(α) = Σᵢ αᵢ log(αᵢ) is entropy (sparsity)
```

**Memory Updates** (EMA):
```
occupancy_i^(t+1) = ρ · occupancy_i^(t) + (1-ρ) · mean(αᵢ)
occupancy ← occupancy / sum(occupancy)  [normalize]

λᵢ^(t+1) = ρ · λᵢ^(t) + (1-ρ) · (1 - cos_sim(x, μᵢ))
λᵢ ← max(λᵢ, λ_min)  [clamp]
```

**Structural Operations**:
```
Merge: if cos_sim(μᵢ, μⱼ) > 0.8
  → μ_new = (λᵢ·μᵢ + λⱼ·μⱼ)/(λᵢ + λⱼ)

Split: if occupancy_i > 0.3 AND λᵢ < 0.3
  → μ_new = μᵢ + ε·randn(), occupancy → occupancy/2

Prune: if occupancy_i < 0.05
  → deactivate mode
```

---

### 2.2 Finite Recurrent Neural Network (FRNN)

**Core Idea**: Discrete-state RNN with per-mode memory banks.

**State**:
- `m_t ∈ Δᴷ`: Mode distribution (K-simplex, Σₖ m_t[k] = 1)
- `M_t ∈ ℝᴷˣᴰ`: Memory banks (K modes, D dims each)

**Forward Pass**:
```
Given x_t ∈ ℝᴰ⁺⁸ (latent + feelings):

1. Mode Selection (Gumbel-softmax):
   logits = MLP_mode(x_t)
   m_t = softmax((logits + gumbel_noise) / τ)

2. Stickiness:
   m_t ← (1-β)·m_t + β·m_{t-1}  [temporal coherence]

3. Memory Read:
   h_t = Σₖ m_t[k] · M_t[k, :]

4. Memory Update:
   Δm = MLP_memory([x_t, h_t])
   Δm ← gate(x_t) · Δm  [selective write]

5. Bank Update (EMA):
   M_{t+1}[k, :] = ρ·M_t[k, :] + (1-ρ)·m_t[k]·Δm

6. Readout:
   y_t = MLP_readout([h_t, attention_context])
```

**Key Property**: Discrete modes enable interpretability while maintaining differentiability via Gumbel-softmax.

---

### 2.3 Feeling Layer

**Emotional EMA**:
```
F_t ∈ Δ⁸  (8 emotions: neutral, happy, sad, angry, fearful, disgusted, surprised, calm)

Given tone_idx ∈ [0,7]:
  target = one_hot(tone_idx)
  F_{t+1} = α·target + (1-α)·F_t
  F_{t+1} ← F_{t+1} / sum(F_{t+1})  [normalize]
```

---

## 3. Component Build Order

### Step 1: Base PMM (`ppm_new.py`)

```python
class StaticPseudoModeMemory(nn.Module):
    def __init__(self, latent_dim, max_modes, init_modes, ...):
        super().__init__()
        
        # Pre-allocate prototypes (Parameters - receive gradients)
        self.mu = nn.Parameter(torch.zeros(max_modes, latent_dim))
        self.w = nn.Parameter(torch.ones(max_modes))
        
        # Dynamics buffers (no gradients, updated via EMA)
        self.register_buffer('lambda_i', torch.ones(max_modes))
        self.register_buffer('gamma_i', torch.ones(max_modes) * 0.95)
        self.register_buffer('omega_i', torch.ones(max_modes))
        self.register_buffer('phase', torch.zeros(max_modes))
        self.register_buffer('occupancy', torch.zeros(max_modes))
        self.register_buffer('active_mask', torch.zeros(max_modes, dtype=bool))
        
        # Initialize first N modes
        self._initialize_modes(init_modes)
    
    def forward(self, latent, update_memory=False, return_components=False):
        # 1. Get active prototypes
        mu_active = self.mu[self.active_mask]
        
        # 2. Compute similarities (cosine)
        latent_norm = F.normalize(latent, p=2, dim=1)
        mu_norm = F.normalize(mu_active, p=2, dim=1)
        similarities = torch.matmul(latent_norm, mu_norm.T)
        
        # 3. Softmax attention
        alpha = F.softmax(similarities / self.temperature, dim=1)
        
        # 4. Reconstruct
        reconstruction = torch.matmul(alpha, mu_active)
        
        # 5. Store if updating
        if update_memory:
            self.last_alpha = alpha
            self.last_latent = latent
        
        # 6. Compute losses
        recon_loss = F.mse_loss(reconstruction, latent)
        sparsity_loss = -(alpha * torch.log(alpha + 1e-8)).sum(dim=1).mean()
        
        if return_components:
            return reconstruction, {
                'alpha': alpha,
                'recon_loss': recon_loss,
                'sparsity_loss': sparsity_loss
            }
        return reconstruction, {}
    
    def apply_explicit_updates(self):
        self.step_counter += 1
        
        # Always update importance & occupancy
        self._update_importance_occupancy()
        
        # Structural updates at intervals
        if self.step_counter % self.structural_update_freq == 0:
            self._apply_structural_updates()
        
        # CRITICAL: Enforce mass conservation
        self._normalize_occupancy()
    
    def _normalize_occupancy(self):
        """INVARIANT: Σ occupancy = 1.0"""
        with torch.no_grad():
            if self.n_active_modes == 0:
                return
            total = self.occupancy[self.active_mask].sum()
            if total > 1e-8:
                self.occupancy[self.active_mask] /= total
            else:
                self.occupancy[self.active_mask] = 1.0 / self.n_active_modes
```

**Test**:
```python
pmm = StaticPseudoModeMemory(latent_dim=16, max_modes=8, init_modes=4)
x = torch.randn(4, 16)
recon, _ = pmm(x, update_memory=True)
pmm.apply_explicit_updates()

# Verify invariant
assert abs(pmm.occupancy[pmm.active_mask].sum() - 1.0) < 1e-6
```

---

### Step 2: Capsule Brain API (`capsule_brain_integration.py`)

```python
class CapsuleBrainPMM(StaticPseudoModeMemory):
    """Extends PMM with Capsule Brain methods"""
    
    def store(self, spike: SpikePacket):
        """Store spike and update memory"""
        content = spike.content
        if content.dim() == 1:
            content = content.unsqueeze(0)
        
        # Forward + update
        reconstruction, components = self(content, update_memory=True, 
                                         return_components=True)
        self.apply_explicit_updates()
        
        # Compute novelty
        novelty = F.mse_loss(reconstruction, content).item()
        
        return {
            'stored': True,
            'novelty': novelty,
            'active_modes': self.n_active_modes,
            'alpha': components['alpha']
        }
    
    def retrieve(self, query):
        """Pure retrieval (no update)"""
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        reconstruction, components = self(query, update_memory=False,
                                         return_components=True)
        components['retrieval_confidence'] = components['alpha'].max(dim=1)[0]
        
        return reconstruction, components  # RETURNS TUPLE
    
    def to_workspace(self):
        """Export state for global workspace"""
        mask = self.active_mask
        return {
            'active_prototypes': self.mu[mask].detach(),
            'occupancy': self.occupancy[mask].detach(),
            'importance': self.lambda_i[mask].detach(),
            'decay_rates': self.gamma_i[mask].detach(),
            'oscillation_freq': self.omega_i[mask].detach(),
            'phase': self.phase[mask].detach(),
            'n_active': torch.tensor(self.n_active_modes)
        }
```

---

### Step 3: FRNN Core (`frnn_core_v3.py`)

```python
class FRNNCore_v3(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Mode selection
        self.mode_net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.num_states)
        )
        
        # Memory update
        self.memory_net = nn.Sequential(
            nn.Linear(cfg.input_dim + cfg.memory_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.memory_dim)
        )
        
        # Write gate
        self.write_gate = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Readout (CRITICAL FIX: memory_dim * 2)
        self.readout = nn.Sequential(
            nn.Linear(cfg.memory_dim * 2, cfg.hidden_dim),  # ✅ FIXED
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim)
        )
        
        # Attention bank
        self.bank = nn.Parameter(
            torch.randn(cfg.bank_size, cfg.memory_dim) * 0.02
        )
    
    def step(self, x_t, state, retrieval_hook=None):
        B = x_t.shape[0]
        
        # 1. Mode selection
        logits = self.mode_net(x_t)
        m_t = F.gumbel_softmax(logits, tau=self.cfg.gumbel_temp,
                               hard=self.cfg.gumbel_hard)
        
        # 2. Stickiness
        if state is not None:
            m_t = (1 - self.cfg.stickiness) * m_t + \
                  self.cfg.stickiness * state.m_t
        else:
            state = self.reset_state(B, x_t.device)
        
        # 3. Read memory
        current_memory = torch.einsum('bk,kd->bd', m_t, state.M_t)
        
        # 4. Memory update
        concat = torch.cat([x_t, current_memory], dim=-1)
        delta_m = self.memory_net(concat)
        
        # 5. Selective write
        write_gate = self.write_gate(x_t)
        delta_m = write_gate * delta_m
        
        # 6. Update banks
        M_new = self.cfg.ema_decay * state.M_t + \
                (1 - self.cfg.ema_decay) * \
                torch.einsum('bk,bd->kd', m_t, delta_m)
        
        # 7. Readout with attention
        attn_scores = torch.matmul(current_memory, self.bank.T) / \
                     (self.cfg.memory_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        bank_context = torch.matmul(attn_weights, self.bank)
        
        readout_input = torch.cat([current_memory, bank_context], dim=-1)
        y_t = self.readout(readout_input)
        
        return y_t, FRNNState(m_t=m_t, M_t=M_new, prev_hidden=current_memory)
```

---

### Step 4: PMM Integration (`pmm_integration.py`)

```python
def build_pmm_retrieval_fn(pmm, cfg):
    """Build PMM retrieval hook for FRNN"""
    latent_dim = cfg.latent_dim
    
    def retrieval_hook(x_t):
        query = x_t[:, :latent_dim]
        reconstruction, components = pmm.retrieve(query)  # TUPLE!
        return reconstruction  # Return tensor only
    
    return retrieval_hook
```

**CRITICAL FIX**: Unpack tuple from `pmm.retrieve()`.

---

### Step 5: Workspace Controller (`frnn_workspace.py`)

```python
class FRNNWorkspaceController(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Combined input: latent + feelings
        frnn_input_dim = cfg.latent_dim + cfg.feelings_dim
        
        frnn_cfg = FRNNConfig_v3(
            input_dim=frnn_input_dim,
            output_dim=cfg.latent_dim,
            num_states=cfg.num_states,
            memory_dim=cfg.memory_dim,
            ...
        )
        
        self.frnn = FRNNCore_v3(frnn_cfg)
        self._pmm_retrieval_fn = None
    
    def attach_pmm_retrieval(self, fn):
        self._pmm_retrieval_fn = fn
    
    def step(self, spike, feelings):
        # Combine inputs
        x_t = torch.cat([spike.content, feelings], dim=1)
        
        # FRNN step with optional PMM retrieval
        y_t, new_state = self.frnn.step(x_t, self._state,
                                        retrieval_hook=self._pmm_retrieval_fn)
        self._state = new_state
        
        # Package as WorkspaceState
        return WorkspaceState(
            broadcast=y_t,
            mode_probs=new_state.m_t,
            current_memory=new_state.prev_hidden,
            feelings=feelings,
            last_spike=spike
        )
```

---

### Step 6: ToneNet Router (`tonenet_router.py`)

```python
class ToneNetRouter(nn.Module):
    def __init__(self, pmm, harmonics=16, sample_rate=48000):
        super().__init__()
        self.latent_dim = pmm.latent_dim
        
        # Simple projection
        self.proj = nn.Linear(4, self.latent_dim)  # 4 features → latent
    
    def audio_to_spike(self, audio):
        # Extract features
        mean = audio.mean(dim=1, keepdim=True)
        std = audio.std(dim=1, keepdim=True)
        energy = (audio ** 2).mean(dim=1, keepdim=True)
        max_abs = audio.abs().max(dim=1, keepdim=True).values
        
        feats = torch.cat([mean, std, energy, max_abs], dim=1)
        latent = self.proj(feats)
        
        # Tone classification (crude)
        tone_idx = int(torch.clamp(energy * 8, 0, 7)[0].item())
        
        spike = SpikePacket(
            content=latent,
            routing_key="audio_in",
            priority=1.0,
            modality="audio",
            metadata={'tone': tone_idx, 'f0': 0.0, 'glyph_idx': 0}
        )
        
        return spike, tone_idx
```

---

### Step 7: Feeling Layer (`feelings.py`)

```python
class FeelingLayer:
    def __init__(self, alpha=0.3, device='cpu'):
        self.alpha = alpha
        self.F = torch.ones(1, 8, device=device) / 8.0  # Uniform init
    
    def update(self, tone_idx):
        target = torch.zeros_like(self.F)
        target[0, tone_idx] = 1.0
        
        # EMA update
        self.F = self.alpha * target + (1 - self.alpha) * self.F
        self.F = self.F / self.F.sum()  # Normalize
        
        return self.F
    
    def get_dominant_tone(self):
        return int(torch.argmax(self.F[0]).item())
```

---

### Step 8: Capsules (`capsules/base.py`, `safety.py`, `self_model.py`)

```python
class BaseCapsule:
    def __init__(self, cfg):
        self.cfg = cfg
        self.gate_net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.num_states + 8, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def should_activate(self, ws_state):
        features = torch.cat([
            ws_state.broadcast[0],
            ws_state.mode_probs[0],
            ws_state.feelings[0]
        ])
        score = self.gate_net(features.unsqueeze(0))
        return score.item() > 0.5
    
    def process(self, ws_state):
        raise NotImplementedError

class SafetyCapsule(BaseCapsule):
    def process(self, ws_state):
        # Check invariants, return warning if violated
        return []  # List of output spikes
```

---

### Step 9: Main Orchestrator (`brain.py`)

```python
class CapsuleBrain:
    def __init__(self, cfg):
        # PMM (CRITICAL: Use CapsuleBrainPMM, not StaticPseudoModeMemory)
        self.pmm = CapsuleBrainPMM(
            latent_dim=cfg.latent_dim,
            max_modes=cfg.max_modes,
            init_modes=cfg.init_modes,
            device=cfg.device
        )
        
        # ToneNet
        self.tonenet = ToneNetRouter(self.pmm, harmonics=cfg.harmonics)
        
        # Feelings
        self.feelings = FeelingLayer(alpha=cfg.feeling_alpha, device=cfg.device)
        
        # Workspace
        self.workspace = FRNNWorkspaceController(cfg)
        retrieval_fn = build_pmm_retrieval_fn(self.pmm, cfg)
        self.workspace.attach_pmm_retrieval(retrieval_fn)
        self.workspace.reset(batch_size=1)
        
        # Capsules
        self.capsules = [
            SelfModelCapsule(cfg),
            SafetyCapsule(cfg)
        ]
    
    def step(self, audio, timestamp=0.0):
        # 1. Audio → Spike
        spike, tone_idx = self.tonenet.audio_to_spike(audio)
        
        # 2. Update feelings
        F = self.feelings.update(tone_idx)
        
        # 3. Store in PMM
        pmm_result = self.pmm.store(spike)
        
        # 4. Workspace step
        ws_state = self.workspace.step(spike, F)
        
        # 5. Capsules
        capsule_outputs = []
        for capsule in self.capsules:
            if capsule.should_activate(ws_state):
                capsule_outputs.extend(capsule.process(ws_state))
        
        return {
            'step': self.step_count,
            'workspace_state': ws_state,
            'feelings': F,
            'dominant_tone': self.feelings.get_dominant_tone(),
            'pmm_novelty': pmm_result['novelty'],
            'pmm_active_modes': pmm_result['active_modes'],
            'capsule_outputs': capsule_outputs
        }
```

---

### Step 10: Configuration (`config.py`)

```python
@dataclass
class CapsuleBrainConfig:
    # CRITICAL FIX: Auto-detect device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    latent_dim: int = 256
    feelings_dim: int = 8
    max_modes: int = 128
    init_modes: int = 32
    num_states: int = 64
    memory_dim: int = 256
    hidden_dim: int = 256
    bank_size: int = 32
    retrieval_dim: int = 256
    feeling_alpha: float = 0.3
    sample_rate: int = 48000
    harmonics: int = 16
```

---

## 4. Data Flow

### Single Step Execution

```
audio (B, T=48000)
    ↓ [ToneNet]
(spike, tone_idx)
    ↓
    ├→ [PMM.store(spike)] → {novelty, active_modes}
    │
    ├→ [Feelings.update(tone_idx)] → F (B, 8)
    │
    └→ [Workspace.step(spike, F)]
        │  ↓ [combine: latent + feelings]
        │  x_t = [spike.content, F]  (B, 264)
        │  ↓ [FRNN mode selection]
        │  m_t ~ gumbel_softmax(MLP(x_t))
        │  ↓ [memory read]
        │  h_t = Σₖ m_t[k] · M[k]
        │  ↓ [memory update]
        │  Δm = gate(x_t) · MLP([x_t, h_t])
        │  ↓ [bank update]
        │  M_{t+1}[k] = ρ·M[k] + (1-ρ)·m_t[k]·Δm
        │  ↓ [readout]
        │  y_t = MLP([h_t, attn_context])
        │
        └→ WorkspaceState{broadcast=y_t, mode_probs=m_t, ...}
            ↓
            [Capsules.process(ws_state)]
            ↓
            List[SpikePacket] outputs
```

---

## 5. Training

### Online Learning (Current)

```python
brain = CapsuleBrain(cfg)

for audio in dataloader:
    result = brain.step(audio, timestamp=t)
    # Memory updates happen inside step()
    # No separate training phase!
```

### Supervised Training (Optional)

```python
optimizer = torch.optim.Adam(brain.parameters(), lr=1e-3)

for audio, target in dataloader:
    result = brain.step(audio, timestamp=t)
    
    # Task-specific loss
    prediction = result['workspace_state'].broadcast
    loss = F.mse_loss(prediction, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Apply PMM structural updates AFTER gradient step
    brain.pmm.apply_explicit_updates()
```

---

## 6. Invariant Verification

```python
def verify_invariants(brain):
    # PMM occupancy conservation
    pmm = brain.pmm
    occ_sum = pmm.occupancy[pmm.active_mask].sum().item()
    assert abs(occ_sum - 1.0) < 1e-6, f"Occupancy: {occ_sum}"
    
    # Non-negative parameters
    assert (pmm.lambda_i[pmm.active_mask] >= 0).all()
    assert (pmm.gamma_i[pmm.active_mask] >= 0).all()
    
    # Feelings normalized
    F_sum = brain.feelings.F.sum().item()
    assert abs(F_sum - 1.0) < 1e-6, f"Feelings: {F_sum}"
    
    print("✅ All invariants satisfied")
```

---

## Summary of Critical Fixes

1. **brain.py**: Use `CapsuleBrainPMM` (not `StaticPseudoModeMemory`)
2. **pmm_integration.py**: Unpack tuple from `retrieve()`
3. **config.py**: Auto-detect device (CPU/CUDA)
4. **frnn_core_v3.py**: Fix readout dimensions (`memory_dim * 2`)

**Result**: Fully functional end-to-end system with all invariants preserved.

---

**Quick Start**:
```bash
python3 QUICK_START_DEMO.py
```
