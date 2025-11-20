# Capsule Brain - Mathematical Foundations

**Detailed Mathematical Theory & Derivations**

---

## Table of Contents

1. [Pseudo-Mode Memory Theory](#1-pseudo-mode-memory)
2. [FRNN Workspace Theory](#2-frnn-workspace)
3. [Invariants & Constraints](#3-invariants)
4. [Gradient Flow Analysis](#4-gradients)
5. [Convergence Properties](#5-convergence)

---

## 1. Pseudo-Mode Memory Theory

### 1.1 Probabilistic Interpretation

**Memory as Distribution**: PMM represents memory as a mixture distribution over prototype locations:

```
p(x) = Σᵢ occupancy_i · δ(x - μᵢ)

where:
  Σᵢ occupancy_i = 1        (probability axiom)
  μᵢ ∈ ℝᴰ                   (prototype locations)
```

### 1.2 Reconstruction via Kernel Density

**Soft Assignment**: Instead of hard assignment to nearest prototype, use soft weighting:

```
α(x) = softmax(similarity(x, {μᵢ}))

where similarity can be:
  - Cosine: s(x, μ) = (x·μ)/(‖x‖·‖μ‖)
  - RBF: s(x, μ) = exp(-‖x-μ‖²/2σ²)
  - Dot product: s(x, μ) = x·μ
```

**Temperature-Scaled Softmax**:
```
αᵢ(x; τ) = exp(sᵢ/τ) / Σⱼ exp(sⱼ/τ)

Properties:
  - τ → 0: Hard assignment (argmax)
  - τ → ∞: Uniform distribution
  - τ = 1: Standard softmax
```

### 1.3 Loss Function Derivation

**Reconstruction Error**:
```
L_recon = 𝔼ₓ[‖x - x̂‖²]
        = 𝔼ₓ[‖x - Σᵢ αᵢ(x)·μᵢ‖²]
```

**Gradient w.r.t. Prototypes**:
```
∂L_recon/∂μⱼ = -𝔼ₓ[2·αⱼ(x)·(x - x̂)]
             = -2·𝔼ₓ[αⱼ(x)·residual(x)]
```

This is a **Hebbian-like update**: Prototypes move toward inputs they activate on, weighted by the reconstruction error.

**Sparsity Regularization**:
```
L_sparse = 𝔼ₓ[H(α(x))]
         = 𝔼ₓ[Σᵢ αᵢ(x)·log(αᵢ(x))]

Goal: Encourage sparse activations (few prototypes respond to each input)

Gradient:
∂L_sparse/∂μⱼ involves ∂αᵢ/∂μⱼ (implicit differentiation through softmax)
```

### 1.4 Occupancy Dynamics

**EMA Update**:
```
occupancy_i^(t+1) = ρ·occupancy_i^(t) + (1-ρ)·𝔼_batch[αᵢ(x)]

where:
  ρ ∈ [0,1]: Memory factor (typical: 0.99)
  𝔼_batch[αᵢ(x)]: Average activation over current batch
```

**Normalization** (enforces probability constraint):
```
occupancy_i ← occupancy_i / Σⱼ occupancy_j

Ensures: Σᵢ occupancy_i = 1.0 exactly
```

**Interpretation**: 
- High occupancy → Prototype represents frequently-seen region
- Low occupancy → Prototype underutilized (candidate for pruning)

### 1.5 Importance Dynamics

**Relevance Metric**:
```
relevance_i(x) = 1 - cos_sim(x, μᵢ)
               ∈ [0, 2]

High relevance → Poor reconstruction → Need more attention
```

**EMA Update**:
```
λᵢ^(t+1) = ρ·λᵢ^(t) + (1-ρ)·(1 - relevance_i(x))

Clamping: λᵢ ← max(λᵢ, λ_min) to prevent negatives
```

**Interpretation**:
- High λ → Good reconstruction quality → Keep prototype
- Low λ → Poor reconstruction → Candidate for split/removal

### 1.6 Structural Operations (Theory)

**Merge Criterion**:
```
Merge μᵢ and μⱼ if:
  1. cos_sim(μᵢ, μⱼ) > θ_merge    (close in space)
  2. Both active (occupancy > 0)

Merged prototype:
  μ_new = (λᵢ·μᵢ + λⱼ·μⱼ) / (λᵢ + λⱼ)    [importance-weighted average]
  λ_new = λᵢ + λⱼ
  occupancy_new = occupancy_i + occupancy_j
```

**Mathematical Justification**: This preserves the "center of mass" in latent space, weighted by importance.

**Split Criterion**:
```
Split μᵢ if:
  1. occupancy_i > θ_split_high    (overutilized)
  2. λᵢ < θ_split_low              (poor quality)

New prototypes:
  μᵢ' = μᵢ - ε·direction
  μⱼ = μᵢ + ε·direction
  
  where direction ~ 𝒩(0, I) or learned

  λᵢ' = λⱼ = λᵢ / 2
  occupancy_i' = occupancy_j = occupancy_i / 2
```

**Intuition**: High occupancy + low quality → Region needs finer resolution.

**Prune Criterion**:
```
Prune μᵢ if:
  occupancy_i < θ_prune

Action:
  - Set active_mask[i] = False
  - Set occupancy_i = 0
  - Redistribute mass to remaining modes (via normalization)
```

### 1.7 Spectral Parameters

**Temporal Dynamics** (future work):
```
Pseudomode evolution:
  μᵢ(t) = μᵢ(0) · exp(-γᵢ·t) · cos(ωᵢ·t + φᵢ)

where:
  γᵢ: Decay rate (memory fade)
  ωᵢ: Oscillation frequency (rhythmic recall)
  φᵢ: Phase offset
```

**Current Implementation**: Parameters stored but not yet used in dynamics.

---

## 2. FRNN Workspace Theory

### 2.1 Discrete State Space

**Finite State Machine with Soft Transitions**:
```
State space: S = {s₁, s₂, ..., sₖ}    (K discrete modes)

Soft state: m_t ∈ Δᴷ    (probability simplex)
  Σₖ m_t[k] = 1
  m_t[k] ≥ 0  ∀k
```

**Gumbel-Softmax Trick** (for differentiability):
```
Hard mode selection: k* = argmax(logits)    [not differentiable]

Soft relaxation:
  gₖ ~ Gumbel(0, 1)
  m_t[k] = exp((logits[k] + gₖ)/τ) / Σⱼ exp((logits[j] + gⱼ)/τ)

Properties:
  - Differentiable
  - τ → 0: Approaches hard selection
  - Allows gradient flow through discrete choice
```

### 2.2 Per-Mode Memory Banks

**Memory Tensor**:
```
M_t ∈ ℝᴷˣᴰ

M_t[k, :] = memory vector for mode k

Reading:
  h_t = Σₖ m_t[k] · M_t[k, :]    [weighted sum]
  
  Special cases:
    - If m_t = one_hot(k*): h_t = M_t[k*, :]    (hard selection)
    - If m_t = uniform: h_t = (1/K)·Σₖ M_t[k, :]    (average all)
```

### 2.3 Memory Update Dynamics

**Delta Computation**:
```
Δm_t = f_memory([x_t, h_t])

where f_memory is an MLP:
  f: ℝ^(input_dim + memory_dim) → ℝ^(memory_dim)
```

**Selective Write Gating**:
```
gate_t = σ(g([x_t]))    ∈ [0, 1]

Δm_t ← gate_t · Δm_t

Purpose: Don't update memory on every step (learn when to write)
```

**EMA Update** (per mode):
```
M_{t+1}[k, :] = ρ·M_t[k, :] + (1-ρ)·m_t[k]·Δm_t

Interpretation:
  - Mode k updates proportional to its activation m_t[k]
  - Inactive modes (m_t[k] ≈ 0) barely update
  - EMA prevents sudden memory changes
```

### 2.4 Stickiness (Temporal Coherence)

**Motivation**: Prevent rapid mode switching.

**Update Rule**:
```
m_t^(raw) = gumbel_softmax(MLP(x_t))    [initial selection]

m_t = (1-β)·m_t^(raw) + β·m_{t-1}      [blend with previous]

where β ∈ [0, 1] is stickiness factor
```

**Effect**:
- β = 0: No memory of previous mode (reactive)
- β = 1: Never change mode (stuck)
- β = 0.1 (typical): Smooth transitions, inertia

### 2.5 Attention Bank (Optional Context)

**Learnable Context Vectors**:
```
Bank ∈ ℝ^(B×D)    where B = bank_size

Attention over bank:
  scores = (h_t · Bank^T) / √D
  weights = softmax(scores)    ∈ ℝᴮ
  context = weights · Bank     ∈ ℝᴰ

Readout input: [h_t, context]  ∈ ℝ^(2D)
```

**Purpose**: Provides additional context beyond current memory, learned during training.

### 2.6 Readout Network

**CRITICAL**: Input dimension must match concatenation:
```
If using bank:
  readout_input = [h_t, context]
  dim(readout_input) = D + D = 2D

MLP: ℝ^(2D) → ℝᴴ → ℝᴰ_out

ERROR (original): Linear(D + bank_size, ...) assumed wrong concatenation
FIX: Linear(2D, ...) since context ∈ ℝᴰ
```

---

## 3. Invariants & Constraints

### 3.1 Mass Conservation (PMM)

**Mathematical Statement**:
```
∀t: Σᵢ∈Active occupancy_i(t) = 1.0
```

**Enforcement**:
```
After every update:
  total = Σᵢ occupancy_i
  if total > ε:
    occupancy_i ← occupancy_i / total
  else:
    occupancy_i ← 1 / n_active    [equal distribution]
```

**Why Critical**: Violating this breaks probabilistic interpretation and can cause numerical instability.

### 3.2 Non-Negativity (PMM)

**Constraints**:
```
λᵢ ≥ 0    (importance)
γᵢ ∈ [0, 1]    (decay rate)
ωᵢ ≥ 0    (frequency)
```

**Enforcement**:
```
After EMA update:
  λᵢ ← max(λᵢ, λ_min)    where λ_min = 1e-6
  γᵢ ← clip(γᵢ, 0, 1)
  ωᵢ ← max(ωᵢ, 0)
```

### 3.3 Simplex Constraints

**PMM Occupancy**:
```
occupancy ∈ Δᴷ    (K-simplex)
```

**FRNN Modes**:
```
m_t ∈ Δᴷ    (automatically satisfied by softmax)
```

**Feelings**:
```
F_t ∈ Δ⁸    (8-simplex for emotions)

Enforcement: F ← F / sum(F) after each update
```

### 3.4 Capacity Constraints

```
n_active_modes ≤ max_modes

Enforcement:
  - Prune underutilized modes if at capacity
  - Block splits if at capacity
```

---

## 4. Gradient Flow Analysis

### 4.1 Parameters vs Buffers

**Parameters** (receive gradients):
```
PMM:
  - μᵢ (prototypes)
  - w (weights - if used)

FRNN:
  - All MLP weights
  - Bank vectors

Tonenet:
  - Projection matrix
```

**Buffers** (no gradients, manual updates):
```
PMM:
  - λᵢ (importance) - updated via EMA
  - γᵢ, ωᵢ, φᵢ (spectral) - updated via EMA
  - occupancy - updated via EMA
  - active_mask - boolean, no gradients needed
```

**Why Separation?**:
- Prototypes (μ) trained via backprop
- Dynamics (λ, occupancy) updated online via statistics
- Prevents gradient interference with structural operations

### 4.2 Loss Gradients

**PMM Loss**:
```
L_total = L_recon + β·L_sparse

∂L/∂μⱼ = ∂L_recon/∂μⱼ + β·∂L_sparse/∂μⱼ

Components:
  ∂L_recon/∂μⱼ = -2·αⱼ·(x - x̂)
  
  ∂L_sparse/∂μⱼ involves ∂αₖ/∂μⱼ via chain rule through softmax
```

**FRNN Loss** (task-specific):
```
L_task = task_loss(readout(x_t), target)

Gradients flow:
  L → readout → memory_update → mode_selection → input
```

### 4.3 Gradient Clipping

**Recommended**:
```
torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)

Prevents:
  - Exploding gradients
  - Catastrophic forgetting
  - Unstable mode switches
```

---

## 5. Convergence Properties

### 5.1 PMM Convergence

**Theorem** (informal): Under EMA updates with normalization, occupancy converges to a stationary distribution.

**Proof Sketch**:
```
occupancy_i^(t+1) = ρ·occupancy_i^(t) + (1-ρ)·αᵢ

At equilibrium: occupancy_i^* = αᵢ^*

The normalization step projects onto simplex, ensuring:
  Σᵢ occupancy_i^* = 1
```

**Convergence Rate**: O(log(1/ε) / (1-ρ)) steps to reach ε-neighborhood.

### 5.2 Mode Stability

**Stable Configuration**:
```
- No two modes too similar (no merge triggers)
- No mode overutilized with low quality (no split triggers)
- No mode underutilized (no prune triggers)
```

**Attracting Set**: System evolves toward configurations where prototypes are:
1. Well-separated
2. Balanced occupancy
3. Good reconstruction quality

### 5.3 FRNN Memory Convergence

**Theorem**: Under fixed mode distribution m_t, memory banks converge exponentially.

**Proof**:
```
M_t[k] = ρ^t·M_0[k] + (1-ρ)·Σ_{s=0}^{t-1} ρ^s·m_s[k]·Δm_s

As t → ∞:
  M_∞[k] ∝ time-average of {m_s[k]·Δm_s}
```

**Convergence Rate**: τ = -1/log(ρ) steps (half-life).

For ρ = 0.99: τ ≈ 69 steps.

---

## 6. Information Theory Perspective

### 6.1 Entropy of Modes

**Mode Distribution Entropy**:
```
H(m_t) = -Σₖ m_t[k]·log(m_t[k])

Min: H = 0 (deterministic, one mode active)
Max: H = log(K) (uniform over K modes)
```

**Interpretation**: Measures uncertainty in mode selection.

### 6.2 Mutual Information

**Between Input and Mode**:
```
I(X; M) = H(M) - H(M|X)

High I(X; M) → Input strongly determines mode
Low I(X; M) → Mode selection independent of input
```

**Goal**: Learn mode selection that captures input structure.

### 6.3 Rate-Distortion Trade-off

**Compression**: PMM compresses input x ∈ ℝᴰ to mode index i ∈ [1, K].

**Rate**: log₂(K) bits

**Distortion**: 𝔼[‖x - x̂‖²]

**Trade-off**: More modes (higher rate) → Lower distortion.

---

## 7. Comparison to Other Architectures

### 7.1 vs Transformers

| Property | Transformer | FRNN |
|----------|-------------|------|
| Attention | O(T²) | O(K) discrete modes |
| Memory | Positional encoding | Explicit memory banks |
| Interpretability | Attention maps | Discrete mode probs |
| Scalability | Quadratic | Linear in modes |

### 7.2 vs Standard RNN

| Property | RNN | FRNN |
|----------|-----|------|
| State | Continuous h_t ∈ ℝᴰ | Discrete m_t ∈ Δᴷ |
| Capacity | Limited by hidden dim | K independent memories |
| Interpretability | Opaque | Clear mode semantics |

### 7.3 vs Vector Quantization (VQ-VAE)

| Property | VQ-VAE | PMM |
|----------|--------|-----|
| Codebook | Fixed K entries | Dynamic merge/split |
| Update | Hard assignment | Soft (differentiable) |
| Online | No | Yes (EMA updates) |

---

## Summary of Key Equations

**PMM Reconstruction**:
```
αᵢ = softmax(cos_sim(x, μᵢ) / τ)
x̂ = Σᵢ αᵢ · μᵢ
```

**PMM Dynamics**:
```
occupancy_i ← ρ·occ + (1-ρ)·mean(αᵢ)
occupancy ← occupancy / sum(occupancy)
```

**FRNN Mode**:
```
m_t = gumbel_softmax(MLP(x_t))
m_t ← (1-β)·m_t + β·m_{t-1}
```

**FRNN Memory**:
```
h_t = Σₖ m_t[k] · M_t[k]
M_{t+1}[k] = ρ·M_t[k] + (1-ρ)·m_t[k]·Δm_t
```

**Feelings**:
```
F_{t+1} = α·one_hot(tone) + (1-α)·F_t
F ← F / sum(F)
```

---

**For implementation details, see `CAPSULE_BRAIN_BUILD_GUIDE.md`**
