# -*- coding: utf-8 -*-
"""
FRNN Core v3 - Finite Recurrent Neural Network

Implements discrete-state recurrent workspace with:
- Gumbel-softmax mode selection
- Per-mode memory banks
- Selective write mechanism
- Stickiness for mode persistence
"""
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FRNNConfig_v3:
    """Configuration for FRNNCore_v3."""
    input_dim: int = 256
    output_dim: int = 256
    num_states: int = 64
    memory_dim: int = 256
    hidden_dim: int = 256
    gumbel_temp: float = 1.0
    gumbel_hard: bool = True
    stickiness: float = 0.1
    selective_write: bool = True
    mlp_dropout: float = 0.1
    attention_bank_in_readout: bool = True
    bank_size: int = 32
    ema_decay: float = 0.99
    retrieval_dim: int = 256


@dataclass
class FRNNState:
    """State container for FRNN."""
    m_t: torch.Tensor
    M_t: torch.Tensor
    prev_hidden: Optional[torch.Tensor] = None


class FRNNCore_v3(nn.Module):
    """
    Finite Recurrent Neural Network Core.

    Discrete mode selection with per-mode memory and selective updates.
    """

    def __init__(self, cfg: FRNNConfig_v3):
        super().__init__()
        self.cfg = cfg

        # Mode selection network
        self.mode_net = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.mlp_dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_states)
        )

        # Memory update network (per-mode)
        self.memory_net = nn.Sequential(
            nn.Linear(cfg.input_dim + cfg.memory_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.mlp_dropout),
            nn.Linear(cfg.hidden_dim, cfg.memory_dim)
        )

        # Retrieval integration (if PMM attached)
        if cfg.retrieval_dim > 0:
            self.retrieval_proj = nn.Linear(cfg.retrieval_dim, cfg.hidden_dim)
        else:
            self.retrieval_proj = None

        # Output readout network
        if cfg.attention_bank_in_readout and cfg.bank_size > 0:
            self.readout = nn.Sequential(
                nn.Linear(cfg.memory_dim + cfg.bank_size, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.output_dim)
            )
            # Attention bank (learnable context vectors)
            self.bank = nn.Parameter(
                torch.randn(cfg.bank_size, cfg.memory_dim) * 0.02
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(cfg.memory_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.output_dim)
            )
            self.bank = None

        # Write gate network (for selective write)
        if cfg.selective_write:
            self.write_gate = nn.Sequential(
                nn.Linear(cfg.input_dim, cfg.hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.hidden_dim, cfg.num_states),
                nn.Sigmoid()
            )
        else:
            self.write_gate = None

        # Probes for inspection
        self._probes = {}

    def reset_state(self, batch_size: int, device: torch.device) -> FRNNState:
        """Initialize FRNN state."""
        # Uniform mode distribution
        m_t = torch.ones(
            batch_size, self.cfg.num_states, device=device
        ) / self.cfg.num_states

        # Zero memory
        M_t = torch.zeros(
            batch_size, self.cfg.num_states, self.cfg.memory_dim,
            device=device
        )

        return FRNNState(m_t=m_t, M_t=M_t, prev_hidden=None)

    def step(
        self,
        x_t: torch.Tensor,
        state: FRNNState,
        retrieval_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, FRNNState]:
        """
        Execute one FRNN step.

        Args:
            x_t: Input (B, input_dim)
            state: Previous FRNN state
            retrieval_hook: Optional fn(x_t) -> retrieval_vec

        Returns:
            y_t: Output (B, output_dim)
            new_state: Updated FRNN state
        """
        B = x_t.shape[0]
        device = x_t.device

        # 1. MODE SELECTION with stickiness
        mode_logits = self.mode_net(x_t)

        # Add stickiness bias (favor previous mode)
        if self.cfg.stickiness > 0:
            prev_mode_logits = torch.log(state.m_t + 1e-8)
            mode_logits = mode_logits + self.cfg.stickiness * prev_mode_logits

        # Gumbel-softmax for discrete mode selection
        m_t = F.gumbel_softmax(
            mode_logits,
            tau=self.cfg.gumbel_temp,
            hard=self.cfg.gumbel_hard,
            dim=-1
        )  # (B, K)

        # 2. READ from memory (weighted by mode distribution)
        # current_memory = Σ_k m_t[k] * M_t[k]
        current_memory = torch.einsum('bk,bkd->bd', m_t, state.M_t)  # (B, D)

        # 3. RETRIEVAL from PMM (if hook provided)
        if retrieval_hook is not None and self.retrieval_proj is not None:
            retrieval_vec = retrieval_hook(x_t)  # (B, retrieval_dim)
            retrieval_feat = self.retrieval_proj(retrieval_vec)  # (B, hidden)
            # Add to memory representation
            current_memory = current_memory + retrieval_feat

        # 4. MEMORY UPDATE
        # Concat input + current memory
        update_input = torch.cat([x_t, current_memory], dim=-1)
        memory_delta = self.memory_net(update_input)  # (B, mem_dim)

        # 5. SELECTIVE WRITE
        if self.write_gate is not None:
            write_gates = self.write_gate(x_t)  # (B, K)
        else:
            write_gates = torch.ones(B, self.cfg.num_states, device=device)

        # Update memory: M_t[k] += m_t[k] * gate[k] * delta
        # Expand dims for broadcasting
        memory_delta_exp = memory_delta.unsqueeze(1)  # (B, 1, D)
        m_t_exp = m_t.unsqueeze(2)  # (B, K, 1)
        write_gates_exp = write_gates.unsqueeze(2)  # (B, K, 1)

        # Selective write with EMA
        alpha = 1.0 - self.cfg.ema_decay
        M_t_new = (
            self.cfg.ema_decay * state.M_t +
            alpha * m_t_exp * write_gates_exp * memory_delta_exp
        )

        # 6. OUTPUT READOUT
        if self.bank is not None:
            # Attention over bank
            # Q = current_memory, K = V = bank
            attn_scores = torch.matmul(
                current_memory,
                self.bank.T
            ) / (self.cfg.memory_dim ** 0.5)  # (B, bank_size)
            attn_weights = F.softmax(attn_scores, dim=-1)
            bank_context = torch.matmul(attn_weights, self.bank)  # (B, mem_dim)

            # Concat for readout
            readout_input = torch.cat([current_memory, bank_context], dim=-1)
        else:
            readout_input = current_memory

        y_t = self.readout(readout_input)  # (B, output_dim)

        # 7. UPDATE STATE
        new_state = FRNNState(
            m_t=m_t,
            M_t=M_t_new,
            prev_hidden=current_memory
        )

        # 8. STORE PROBES
        self._probes = {
            "m_t": m_t.detach(),
            "current_memory": current_memory.detach(),
            "mode_logits": mode_logits.detach(),
            "write_gates": write_gates.detach() if self.write_gate else None,
        }

        return y_t, new_state

    def get_probes(self) -> Dict[str, torch.Tensor]:
        """Return diagnostic probes from last step."""
        return self._probes
