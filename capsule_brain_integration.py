"""
Capsule Brain Integration Layer for Pseudo-Memory Module
========================================================

Provides spike packet format and Capsule Brain API extensions.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

# Import the corrected PMM
from ppm_new import StaticPseudoModeMemory


@dataclass
class SpikePacket:
    """Capsule Brain spike packet format"""
    content: torch.Tensor  # [B, D] latent representation
    routing_key: Optional[str] = None
    priority: float = 0.5
    timestamp: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class CapsuleBrainPMM(StaticPseudoModeMemory):
    """
    Pseudo-Memory Module with full Capsule Brain integration.
    
    Required API methods:
    - store(spike)
    - retrieve(query)
    - compress()
    - merge_modes()
    - split_modes()
    - route_to_capsule(id)
    - to_workspace()
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # =========================================================================
    # CAPSULE BRAIN API
    # =========================================================================
    
    def store(self, spike: SpikePacket) -> Dict[str, Any]:
        """
        Capsule Brain API: Store spike packet in memory.
        
        Args:
            spike: SpikePacket with content tensor
            
        Returns:
            Storage diagnostics (mode activations, novelty score, etc.)
        """
        content = spike.content
        if content.dim() == 1:
            content = content.unsqueeze(0)
        
        # Forward with memory update
        reconstruction, components = self(content, update_memory=True, return_components=True)
        
        # Apply explicit updates immediately for online learning
        self.apply_explicit_updates()
        
        # Compute novelty (reconstruction error)
        novelty = F.mse_loss(reconstruction, content).item()
        
        return {
            'stored': True,
            'novelty': novelty,
            'active_modes': self.n_active_modes,
            'alpha': components['alpha'],
            'routing_key': spike.routing_key
        }
    
    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Capsule Brain API: Retrieve memory given query.
        
        Args:
            query: Query tensor [B, D] or [D]
            
        Returns:
            reconstruction: Retrieved memory [B, D]
            components: Retrieval diagnostics
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Forward without memory update (pure retrieval)
        reconstruction, components = self(query, update_memory=False, return_components=True)
        
        # Add retrieval-specific metrics
        components['retrieval_confidence'] = components['alpha'].max(dim=1)[0]
        
        return reconstruction, components
    
    def compress(self) -> Dict[str, Any]:
        """
        Capsule Brain API: Compress memory by forcing structural updates.
        
        Returns:
            Compression statistics
        """
        modes_before = self.n_active_modes
        
        with torch.no_grad():
            self._apply_structural_updates()
            self._normalize_occupancy()
        
        modes_after = self.n_active_modes
        
        return {
            'modes_before': modes_before,
            'modes_after': modes_after,
            'compression_ratio': modes_before / max(modes_after, 1),
            'occupancy_sum': self.occupancy[self.active_mask].sum().item()
        }
    
    def merge_modes(self, force: bool = False) -> int:
        """
        Capsule Brain API: Explicitly trigger mode merging.
        
        Args:
            force: If True, use lower threshold for aggressive merging
            
        Returns:
            Number of modes merged
        """
        original_threshold = self.merge_threshold
        if force:
            self.merge_threshold = 0.6  # More aggressive
        
        modes_before = self.n_active_modes
        
        with torch.no_grad():
            self._merge_similar_modes()
            self._normalize_occupancy()
        
        self.merge_threshold = original_threshold
        
        return modes_before - self.n_active_modes
    
    def split_modes(self, force: bool = False) -> int:
        """
        Capsule Brain API: Explicitly trigger mode splitting.
        
        Args:
            force: If True, use lower threshold for aggressive splitting
            
        Returns:
            Number of new modes created
        """
        original_threshold = self.split_threshold
        if force:
            self.split_threshold = 0.1  # More aggressive
        
        modes_before = self.n_active_modes
        
        with torch.no_grad():
            self._split_weak_modes()
            self._normalize_occupancy()
        
        self.split_threshold = original_threshold
        
        return self.n_active_modes - modes_before
    
    def route_to_capsule(self, capsule_id: str, content: torch.Tensor) -> SpikePacket:
        """
        Capsule Brain API: Route memory content to target capsule.
        
        Args:
            capsule_id: Target capsule identifier
            content: Content to route [B, D]
            
        Returns:
            SpikePacket ready for capsule routing
        """
        # Retrieve and encode
        reconstruction, components = self.retrieve(content)
        
        # Create spike packet with routing metadata
        spike = SpikePacket(
            content=reconstruction,
            routing_key=capsule_id,
            priority=components['retrieval_confidence'].mean().item(),
            timestamp=float(self.step_counter),
            metadata={
                'source': 'pseudo_memory',
                'n_modes': self.n_active_modes,
                'alpha': components['alpha']
            }
        )
        
        return spike
    
    def to_workspace(self) -> Dict[str, torch.Tensor]:
        """
        Capsule Brain API: Broadcast memory state to global workspace.
        
        Returns:
            Dictionary of memory state tensors for workspace
        """
        if self.n_active_modes == 0:
            return {
                'active_prototypes': torch.zeros(1, self.latent_dim, device=self.mu.device),
                'occupancy': torch.zeros(1, device=self.mu.device),
                'importance': torch.zeros(1, device=self.mu.device)
            }
        
        active_mask = self.active_mask
        
        return {
            'active_prototypes': self.mu[active_mask].detach(),
            'occupancy': self.occupancy[active_mask].detach(),
            'importance': self.lambda_i[active_mask].detach(),
            'decay_rates': self.gamma_i[active_mask].detach(),
            'oscillation_freq': self.omega_i[active_mask].detach(),
            'phase': self.phase[active_mask].detach(),
            'n_active': torch.tensor(self.n_active_modes, device=self.mu.device),
            'step': torch.tensor(self.step_counter, device=self.mu.device)
        }


# ============================================================================
# SYMBOLIC COMPRESSION & SELF-REWIRING COMPATIBILITY
# ============================================================================

class SymbolicCompressionReactor:
    """
    Symbolic compression reactor for Capsule Brain.
    Compresses pseudo-memory representations into symbolic tokens.
    """
    
    def __init__(self, pmm: CapsuleBrainPMM, symbol_dim: int = 32):
        self.pmm = pmm
        self.symbol_dim = symbol_dim
        
        # Learnable compression matrix: latent_dim -> symbol_dim
        self.compress_matrix = torch.nn.Linear(pmm.latent_dim, symbol_dim)
        self.decompress_matrix = torch.nn.Linear(symbol_dim, pmm.latent_dim)
    
    def compress_to_symbols(self, latent: torch.Tensor) -> torch.Tensor:
        """Compress latent to symbolic representation"""
        symbols = self.compress_matrix(latent)
        return torch.tanh(symbols)  # Bound symbols to [-1, 1]
    
    def decompress_from_symbols(self, symbols: torch.Tensor) -> torch.Tensor:
        """Reconstruct latent from symbols"""
        latent = self.decompress_matrix(symbols)
        return latent
    
    def compress_memory_to_symbols(self) -> Dict[str, torch.Tensor]:
        """Compress entire memory into symbolic tokens"""
        workspace = self.pmm.to_workspace()
        prototypes = workspace['active_prototypes']
        
        if prototypes.numel() == 0:
            return {'symbols': torch.zeros(1, self.symbol_dim)}
        
        symbols = self.compress_to_symbols(prototypes)
        
        return {
            'symbols': symbols,
            'occupancy': workspace['occupancy'],
            'n_symbols': torch.tensor(len(symbols))
        }


class SelfRewiringEngine:
    """
    Self-rewiring engine for Capsule Brain PMM.
    Allows dynamic modification of mode structure based on task performance.
    """
    
    def __init__(self, pmm: CapsuleBrainPMM):
        self.pmm = pmm
        self.rewiring_history = []
    
    def rewire_based_on_feedback(self, performance_score: float):
        """
        Adjust mode structure based on performance feedback.
        
        Args:
            performance_score: 0-1, higher is better
        """
        if performance_score < 0.5:
            # Poor performance → increase capacity (split modes)
            n_split = self.pmm.split_modes(force=True)
            self.rewiring_history.append({
                'action': 'split',
                'count': n_split,
                'performance': performance_score
            })
        elif performance_score > 0.8 and self.pmm.n_active_modes > 4:
            # Excellent performance → compress (merge modes)
            n_merged = self.pmm.merge_modes(force=True)
            self.rewiring_history.append({
                'action': 'merge',
                'count': n_merged,
                'performance': performance_score
            })
    
    def get_rewiring_stats(self) -> Dict[str, Any]:
        """Get statistics on rewiring operations"""
        if not self.rewiring_history:
            return {'total_rewirings': 0}
        
        return {
            'total_rewirings': len(self.rewiring_history),
            'splits': sum(1 for h in self.rewiring_history if h['action'] == 'split'),
            'merges': sum(1 for h in self.rewiring_history if h['action'] == 'merge'),
            'history': self.rewiring_history[-10:]  # Last 10
        }


# ============================================================================
# EXTERNAL TUTOR BRIDGE
# ============================================================================

class ExternalTutorBridge:
    """
    Bridge to external AI tutors (GPT → Sonnet → Mistral chain).
    Provides teacher-forcing signals for pseudo-memory training.
    """
    
    def __init__(self, pmm: CapsuleBrainPMM):
        self.pmm = pmm
        self.tutor_feedback_buffer = []
    
    def receive_tutor_feedback(
        self,
        student_latent: torch.Tensor,
        teacher_latent: torch.Tensor,
        feedback_weight: float = 0.5
    ):
        """
        Receive feedback from external tutor and guide memory formation.
        
        Args:
            student_latent: Current PMM reconstruction
            teacher_latent: Ground truth from external tutor
            feedback_weight: How much to weight teacher signal
        """
        # Blend student and teacher latents
        guided_latent = (
            (1 - feedback_weight) * student_latent +
            feedback_weight * teacher_latent
        )
        
        # Store guided latent as a "corrected" memory
        spike = SpikePacket(
            content=guided_latent,
            routing_key='tutor_corrected',
            priority=1.0,  # High priority
            metadata={'source': 'external_tutor', 'feedback_weight': feedback_weight}
        )
        
        result = self.pmm.store(spike)
        self.tutor_feedback_buffer.append(result)
        
        return result
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get statistics on tutor feedback"""
        if not self.tutor_feedback_buffer:
            return {'total_feedback': 0}
        
        novelties = [fb['novelty'] for fb in self.tutor_feedback_buffer]
        
        return {
            'total_feedback': len(self.tutor_feedback_buffer),
            'avg_novelty': sum(novelties) / len(novelties),
            'recent_feedback': self.tutor_feedback_buffer[-5:]
        }
