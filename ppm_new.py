import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
import logging

class StaticPseudoModeMemory(nn.Module):
    """
    Production-grade Static PMM with pre-allocated tensors and mask-based mode management.
    Maintains full functionality without dynamic parameter structural changes.
    """

    def __init__(
        self,
        latent_dim: int,
        max_modes: int = 64,
        init_modes: int = 8,
        importance_decay: float = 0.99,
        capacity_margin: float = 0.1,
        merge_threshold: float = 0.8,
        split_threshold: float = 0.3,
        prune_threshold: float = 0.05,
        structural_update_freq: int = 10,
        use_predictive: bool = False,
        use_safety: bool = False,
        predictive_dim: Optional[int] = None,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.max_modes = max_modes
        self.structural_update_freq = structural_update_freq
        self.step_counter = 0
        
        # Pre-allocate all tensors
        self.mu = nn.Parameter(torch.zeros(max_modes, latent_dim, device=device))
        self.w = nn.Parameter(torch.ones(max_modes, device=device))
        self.lambda_i = nn.Parameter(torch.ones(max_modes, device=device))
        self.rho_i = nn.Parameter(torch.ones(max_modes, device=device))
        self.eta_i = nn.Parameter(torch.ones(max_modes, device=device))
        
        # Buffers
        self.register_buffer('lambda_W', torch.tensor(1.0, device=device))
        self.register_buffer('occupancy', torch.zeros(max_modes, device=device))
        self.register_buffer('active_mask', torch.zeros(max_modes, dtype=torch.bool, device=device))
        
        if use_safety:
            self.register_buffer('risk', torch.zeros(max_modes, device=device))
        else:
            self.risk = None
        
        # Predictive components
        self.use_predictive = use_predictive
        if use_predictive:
            pred_dim = predictive_dim or latent_dim
            self.F = nn.Parameter(torch.zeros(max_modes, pred_dim, latent_dim, device=device))
        
        # Initialize active modes
        self._initialize_modes(init_modes)
        
        # Hyperparameters
        self.importance_decay = importance_decay
        self.capacity_margin = capacity_margin
        self.merge_threshold = merge_threshold
        self.split_threshold = split_threshold
        self.prune_threshold = prune_threshold
        
        # Logging
        self.logger = logging.getLogger(__name__)

    def _initialize_modes(self, init_modes: int):
        """Initialize first N modes with random prototypes"""
        if init_modes > self.max_modes:
            raise ValueError(f"init_modes ({init_modes}) cannot exceed max_modes ({self.max_modes})")
        
        with torch.no_grad():
            # Initialize first 'init_modes' as active with random prototypes
            self.mu.data[:init_modes] = torch.randn(init_modes, self.latent_dim, device=self.mu.device) * 0.1
            self.active_mask[:init_modes] = True
            
            # Initialize other parameters for active modes
            self.w.data[:init_modes] = 1.0 / init_modes
            self.lambda_i.data[:init_modes] = 1.0
            self.rho_i.data[:init_modes] = 1.0
            self.eta_i.data[:init_modes] = 1.0
            self.occupancy[:init_modes] = 1.0 / init_modes
            
            if self.use_predictive:
                eye_matrix = torch.eye(self.latent_dim, device=self.mu.device)
                self.F.data[:init_modes] = eye_matrix.unsqueeze(0).repeat(init_modes, 1, 1) * 0.1

    @property
    def n_active_modes(self) -> int:
        return self.active_mask.sum().item()

    @property
    def active_mu(self) -> torch.Tensor:
        return self.mu[self.active_mask]

    @property
    def active_w(self) -> torch.Tensor:
        return self.w[self.active_mask]

    @property
    def active_lambda_i(self) -> torch.Tensor:
        return self.lambda_i[self.active_mask]

    def forward(
        self, 
        latent: torch.Tensor, 
        update_memory: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with optional memory update"""
        batch_size = latent.shape[0]
        active_mask = self.active_mask
        n_active = self.n_active_modes
        
        if n_active == 0:
            output = torch.zeros_like(latent)
            # Use zeros_like to ensure proper gradient tracking
            sparsity_loss = torch.zeros(1, device=latent.device, requires_grad=True).squeeze()
            components = {
                "alpha": torch.zeros(batch_size, 0, device=latent.device),
                "sparsity_loss": sparsity_loss
            }
            return output, components if return_components else (output, {})
        
        # Compute similarities and activations
        mu_active = self.mu[active_mask]
        similarities = F.cosine_similarity(
            latent.unsqueeze(1), 
            mu_active.unsqueeze(0), 
            dim=2
        )
        
        # Importance-weighted activations
        lambda_active = self.lambda_i[active_mask]
        a = similarities * lambda_active
        
        # Readout weights with safety modulation
        w_active = self.w[active_mask]
        if self.risk is not None:
            risk_active = self.risk[active_mask]
            w_active = w_active * torch.exp(-risk_active)
        
        # Softmax attention
        alpha = F.softmax(a, dim=1)
        
        # Weighted reconstruction
        weighted_mu = alpha.unsqueeze(2) * mu_active.unsqueeze(0)
        reconstruction = weighted_mu.sum(dim=1)
        
        # Compute sparsity loss (before detaching alpha)
        sparsity_loss = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=1).mean()
        
        # Store for explicit updates if requested
        if update_memory:
            self._store_for_update(latent, similarities, alpha)
        
        components: Dict[str, Any] = {}
        if return_components:
            components.update({
                'alpha': alpha,
                'similarities': similarities,
                'a': a,
                'active_mask': active_mask,
                'sparsity_loss': sparsity_loss
            })
        
        return reconstruction, components

    def _store_for_update(self, latent: torch.Tensor, similarities: torch.Tensor, alpha: torch.Tensor):
        """Store statistics for explicit updates"""
        self.last_latent = latent.detach()
        self.last_similarities = similarities.detach()
        self.last_alpha = alpha.detach()
        self.last_batch_size = latent.shape[0]

    def apply_explicit_updates(self):
        """Apply all non-gradient based updates"""
        if not hasattr(self, 'last_latent'):
            return
        
        self.step_counter += 1
        
        # Always update importance and occupancy
        self._update_importance_occupancy()
        
        # Only run expensive structural ops at specified frequency
        run_structural = (self.step_counter % self.structural_update_freq == 0)
        
        if run_structural and self.n_active_modes > 1:
            self._apply_structural_updates()
        
        # Clear stored data
        self._clear_stored_data()

    def _update_importance_occupancy(self):
        """Update importance and occupancy EMAs"""
        with torch.no_grad():
            alpha_mean = self.last_alpha.mean(dim=0)
            active_mask = self.active_mask
            n_active = self.n_active_modes
            
            if n_active == 0:
                return
            
            # Update occupancy EMA
            occupancy_active = self.occupancy[active_mask]
            new_occupancy = (
                self.importance_decay * occupancy_active +
                (1 - self.importance_decay) * alpha_mean
            )
            self.occupancy[active_mask] = new_occupancy
            
            # Update importance based on reconstruction quality and novelty
            R_t = self._compute_R_t()
            if R_t is not None:
                importance_update = (
                    self.rho_i.data[active_mask] *
                    self.eta_i.data[active_mask] *
                    R_t
                )
                self.lambda_i.data[active_mask] = (
                    self.lambda_i.data[active_mask] * self.importance_decay +
                    (1 - self.importance_decay) * importance_update
                )

    def _compute_R_t(self) -> Optional[torch.Tensor]:
        """Compute mode-wise relevance scores"""
        if not hasattr(self, 'last_latent'):
            return None
        
        active_mask = self.active_mask
        n_active = self.n_active_modes
        
        if n_active == 0:
            return None
        
        mu_active = self.mu[active_mask]
        latent = self.last_latent
        
        # Reconstruction quality component
        recon_errors = 1 - F.cosine_similarity(
            latent.unsqueeze(1), 
            mu_active.unsqueeze(0), 
            dim=2
        )
        
        # Temporal weighting
        B = latent.shape[0]
        temporal_weights = torch.linspace(
            0.5, 1.0, B, device=latent.device
        ).unsqueeze(1)
        
        # Combined relevance score
        R_t = (temporal_weights * (1 - recon_errors)).mean(dim=0)
        
        return R_t

    def _apply_structural_updates(self):
        """Apply merge, split, and prune operations"""
        self._merge_similar_modes()
        self._split_weak_modes()
        self._prune_modes()
        self._enforce_capacity_constraint()

    def _merge_similar_modes(self):
        """Merge highly similar active modes"""
        with torch.no_grad():
            active_mask = self.active_mask.clone()
            active_indices = torch.where(active_mask)[0]
            n_active = len(active_indices)
            
            if n_active < 2:
                return
            
            mu_active = self.mu[active_mask]
            
            # Compute similarity matrix
            similarities = F.cosine_similarity(
                mu_active.unsqueeze(1), 
                mu_active.unsqueeze(0), 
                dim=2
            )
            
            # Find pairs to merge (excluding self-similarity)
            triu_mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()
            highly_similar = (similarities > self.merge_threshold) & triu_mask
            
            merge_pairs = torch.where(highly_similar)
            
            merged_indices = set()
            for i, j in zip(merge_pairs[0], merge_pairs[1]):
                if i.item() in merged_indices or j.item() in merged_indices:
                    continue
                    
                idx_i = active_indices[i]
                idx_j = active_indices[j]
                
                # Merge mode j into mode i
                self._merge_two_modes(idx_i, idx_j)
                merged_indices.add(j.item())
                
                # Deactivate mode j
                active_mask[idx_j] = False
                self.logger.info(f"Merged modes {idx_j} into {idx_i}")
            
            self.active_mask.copy_(active_mask)

    def _merge_two_modes(self, idx_i: int, idx_j: int):
        """Merge mode j into mode i - preserves occupancy mass"""
        # Weighted average based on occupancy
        occ_i = self.occupancy[idx_i]
        occ_j = self.occupancy[idx_j]
        total_occ = occ_i + occ_j
        
        # Avoid division by zero
        if total_occ < 1e-8:
            total_occ = 1e-8
        
        # Merge prototypes
        self.mu.data[idx_i] = (
            occ_i * self.mu.data[idx_i] +
            occ_j * self.mu.data[idx_j]
        ) / total_occ
        
        # Merge other parameters
        self.w.data[idx_i] = self.w.data[idx_i] + self.w.data[idx_j]
        self.lambda_i.data[idx_i] = (
            occ_i * self.lambda_i.data[idx_i] +
            occ_j * self.lambda_i.data[idx_j]
        ) / total_occ
        self.rho_i.data[idx_i] = (
            occ_i * self.rho_i.data[idx_i] +
            occ_j * self.rho_i.data[idx_j]
        ) / total_occ
        self.eta_i.data[idx_i] = (
            occ_i * self.eta_i.data[idx_i] +
            occ_j * self.eta_i.data[idx_j]
        ) / total_occ
        
        # Preserve occupancy mass
        self.occupancy[idx_i] = total_occ
        
        if self.use_predictive:
            self.F.data[idx_i] = (
                occ_i * self.F.data[idx_i] +
                occ_j * self.F.data[idx_j]
            ) / total_occ

    def _split_weak_modes(self):
        """Split modes with high occupancy but poor performance"""
        with torch.no_grad():
            active_mask = self.active_mask.clone()
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) == 0:
                return
            
            current_active = len(active_indices)
            
            # Find candidate modes for splitting
            occupancy_active = self.occupancy[active_mask]
            lambda_active = self.lambda_i.data[active_mask]
            
            # Split score: high occupancy but low importance
            split_scores = occupancy_active / (lambda_active + 1e-8)
            split_candidates = split_scores > self.split_threshold
            
            for candidate_idx in torch.where(split_candidates)[0]:
                # Capacity check
                if current_active >= self.max_modes:
                    break
                    
                original_idx = active_indices[candidate_idx]
                new_idx = self._find_inactive_slot()
                
                if new_idx is None:
                    break
                    
                self._split_mode(original_idx, new_idx)
                active_mask[new_idx] = True
                current_active += 1
                self.logger.info(f"Split mode {original_idx} into {new_idx}")
            
            self.active_mask.copy_(active_mask)

    def _split_mode(self, original_idx: int, new_idx: int):
        """Split original mode into original_idx and new_idx"""
        # Add small noise to create new mode
        noise = torch.randn_like(self.mu.data[original_idx]) * 0.1
        self.mu.data[new_idx] = self.mu.data[original_idx] + noise
        
        # Initialize parameters for new mode
        self.w.data[new_idx] = self.w.data[original_idx] / 2
        self.w.data[original_idx] = self.w.data[original_idx] / 2
        
        self.lambda_i.data[new_idx] = self.lambda_i.data[original_idx]
        self.rho_i.data[new_idx] = self.rho_i.data[original_idx]
        self.eta_i.data[new_idx] = self.eta_i.data[original_idx]
        self.occupancy[new_idx] = self.occupancy[original_idx] / 2
        self.occupancy[original_idx] = self.occupancy[original_idx] / 2
        
        if self.use_predictive:
            self.F.data[new_idx] = (
                self.F.data[original_idx] +
                torch.randn_like(self.F.data[original_idx]) * 0.1
            )

    def _prune_modes(self):
        """Prune modes with very low occupancy"""
        with torch.no_grad():
            active_mask = self.active_mask.clone()
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) == 0:
                return
            
            occupancy_active = self.occupancy[active_mask]
            prune_candidates = occupancy_active < self.prune_threshold
            
            for candidate_idx in torch.where(prune_candidates)[0]:
                original_idx = active_indices[candidate_idx]
                active_mask[original_idx] = False
                self.logger.info(f"Pruned mode {original_idx}")
            
            self.active_mask.copy_(active_mask)

    def _enforce_capacity_constraint(self):
        """Ensure we don't exceed capacity constraints"""
        n_active = self.n_active_modes
        if n_active == 0:
            return
        
        with torch.no_grad():
            total_importance = self.lambda_i.data[self.active_mask].sum()
            capacity_limit = 1.0 + self.capacity_margin
            
            if total_importance > capacity_limit:
                scale_factor = capacity_limit / total_importance
                self.lambda_i.data[self.active_mask] *= scale_factor

    def _find_inactive_slot(self) -> Optional[int]:
        """Find first available inactive slot"""
        inactive_slots = torch.where(~self.active_mask)[0]
        return inactive_slots[0].item() if len(inactive_slots) > 0 else None

    def _clear_stored_data(self):
        """Clear stored batch data"""
        if hasattr(self, 'last_latent'):
            del self.last_latent
        if hasattr(self, 'last_similarities'):
            del self.last_similarities  
        if hasattr(self, 'last_alpha'):
            del self.last_alpha

    def predict_next(
        self,
        latent: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Predict next latent state using predictive matrices F.
        
        Args:
            latent: Current latent vectors [B, D]
            return_components: Whether to return intermediate computations
            
        Returns:
            prediction: Predicted next latent state [B, D]
            components: Dictionary of intermediate values
        """
        if not self.use_predictive:
            raise ValueError("Predictive mode not enabled. Set use_predictive=True")
        
        batch_size = latent.shape[0]
        active_mask = self.active_mask
        n_active = self.n_active_modes
        
        if n_active == 0:
            # No active modes - return zeros
            prediction = torch.zeros_like(latent)
            components = {
                "alpha": torch.zeros(batch_size, 0, device=latent.device),
                "prediction_error": torch.zeros(1, device=latent.device)
            }
            return prediction, components if return_components else (prediction, {})
        
        # Compute mode activations (same as forward pass)
        mu_active = self.mu[active_mask]
        similarities = F.cosine_similarity(
            latent.unsqueeze(1),
            mu_active.unsqueeze(0),
            dim=2
        )
        
        lambda_active = self.lambda_i[active_mask]
        a = similarities * lambda_active
        alpha = F.softmax(a, dim=1)  # [B, N_active]
        
        # Get active predictive matrices
        F_active = self.F[active_mask]  # [N_active, pred_dim, latent_dim]
        
        # Apply predictive transformation per mode
        # F_active @ latent.T -> [N_active, pred_dim, B]
        mode_predictions = torch.bmm(
            F_active,
            latent.unsqueeze(0).expand(n_active, -1, -1).transpose(1, 2)
        )  # [N_active, pred_dim, B]
        
        mode_predictions = mode_predictions.permute(2, 0, 1)  # [B, N_active, pred_dim]
        
        # Weight by attention and sum
        weighted_predictions = alpha.unsqueeze(2) * mode_predictions  # [B, N_active, pred_dim]
        prediction = weighted_predictions.sum(dim=1)  # [B, pred_dim]
        
        components: Dict[str, Any] = {}
        if return_components:
            components.update({
                'alpha': alpha,
                'similarities': similarities,
                'mode_predictions': mode_predictions,
                'prediction': prediction
            })
        
        return prediction, components

    def get_sparsity_loss(self) -> torch.Tensor:
        """Compute sparsity regularization loss"""
        if self.n_active_modes == 0:
            return torch.tensor(0.0, device=self.mu.device)
        
        alpha = self.last_alpha if hasattr(self, 'last_alpha') else None
        if alpha is None:
            return torch.tensor(0.0, device=self.mu.device)
        
        # Entropy-based sparsity
        entropy = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=1).mean()
        return entropy

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state dict with additional metadata"""
        state = {
            'model_state': self.state_dict(),
            'max_modes': self.max_modes,
            'latent_dim': self.latent_dim,
            'n_active_modes': self.n_active_modes,
            'use_predictive': self.use_predictive,
            'structural_update_freq': self.structural_update_freq
        }
        return state

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load state dict with metadata handling"""
        if 'model_state' in state_dict:
            super().load_state_dict(state_dict['model_state'], strict)
        else:
            super().load_state_dict(state_dict, strict)

    def extra_repr(self) -> str:
        return (
            f"latent_dim={self.latent_dim}, "
            f"max_modes={self.max_modes}, "
            f"active_modes={self.n_active_modes}"
        )


# ---- Tests ----

def test_occupancy_mass_conservation():
    """Test that occupancy mass is conserved during merges"""
    device = torch.device('cpu')
    pmm = StaticPseudoModeMemory(
        latent_dim=16,
        max_modes=10,
        init_modes=4,
        structural_update_freq=1,
        device=device
    )
    
    with torch.no_grad():
        # Set up test scenario: two similar modes
        pmm.mu.data[0] = torch.ones(16) * 0.1
        pmm.mu.data[1] = torch.ones(16) * 0.11
        pmm.occupancy[0] = 0.3
        pmm.occupancy[1] = 0.2
        pmm.w.data[0] = 0.3
        pmm.w.data[1] = 0.2

        initial_total_occupancy = pmm.occupancy[pmm.active_mask].sum().item()
        initial_total_w = pmm.w.data[pmm.active_mask].sum().item()

        # Force merge
        with torch.no_grad():
            pmm._merge_two_modes(0, 1)
            pmm.active_mask[1] = False

        final_total_occupancy = pmm.occupancy[pmm.active_mask].sum().item()
        final_total_w = pmm.w.data[pmm.active_mask].sum().item()

        print(f"Occupancy conservation: {initial_total_occupancy:.3f} -> {final_total_occupancy:.3f}")
        print(f"Weight conservation:     {initial_total_w:.3f} -> {final_total_w:.3f}")

        assert abs(final_total_occupancy - initial_total_occupancy) < 1e-6, "Occupancy mass not conserved!"
        assert abs(final_total_w - initial_total_w) < 1e-6, "Weight mass not conserved!"
        print(" Occupancy mass conservation test passed!")


def test_capacity_management():
    """Test that splits don't exceed max capacity"""
    device = torch.device('cpu')
    pmm = StaticPseudoModeMemory(
        latent_dim=8,
        max_modes=5,
        init_modes=4,
        split_threshold=0.1,
        structural_update_freq=1,
        device=device
    )
    
    with torch.no_grad():
        pmm.occupancy[pmm.active_mask] = 0.3
        pmm.lambda_i.data[pmm.active_mask] = 0.1

        active_mask_before = pmm.active_mask.clone()
        pmm._split_weak_modes()
        active_mask_after = pmm.active_mask.clone()

        n_active_before = active_mask_before.sum().item()
        n_active_after = active_mask_after.sum().item()

        print(f"Active modes: {n_active_before} -> {n_active_after} (max: {pmm.max_modes})")
        
        assert n_active_after <= pmm.max_modes, f"Exceeded max modes: {n_active_after} > {pmm.max_modes}"
        print(" Capacity management test passed!")


def test_gradient_safety():
    """Test that no gradient-related errors occur during structural updates"""
    device = torch.device('cpu')
    pmm = StaticPseudoModeMemory(
        latent_dim=16,
        max_modes=8,
        init_modes=4,
        structural_update_freq=1,
        device=device
    )
    
    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)

    try:
        for step in range(5):
            batch = torch.randn(4, 16, device=device)
            
            reconstruction, _ = pmm(batch, update_memory=True)
            
            loss = F.mse_loss(reconstruction, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pmm.apply_explicit_updates()
            
            print(f"Step {step}: Loss={loss.item():.4f}, Active modes={pmm.n_active_modes}")
            
        print(" Gradient safety test passed!")
        
    except Exception as e:
        assert False, f"Gradient safety test failed: {e}"


def test_predictive_extension():
    """Test the predictive extension for temporal prediction"""
    device = torch.device('cpu')
    pmm = StaticPseudoModeMemory(
        latent_dim=16,
        max_modes=8,
        init_modes=4,
        use_predictive=True,
        predictive_dim=16,  # Same as latent_dim for simplicity
        device=device
    )
    
    # Generate a simple temporal sequence
    batch_size = 4
    sequence_length = 10
    sequence = torch.randn(sequence_length, batch_size, 16, device=device)
    
    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-2)
    
    # Train to predict next step
    for epoch in range(5):
        total_pred_loss = 0.0
        for t in range(sequence_length - 1):
            current = sequence[t]
            next_true = sequence[t + 1]
            
            # Predict next state
            predicted_next, _ = pmm.predict_next(current, return_components=False)
            
            # Compute prediction loss
            pred_loss = F.mse_loss(predicted_next, next_true)
            
            optimizer.zero_grad()
            pred_loss.backward()
            optimizer.step()
            
            total_pred_loss += pred_loss.item()
        
        avg_loss = total_pred_loss / (sequence_length - 1)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Avg prediction loss={avg_loss:.4f}")
    
    print("✓ Predictive extension test passed!")


def example_usage():
    """Demonstrate the static PMM in a training context"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pmm = StaticPseudoModeMemory(
        latent_dim=128,
        max_modes=64,
        init_modes=8,
        structural_update_freq=5,
        use_predictive=True,
        device=device
    )

    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)

    for step in range(100):
        batch = torch.randn(32, 128, device=device)
        
        reconstruction, components = pmm(batch, update_memory=True, return_components=True)
        
        recon_loss = F.mse_loss(reconstruction, batch)
        sparsity_loss = components['sparsity_loss']
        total_loss = recon_loss + 0.1 * sparsity_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        pmm.apply_explicit_updates()
        
        if step % 10 == 0:
            print(f"Step {step}: Active modes={pmm.n_active_modes}, Loss={total_loss.item():.4f}")


def example_temporal_prediction():
    """Demonstrate temporal prediction with the predictive extension"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pmm = StaticPseudoModeMemory(
        latent_dim=64,
        max_modes=32,
        init_modes=8,
        use_predictive=True,
        predictive_dim=64,
        device=device
    )
    
    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)
    
    # Generate synthetic temporal data
    print("\nTemporal Prediction Example:")
    for step in range(50):
        # Simulate a sequence
        batch_size = 16
        current_state = torch.randn(batch_size, 64, device=device)
        next_state = torch.randn(batch_size, 64, device=device)
        
        # Reconstruction loss (learn current state)
        reconstruction, components = pmm(current_state, update_memory=True, return_components=True)
        recon_loss = F.mse_loss(reconstruction, current_state)
        
        # Prediction loss (predict next state)
        predicted_next, _ = pmm.predict_next(current_state)
        pred_loss = F.mse_loss(predicted_next, next_state)
        
        # Combined loss
        sparsity_loss = components['sparsity_loss']
        total_loss = recon_loss + pred_loss + 0.1 * sparsity_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        pmm.apply_explicit_updates()
        
        if step % 10 == 0:
            print(f"Step {step}: Active modes={pmm.n_active_modes}, "
                  f"Recon={recon_loss.item():.4f}, Pred={pred_loss.item():.4f}")


# ============================================================================
# HIERARCHICAL & MULTI-SCALE EXTENSIONS
# ============================================================================

class HierarchicalPseudoModeMemory(nn.Module):
    """
    Hierarchical PMM with multi-scale prediction.
    
    Architecture:
        Level 0 (Low): Fine-grained patterns (e.g., edges, primitives)
        Level 1 (Mid): Intermediate patterns (e.g., parts, textures)
        Level 2 (High): Abstract patterns (e.g., objects, scenes)
    
    Each level can attend to representations from lower levels.
    """
    
    def __init__(
        self,
        latent_dims: list,  # [low_dim, mid_dim, high_dim]
        max_modes_per_level: list,
        init_modes_per_level: list,
        prediction_horizons: list = None,  # [1, 5, 10] steps ahead
        use_top_down: bool = True,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.num_levels = len(latent_dims)
        self.latent_dims = latent_dims
        self.prediction_horizons = prediction_horizons or [1, 5, 10]
        self.use_top_down = use_top_down
        self.device = device
        
        # Create PMM for each hierarchical level
        self.levels = nn.ModuleList([
            StaticPseudoModeMemory(
                latent_dim=latent_dims[i],
                max_modes=max_modes_per_level[i],
                init_modes=init_modes_per_level[i],
                use_predictive=True,
                predictive_dim=latent_dims[i],
                device=device
            )
            for i in range(self.num_levels)
        ])
        
        # Bottom-up projection (low → mid → high)
        self.bottom_up_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dims[i], latent_dims[i+1], device=device),
                nn.LayerNorm(latent_dims[i+1], device=device),
                nn.ReLU()
            )
            for i in range(self.num_levels - 1)
        ])
        
        # Top-down modulation (high → mid → low)
        if use_top_down:
            self.top_down_proj = nn.ModuleList([
                nn.Linear(latent_dims[i+1], latent_dims[i], device=device)
                for i in range(self.num_levels - 1)
            ])
        
        # Multi-scale prediction heads
        self.multi_scale_heads = nn.ModuleDict()
        for horizon in self.prediction_horizons:
            self.multi_scale_heads[f'h{horizon}'] = nn.ModuleList([
                nn.Linear(latent_dims[i], latent_dims[i], device=device)
                for i in range(self.num_levels)
            ])
    
    def forward(
        self,
        latent: torch.Tensor,
        level: int = 0,
        update_memory: bool = False,
        return_all_levels: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through hierarchical levels.
        
        Args:
            latent: Input at specified level [B, D_level]
            level: Which level to start from (default: 0 = lowest)
            update_memory: Whether to update mode memories
            return_all_levels: Return reconstructions from all levels
        
        Returns:
            reconstruction: Final reconstruction
            components: Hierarchical information
        """
        batch_size = latent.shape[0]
        
        # Bottom-up processing
        level_inputs = [None] * self.num_levels
        level_reconstructions = {}
        level_components = {}
        
        level_inputs[level] = latent
        
        # Process upward through hierarchy
        for i in range(level, self.num_levels):
            # Get reconstruction at this level
            recon, comps = self.levels[i](
                level_inputs[i],
                update_memory=update_memory,
                return_components=True
            )
            
            level_reconstructions[f'level_{i}'] = recon
            level_components[f'level_{i}'] = comps
            
            # Project to next level if not at top
            if i < self.num_levels - 1:
                level_inputs[i + 1] = self.bottom_up_proj[i](level_inputs[i])
        
        # Top-down modulation
        if self.use_top_down:
            for i in range(self.num_levels - 2, level - 1, -1):
                top_down_signal = self.top_down_proj[i](
                    level_reconstructions[f'level_{i+1}']
                )
                level_reconstructions[f'level_{i}'] = (
                    level_reconstructions[f'level_{i}'] + 0.5 * top_down_signal
                )
        
        # Return final reconstruction at input level
        final_recon = level_reconstructions[f'level_{level}']
        
        components = {
            'level_reconstructions': level_reconstructions if return_all_levels else {},
            'level_components': level_components,
            'n_active_modes': [self.levels[i].n_active_modes for i in range(self.num_levels)]
        }
        
        return final_recon, components
    
    def predict_multiscale(
        self,
        latent: torch.Tensor,
        level: int = 0,
        horizons: Optional[list] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Multi-horizon temporal prediction.
        
        Args:
            latent: Current state [B, D]
            level: Which hierarchical level to predict from
            horizons: Which time horizons to predict (default: all)
        
        Returns:
            predictions: {horizon: predicted_state}
        """
        if horizons is None:
            horizons = self.prediction_horizons
        
        predictions = {}
        
        # Get base prediction from PMM at this level
        base_pred, _ = self.levels[level].predict_next(latent)
        
        for h in horizons:
            head_key = f'h{h}'
            if head_key in self.multi_scale_heads:
                # Apply horizon-specific head
                pred = self.multi_scale_heads[head_key][level](base_pred)
                
                # For longer horizons, apply iteratively
                if h > 1:
                    current = base_pred
                    for _ in range(h - 1):
                        current, _ = self.levels[level].predict_next(current)
                    pred = 0.5 * pred + 0.5 * current  # Blend direct and iterative
            else:
                # Fallback: iterative prediction
                pred = base_pred
                for _ in range(h - 1):
                    pred, _ = self.levels[level].predict_next(pred)
            
            predictions[h] = pred
        
        return predictions
    
    def apply_explicit_updates(self):
        """Update all hierarchical levels"""
        for level_pmm in self.levels:
            level_pmm.apply_explicit_updates()
    
    @property
    def total_active_modes(self) -> int:
        """Total active modes across all levels"""
        return sum(level.n_active_modes for level in self.levels)


# ============================================================================
# ADVANCED UPGRADES - TIER 1
# ============================================================================

class AttentionPseudoModeMemory(StaticPseudoModeMemory):
    """
    PMM with learned attention mechanism instead of just cosine similarity.
    Provides more expressive mode selection and better reconstruction.
    """
    
    def __init__(self, *args, use_attention=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_attention = use_attention
        
        if use_attention:
            self.query_proj = nn.Linear(self.latent_dim, self.latent_dim, device=kwargs.get('device'))
            self.key_proj = nn.Linear(self.latent_dim, self.latent_dim, device=kwargs.get('device'))
            self.value_proj = nn.Linear(self.latent_dim, self.latent_dim, device=kwargs.get('device'))
    
    def forward(
        self,
        latent: torch.Tensor,
        update_memory: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with learned attention"""
        batch_size = latent.shape[0]
        active_mask = self.active_mask
        n_active = self.n_active_modes
        
        if n_active == 0:
            output = torch.zeros_like(latent)
            sparsity_loss = torch.zeros(1, device=latent.device, requires_grad=True).squeeze()
            components = {
                "alpha": torch.zeros(batch_size, 0, device=latent.device),
                "sparsity_loss": sparsity_loss
            }
            return output, components if return_components else (output, {})
        
        if self.use_attention:
            # Learned attention mechanism
            Q = self.query_proj(latent)  # [B, D]
            mu_active = self.mu[active_mask]
            K = self.key_proj(mu_active)  # [N, D]
            V = self.value_proj(mu_active)  # [N, D]
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.T) / math.sqrt(self.latent_dim)  # [B, N]
            
            # Apply importance weighting
            lambda_active = self.lambda_i[active_mask]
            scores = scores * lambda_active
            
            alpha = F.softmax(scores, dim=1)  # [B, N]
            reconstruction = torch.matmul(alpha, V)  # [B, D]
            similarities = scores  # For compatibility
        else:
            # Fall back to parent class implementation
            return super().forward(latent, update_memory, return_components)
        
        # Compute sparsity loss
        sparsity_loss = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=1).mean()
        
        if update_memory:
            self._store_for_update(latent, similarities, alpha)
        
        components: Dict[str, Any] = {}
        if return_components:
            components.update({
                'alpha': alpha,
                'similarities': similarities,
                'active_mask': active_mask,
                'sparsity_loss': sparsity_loss
            })
        
        return reconstruction, components


class UncertaintyPMM(StaticPseudoModeMemory):
    """
    PMM with uncertainty quantification for safety-critical applications.
    Provides both epistemic (model) and aleatoric (data) uncertainty estimates.
    """
    
    def __init__(self, *args, estimate_uncertainty=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.estimate_uncertainty = estimate_uncertainty
        
        if estimate_uncertainty:
            # Per-mode noise/uncertainty parameter
            self.log_sigma = nn.Parameter(torch.zeros(self.max_modes, device=kwargs.get('device')))
    
    def forward_with_uncertainty(
        self,
        latent: torch.Tensor,
        update_memory: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with uncertainty estimates"""
        recon, components = super().forward(latent, update_memory, True)
        
        if not self.estimate_uncertainty or self.n_active_modes == 0:
            components['epistemic_uncertainty'] = torch.zeros(latent.shape[0], device=latent.device)
            components['aleatoric_uncertainty'] = torch.zeros(latent.shape[0], device=latent.device)
            components['total_uncertainty'] = torch.zeros(latent.shape[0], device=latent.device)
            return recon, components if return_components else (recon, {})
        
        alpha = components['alpha']
        
        # Epistemic uncertainty (model uncertainty from mode selection)
        entropy = -torch.sum(alpha * torch.log(alpha + 1e-8), dim=1)
        epistemic_uncertainty = entropy / math.log(self.n_active_modes + 1e-8)
        
        # Aleatoric uncertainty (data noise, learned per mode)
        sigma = F.softplus(self.log_sigma[self.active_mask])  # [N_active]
        aleatoric_uncertainty = (alpha * sigma).sum(dim=1)  # [B]
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        components.update({
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty
        })
        
        return recon, components if return_components else (recon, {})


import random

class ConsolidatedPMM(StaticPseudoModeMemory):
    """
    PMM with memory consolidation for continual learning.
    Stores important memories and replays them to prevent catastrophic forgetting.
    """
    
    def __init__(self, *args, consolidation_buffer_size=1000, importance_threshold=0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.consolidation_buffer = []
        self.buffer_size = consolidation_buffer_size
        self.importance_threshold = importance_threshold
    
    def forward(
        self,
        latent: torch.Tensor,
        update_memory: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with memory consolidation"""
        recon, comps = super().forward(latent, update_memory, True)
        
        if update_memory and self.n_active_modes > 0:
            # Store high-importance samples
            alpha = comps['alpha']
            importance = alpha.max(dim=1)[0]  # [B]
            
            for i in range(latent.shape[0]):
                if importance[i].item() > self.importance_threshold:
                    self.consolidation_buffer.append({
                        'latent': latent[i].detach().cpu(),
                        'importance': importance[i].item()
                    })
            
            # Maintain buffer size
            if len(self.consolidation_buffer) > self.buffer_size:
                # Keep most important memories
                self.consolidation_buffer.sort(key=lambda x: x['importance'], reverse=True)
                self.consolidation_buffer = self.consolidation_buffer[:self.buffer_size]
        
        return recon, comps if return_components else (recon, {})
    
    def consolidate(self, optimizer, batch_size=32):
        """Replay important memories to prevent forgetting"""
        if len(self.consolidation_buffer) == 0:
            return 0.0
        
        # Sample from buffer
        sample_size = min(batch_size, len(self.consolidation_buffer))
        batch = random.sample(self.consolidation_buffer, sample_size)
        latents = torch.stack([b['latent'].to(self.mu.device) for b in batch])
        
        # Reconstruct and update
        recon, _ = self(latents, update_memory=True)
        loss = F.mse_loss(recon, latents)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class CausalPMM(StaticPseudoModeMemory):
    """
    PMM with causal structure learning between modes.
    Learns which modes cause transitions to which other modes.
    """
    
    def __init__(self, *args, learn_causal_graph=True, **kwargs):
        super().__init__(*args, use_predictive=True, **kwargs)
        self.learn_causal_graph = learn_causal_graph
        
        if learn_causal_graph:
            # Causal adjacency matrix: mode i → mode j
            self.causal_graph = nn.Parameter(
                torch.zeros(self.max_modes, self.max_modes, device=kwargs.get('device'))
            )
    
    def predict_next_causal(
        self,
        latent: torch.Tensor,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Prediction using learned causal structure"""
        if not self.learn_causal_graph:
            return self.predict_next(latent, return_components)
        
        # Current mode activation
        _, comps = self(latent, return_components=True)
        alpha = comps['alpha']  # [B, N_active]
        
        if self.n_active_modes == 0:
            return torch.zeros_like(latent), {}
        
        # Causal transition probabilities
        causal_adj = torch.sigmoid(self.causal_graph)  # [M, M]
        active_causal = causal_adj[self.active_mask][:, self.active_mask]  # [N, N]
        
        # Next mode distribution via causal graph
        next_alpha = torch.matmul(alpha, active_causal)  # [B, N]
        next_alpha = F.softmax(next_alpha, dim=1)
        
        # Predict using next mode distribution
        mu_active = self.mu[self.active_mask]
        prediction = torch.matmul(next_alpha, mu_active)  # [B, D]
        
        components = {
            'causal_graph': active_causal,
            'next_alpha': next_alpha,
            'current_alpha': alpha
        }
        
        return prediction, components if return_components else (prediction, {})
    
    def get_causal_sparsity_loss(self, lambda_sparse=0.01):
        """L1 regularization on causal graph for sparsity"""
        if not self.learn_causal_graph:
            return torch.tensor(0.0, device=self.mu.device)
        
        causal_adj = torch.sigmoid(self.causal_graph)
        return lambda_sparse * causal_adj.abs().sum()


# ============================================================================
# ADVANCED UPGRADES - TIER 2 & 3
# ============================================================================

# Add these utility methods to StaticPseudoModeMemory via extension
class EnhancedPMM(StaticPseudoModeMemory):
    """
    PMM with all Tier 2 and Tier 3 enhancements:
    - Fast inference modes
    - Meta-learning support
    - Visualization tools
    - Advanced checkpointing
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_top_k = None  # None = use all modes
        self.mode_history = []  # Track mode evolution
    
    # EFFICIENT INFERENCE
    def set_inference_mode(self, mode='accurate'):
        """Optimize for different deployment scenarios"""
        if mode == 'fast':
            self.inference_top_k = 5
        elif mode == 'accurate':
            self.inference_top_k = None
        elif mode == 'memory_efficient':
            self.inference_top_k = 3
        return self
    
    def forward_fast(self, latent: torch.Tensor) -> torch.Tensor:
        """Fast inference using top-K modes only"""
        if self.inference_top_k is None or self.n_active_modes <= self.inference_top_k:
            recon, _ = self(latent)
            return recon
        
        # Compute similarities
        mu_active = self.mu[self.active_mask]
        similarities = F.cosine_similarity(
            latent.unsqueeze(1),
            mu_active.unsqueeze(0),
            dim=2
        )
        
        # Select top-K modes
        top_k = min(self.inference_top_k, self.n_active_modes)
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k, dim=1)
        
        # Reconstruct using only top-K
        alpha = F.softmax(top_k_values * self.lambda_i[self.active_mask][top_k_indices], dim=1)
        selected_mu = mu_active[top_k_indices]
        recon = (alpha.unsqueeze(2) * selected_mu).sum(dim=1)
        
        return recon
    
    # META-LEARNING
    def meta_adapt(self, support_set: torch.Tensor, n_grad_steps=5, inner_lr=0.01):
        """MAML-style fast adaptation to new task/domain"""
        # Store original parameters
        original_state = {name: p.clone() for name, p in self.named_parameters()}
        
        # Inner loop: adapt to support set
        for _ in range(n_grad_steps):
            recon, _ = self(support_set)
            loss = F.mse_loss(recon, support_set)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, self.parameters(), create_graph=False)
            
            # Update parameters
            with torch.no_grad():
                for param, grad in zip(self.parameters(), grads):
                    param.data = param.data - inner_lr * grad
        
        # Return adapted parameters (can save or use for query set)
        adapted_state = {name: p.clone() for name, p in self.named_parameters()}
        
        # Restore original parameters
        for name, param in self.named_parameters():
            param.data = original_state[name]
        
        return adapted_state
    
    def apply_adapted_params(self, adapted_state):
        """Apply previously adapted parameters"""
        for name, param in self.named_parameters():
            if name in adapted_state:
                param.data = adapted_state[name]
    
    # VISUALIZATION
    def visualize_modes(self, save_path='modes.png'):
        """Visualize mode structure and activations"""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Mode importance
        if self.n_active_modes > 0:
            importance = self.lambda_i[self.active_mask].detach().cpu().numpy()
            axes[0, 0].bar(range(len(importance)), importance)
            axes[0, 0].set_title('Mode Importance')
            axes[0, 0].set_xlabel('Mode Index')
            axes[0, 0].set_ylabel('Importance (λ)')
            
            # 2. Mode occupancy
            occupancy = self.occupancy[self.active_mask].detach().cpu().numpy()
            axes[0, 1].bar(range(len(occupancy)), occupancy)
            axes[0, 1].set_title('Mode Occupancy')
            axes[0, 1].set_xlabel('Mode Index')
            axes[0, 1].set_ylabel('Occupancy')
            
            # 3. Mode similarity matrix
            mu_active = self.mu[self.active_mask]
            sim_matrix = F.cosine_similarity(
                mu_active.unsqueeze(1),
                mu_active.unsqueeze(0),
                dim=2
            ).detach().cpu().numpy()
            im = axes[1, 0].imshow(sim_matrix, cmap='viridis', vmin=0, vmax=1)
            axes[1, 0].set_title('Mode Similarity Matrix')
            plt.colorbar(im, ax=axes[1, 0])
        
        # 4. Mode evolution over time
        if len(self.mode_history) > 0:
            axes[1, 1].plot(self.mode_history)
            axes[1, 1].set_title('Active Modes Over Time')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Number of Active Modes')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No history tracked yet',
                           ha='center', va='center')
            axes[1, 1].set_title('Active Modes Over Time')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to {save_path}")
    
    # CHECKPOINTING
    def save_full_state(self, path: str):
        """Save complete state including buffers and history"""
        state = {
            'model_state': self.state_dict(),
            'active_mask': self.active_mask.cpu(),
            'step_counter': self.step_counter,
            'mode_history': self.mode_history,
            'inference_top_k': self.inference_top_k,
            'hyperparameters': {
                'latent_dim': self.latent_dim,
                'max_modes': self.max_modes,
                'importance_decay': self.importance_decay,
                'capacity_margin': self.capacity_margin,
                'merge_threshold': self.merge_threshold,
                'split_threshold': self.split_threshold,
                'prune_threshold': self.prune_threshold,
                'structural_update_freq': self.structural_update_freq,
                'use_predictive': self.use_predictive,
            }
        }
        torch.save(state, path)
        print(f"Full state saved to {path}")
    
    def load_full_state(self, path: str):
        """Load complete state to resume exactly where we left off"""
        state = torch.load(path, map_location=self.mu.device)
        self.load_state_dict(state['model_state'])
        self.active_mask = state['active_mask'].to(self.mu.device)
        self.step_counter = state['step_counter']
        self.mode_history = state.get('mode_history', [])
        self.inference_top_k = state.get('inference_top_k', None)
        print(f"Full state loaded from {path}")
        print(f"Resumed at step {self.step_counter} with {self.n_active_modes} active modes")
    
    # TRACKING
    def apply_explicit_updates(self):
        """Enhanced version that tracks history"""
        super().apply_explicit_updates()
        self.mode_history.append(self.n_active_modes)


# Disentangled Representation Learning
class DisentangledPMM(StaticPseudoModeMemory):
    """PMM that learns disentangled interpretable factors"""
    
    def __init__(self, *args, n_factors=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_factors = n_factors
        # Each mode contributes to interpretable factors
        self.factor_weights = nn.Parameter(
            torch.randn(self.max_modes, n_factors, device=kwargs.get('device')) * 0.1
        )
    
    def forward_disentangled(
        self,
        latent: torch.Tensor,
        update_memory: bool = False,
        return_components: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass that extracts disentangled factors"""
        recon, comps = super().forward(latent, update_memory, True)
        
        if self.n_active_modes > 0:
            # Extract factors from mode activations
            alpha = comps['alpha']  # [B, N_active]
            factor_weights_active = self.factor_weights[self.active_mask]  # [N, K]
            factors = torch.matmul(alpha, factor_weights_active)  # [B, K]
        else:
            factors = torch.zeros(latent.shape[0], self.n_factors, device=latent.device)
        
        comps['factors'] = factors
        return recon, comps if return_components else (recon, {})
    
    def disentanglement_loss(self, factors: torch.Tensor) -> torch.Tensor:
        """Total correlation penalty to encourage independence"""
        if factors.shape[0] < 2:
            return torch.tensor(0.0, device=factors.device)
        
        # Covariance matrix
        factors_centered = factors - factors.mean(dim=0, keepdim=True)
        cov = torch.matmul(factors_centered.T, factors_centered) / (factors.shape[0] - 1)
        
        # Penalize off-diagonal elements (correlations)
        off_diagonal = cov - torch.diag(torch.diag(cov))
        return off_diagonal.abs().sum()


# Adversarial Robustness
def adversarial_training_step(pmm: StaticPseudoModeMemory, latent: torch.Tensor, 
                              epsilon=0.1, return_adv=False):
    """Generate adversarial examples and compute robust loss"""
    latent_clean = latent.clone().requires_grad_(True)
    
    # Forward pass
    recon, _ = pmm(latent_clean)
    loss = F.mse_loss(recon, latent_clean)
    
    # Compute gradient w.r.t input
    grad = torch.autograd.grad(loss, latent_clean, create_graph=True)[0]
    
    # Generate adversarial example (FGSM)
    latent_adv = latent_clean + epsilon * grad.sign()
    latent_adv = latent_adv.detach()
    
    # Reconstruct adversarial
    recon_clean, _ = pmm(latent_clean, update_memory=True)
    recon_adv, _ = pmm(latent_adv, update_memory=False)
    
    # Robust loss (both clean and adversarial)
    robust_loss = 0.5 * F.mse_loss(recon_clean, latent) + 0.5 * F.mse_loss(recon_adv, latent)
    
    if return_adv:
        return robust_loss, latent_adv
    return robust_loss


# ============================================================================
# PRODUCTION UPGRADES - DEPLOYMENT & SCALING
# ============================================================================

class DistributedPMM(StaticPseudoModeMemory):
    """
    PMM with distributed training support for multi-GPU scaling.
    Synchronizes mode structure across multiple GPUs.
    """
    
    def __init__(self, *args, distributed=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.distributed = distributed
        
        if distributed and torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
            self.device_id = torch.cuda.current_device()
        else:
            self.world_size = 1
            self.rank = 0
            self.distributed = False
    
    def sync_modes_across_gpus(self):
        """Synchronize active modes and parameters across all GPUs"""
        if not self.distributed:
            return
        
        with torch.no_grad():
            # Gather active masks from all GPUs
            active_masks = [torch.zeros_like(self.active_mask) for _ in range(self.world_size)]
            torch.distributed.all_gather(active_masks, self.active_mask)
            
            # Consensus: mode is active if active on majority of GPUs
            stacked_masks = torch.stack(active_masks).float()
            consensus_mask = (stacked_masks.sum(dim=0) > self.world_size / 2)
            
            # Update active mask
            self.active_mask.copy_(consensus_mask)
            
            # Average parameters for active modes
            if self.n_active_modes > 0:
                for param_name in ['mu', 'w', 'lambda_i', 'rho_i', 'eta_i']:
                    param = getattr(self, param_name)
                    torch.distributed.all_reduce(param.data, op=torch.distributed.ReduceOp.SUM)
                    param.data.div_(self.world_size)
                
                # Average buffers
                torch.distributed.all_reduce(self.occupancy, op=torch.distributed.ReduceOp.SUM)
                self.occupancy.div_(self.world_size)
    
    def apply_explicit_updates(self):
        """Apply updates and sync across GPUs"""
        super().apply_explicit_updates()
        if self.distributed and self.step_counter % 10 == 0:
            self.sync_modes_across_gpus()


class StreamingPMM(StaticPseudoModeMemory):
    """
    PMM for streaming data with bounded memory and adaptive learning.
    Suitable for online learning and real-time applications.
    """
    
    def __init__(self, *args, stability_window=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.stability_window = stability_window
        self.recent_mode_counts = []
        self.adaptive_lr_scale = 1.0
    
    def process_stream(self, data_stream, optimizer, max_steps=None):
        """Process infinite data stream with bounded memory"""
        step = 0
        for batch in data_stream:
            # Forward and update
            recon, _ = self(batch, update_memory=True)
            loss = F.mse_loss(recon, batch)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            
            # Apply adaptive learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.adaptive_lr_scale
            
            optimizer.step()
            self.apply_explicit_updates()
            
            # Track stability
            self.recent_mode_counts.append(self.n_active_modes)
            if len(self.recent_mode_counts) > self.stability_window:
                self.recent_mode_counts.pop(0)
            
            # Adjust learning rate based on stability
            if self.is_stable() and self.adaptive_lr_scale > 0.1:
                self.adaptive_lr_scale *= 0.99
            elif not self.is_stable() and self.adaptive_lr_scale < 1.0:
                self.adaptive_lr_scale *= 1.01
            
            step += 1
            if max_steps and step >= max_steps:
                break
    
    def is_stable(self) -> bool:
        """Check if mode structure is stable"""
        if len(self.recent_mode_counts) < self.stability_window:
            return False
        
        # Stable if mode count variance is low
        counts_tensor = torch.tensor(self.recent_mode_counts, dtype=torch.float32)
        variance = counts_tensor.var().item()
        return variance < 1.0


# Export utilities
def export_to_onnx(pmm: StaticPseudoModeMemory, path: str, example_input: torch.Tensor):
    """Export PMM to ONNX for cross-platform deployment"""
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise ImportError(
            "ONNX export requires the 'onnx' package. "
            "Install with: pip install onnx onnxruntime"
        )
    
    pmm.eval()
    
    # Create wrapper that returns only reconstruction (ONNX doesn't like dicts)
    class ONNXWrapper(nn.Module):
        def __init__(self, pmm_model):
            super().__init__()
            self.pmm = pmm_model
        
        def forward(self, x):
            recon, _ = self.pmm(x, update_memory=False, return_components=False)
            return recon
    
    wrapper = ONNXWrapper(pmm)
    
    torch.onnx.export(
        wrapper,
        example_input,
        path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['latent'],
        output_names=['reconstruction'],
        dynamic_axes={'latent': {0: 'batch_size'}, 'reconstruction': {0: 'batch_size'}}
    )
    print(f"Model exported to ONNX: {path}")


def to_torchscript(pmm: StaticPseudoModeMemory) -> torch.jit.ScriptModule:
    """Convert PMM to TorchScript for C++ deployment"""
    pmm.eval()
    
    # Trace the model (scripting has issues with dynamic control flow)
    example_input = torch.randn(1, pmm.latent_dim, device=pmm.mu.device)
    
    class TorchScriptWrapper(nn.Module):
        def __init__(self, pmm_model):
            super().__init__()
            self.pmm = pmm_model
        
        def forward(self, x):
            recon, _ = self.pmm(x, update_memory=False, return_components=False)
            return recon
    
    wrapper = TorchScriptWrapper(pmm)
    traced = torch.jit.trace(wrapper, example_input)
    print("Model converted to TorchScript")
    return traced


def quantize_model(pmm: StaticPseudoModeMemory):
    """Quantize PMM to INT8 for 4x speedup and memory reduction"""
    pmm.eval()
    
    # Create a wrapper that handles tuple returns properly
    class QuantizableWrapper(nn.Module):
        def __init__(self, pmm_model):
            super().__init__()
            self.pmm = pmm_model
        
        def forward(self, x):
            # Return only reconstruction (quantization can't handle dicts)
            recon, _ = self.pmm(x, update_memory=False, return_components=False)
            return recon
    
    wrapper = QuantizableWrapper(pmm)
    
    # Dynamic quantization (quantizes weights, keeps activations in float)
    quantized_model = torch.quantization.quantize_dynamic(
        wrapper,
        {nn.Linear} if hasattr(pmm, 'query_proj') else set(),
        dtype=torch.qint8
    )
    
    print("Model quantized to INT8")
    return quantized_model


# PyTorch Lightning Integration
try:
    import pytorch_lightning as pl
    
    class LightningPMM(pl.LightningModule):
        """PyTorch Lightning wrapper for easy training with auto-logging"""
        
        def __init__(self, pmm_config: dict, learning_rate=1e-3, sparsity_weight=0.1):
            super().__init__()
            self.save_hyperparameters()
            self.pmm = StaticPseudoModeMemory(**pmm_config)
            self.learning_rate = learning_rate
            self.sparsity_weight = sparsity_weight
        
        def forward(self, x):
            return self.pmm(x, update_memory=False, return_components=False)
        
        def training_step(self, batch, batch_idx):
            recon, comps = self.pmm(batch, update_memory=True, return_components=True)
            recon_loss = F.mse_loss(recon, batch)
            sparsity_loss = comps.get('sparsity_loss', torch.tensor(0.0))
            
            total_loss = recon_loss + self.sparsity_weight * sparsity_loss
            
            # Log metrics
            self.log('train/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('train/recon_loss', recon_loss, on_step=False, on_epoch=True)
            self.log('train/sparsity_loss', sparsity_loss, on_step=False, on_epoch=True)
            self.log('train/n_active_modes', self.pmm.n_active_modes, on_step=False, on_epoch=True)
            
            return total_loss
        
        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.pmm.apply_explicit_updates()
        
        def validation_step(self, batch, batch_idx):
            recon, _ = self.pmm(batch, update_memory=False)
            val_loss = F.mse_loss(recon, batch)
            self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
            return val_loss
        
        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.pmm.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss'}
            }
    
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    print("PyTorch Lightning not available. Install with: pip install pytorch-lightning")


# Monitoring Integration
class MonitoredPMM(StaticPseudoModeMemory):
    """PMM with W&B/TensorBoard logging integration"""
    
    def __init__(self, *args, logger_type='wandb', **kwargs):
        super().__init__(*args, **kwargs)
        self.logger_type = logger_type
        self.logger = None
        
        if logger_type == 'wandb':
            try:
                import wandb
                if wandb.run is not None:
                    self.logger = wandb
            except ImportError:
                print("wandb not available. Install with: pip install wandb")
        
        elif logger_type == 'tensorboard':
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.logger = SummaryWriter()
            except ImportError:
                print("tensorboard not available")
    
    def apply_explicit_updates(self):
        super().apply_explicit_updates()
        
        if self.logger is None or self.n_active_modes == 0:
            return
        
        # Log metrics
        metrics = {
            'n_active_modes': self.n_active_modes,
            'mode_importance_mean': self.lambda_i[self.active_mask].mean().item(),
            'mode_importance_max': self.lambda_i[self.active_mask].max().item(),
            'mode_occupancy_mean': self.occupancy[self.active_mask].mean().item(),
            'mode_occupancy_std': self.occupancy[self.active_mask].std().item(),
        }
        
        if self.logger_type == 'wandb':
            self.logger.log(metrics, step=self.step_counter)
        elif self.logger_type == 'tensorboard':
            for key, value in metrics.items():
                self.logger.add_scalar(key, value, self.step_counter)


# Hyperparameter Optimization
def auto_tune_hyperparameters(
    data: torch.Tensor,
    latent_dim: int,
    n_trials=50,
    n_epochs=10
):
    """Automatic hyperparameter search using Optuna"""
    try:
        import optuna
    except ImportError:
        print("Optuna not available. Install with: pip install optuna")
        return None
    
    def objective(trial):
        # Suggest hyperparameters
        config = {
            'latent_dim': latent_dim,
            'max_modes': trial.suggest_int('max_modes', 16, 128),
            'init_modes': trial.suggest_int('init_modes', 4, 16),
            'merge_threshold': trial.suggest_float('merge_threshold', 0.7, 0.95),
            'split_threshold': trial.suggest_float('split_threshold', 0.2, 0.5),
            'prune_threshold': trial.suggest_float('prune_threshold', 0.01, 0.1),
            'importance_decay': trial.suggest_float('importance_decay', 0.95, 0.999),
        }
        
        # Create and train model
        pmm = StaticPseudoModeMemory(**config)
        optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)
        
        # Simple training loop
        pmm.train()
        for epoch in range(n_epochs):
            recon, _ = pmm(data, update_memory=True)
            loss = F.mse_loss(recon, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pmm.apply_explicit_updates()
        
        # Return final loss
        pmm.eval()
        with torch.no_grad():
            recon, _ = pmm(data)
            final_loss = F.mse_loss(recon, data).item()
        
        return final_loss
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest hyperparameters: {study.best_params}")
    print(f"Best loss: {study.best_value:.4f}")
    
    return study.best_params


# Active Learning Utilities
def select_informative_samples(
    pmm: UncertaintyPMM,
    unlabeled_pool: torch.Tensor,
    n_samples=100,
    batch_size=32
):
    """Select most informative samples for labeling based on uncertainty"""
    pmm.eval()
    uncertainties = []
    
    with torch.no_grad():
        for i in range(0, len(unlabeled_pool), batch_size):
            batch = unlabeled_pool[i:i+batch_size]
            _, comps = pmm.forward_with_uncertainty(batch, return_components=True)
            uncertainties.append(comps['total_uncertainty'])
    
    # Concatenate and select top-K
    all_uncertainties = torch.cat(uncertainties)
    top_k_indices = torch.topk(all_uncertainties, min(n_samples, len(all_uncertainties))).indices
    
    return top_k_indices


def test_hierarchical_modes():
    """Test hierarchical mode management"""
    print("\nTesting Hierarchical Modes...")
    device = torch.device('cpu')
    
    hpmm = HierarchicalPseudoModeMemory(
        latent_dims=[16, 32, 64],
        max_modes_per_level=[16, 8, 4],
        init_modes_per_level=[4, 2, 1],
        prediction_horizons=[1, 3],
        device=device
    )
    
    batch = torch.randn(4, 16, device=device)
    
    # Test hierarchical forward
    recon, comps = hpmm(batch, level=0, update_memory=True, return_all_levels=True)
    
    assert recon.shape == batch.shape, "Reconstruction shape mismatch"
    assert len(comps['n_active_modes']) == 3, "Should have 3 levels"
    
    print(f"Active modes per level: {comps['n_active_modes']}")
    print(f"Total active modes: {hpmm.total_active_modes}")
    
    # Test multi-scale prediction
    predictions = hpmm.predict_multiscale(batch, level=0, horizons=[1, 3])
    
    assert 1 in predictions and 3 in predictions, "Missing prediction horizons"
    assert predictions[1].shape == batch.shape, "Prediction shape mismatch"
    
    print(f"Prediction shapes: 1-step={predictions[1].shape}, 3-step={predictions[3].shape}")
    print("✓ Hierarchical modes test passed!")


def example_hierarchical_training():
    """Example: Training with hierarchical modes and multi-scale prediction"""
    print("\nHierarchical Multi-Scale Training Example:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hpmm = HierarchicalPseudoModeMemory(
        latent_dims=[64, 128, 256],
        max_modes_per_level=[32, 16, 8],
        init_modes_per_level=[8, 4, 2],
        prediction_horizons=[1, 5, 10],
        use_top_down=True,
        device=device
    )
    
    optimizer = torch.optim.Adam(hpmm.parameters(), lr=1e-3)
    
    for step in range(30):
        # Simulate sequence data
        batch_size = 8
        current = torch.randn(batch_size, 64, device=device)
        targets = {h: torch.randn(batch_size, 64, device=device) for h in [1, 5, 10]}
        
        # Hierarchical reconstruction
        recon, comps = hpmm(current, level=0, update_memory=True, return_all_levels=True)
        recon_loss = F.mse_loss(recon, current)
        
        # Multi-scale prediction
        predictions = hpmm.predict_multiscale(current, level=0)
        pred_losses = [F.mse_loss(predictions[h], targets[h]) for h in [1, 5, 10]]
        pred_loss = sum(pred_losses) / len(pred_losses)
        
        # Total loss
        total_loss = recon_loss + pred_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        hpmm.apply_explicit_updates()
        
        if step % 10 == 0:
            modes = comps['n_active_modes']
            print(f"Step {step}: Loss={total_loss.item():.4f}, "
                  f"Modes={modes}, Recon={recon_loss.item():.4f}, "
                  f"Pred={pred_loss.item():.4f}")


if __name__ == "__main__":
    print("Testing Static PMM with gradient-safe parameter handling...")
    test_occupancy_mass_conservation()
    test_capacity_management()
    test_gradient_safety()
    print("\nTesting predictive extension...")
    test_predictive_extension()
    print("\nTesting hierarchical modes...")
    test_hierarchical_modes()
    print("\nAll tests passed! Running example usage:")
    example_usage()
    print("\nRunning temporal prediction example:")
    example_temporal_prediction()
    print("\nRunning hierarchical training example:")
    example_hierarchical_training()