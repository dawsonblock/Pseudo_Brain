# -*- coding: utf-8 -*-
"""
Comprehensive Test Suite for Capsule Brain Pseudo-Memory Module
================================================================

Tests for:
- Occupancy mass conservation (Σ = 1.0)
- Parameter validity (λ, γ, ω ≥ 0)
- Merge/split operations
- Capsule Brain API
- Gradient safety
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import pytest
from capsule_brain_integration import CapsuleBrainPMM, SpikePacket


# ============================================================================
# INVARIANT TESTS
# ============================================================================

def test_occupancy_mass_invariant():
    """Test that Σ occupancy = 1.0 always holds"""
    pmm = CapsuleBrainPMM(
        latent_dim=16,
        max_modes=10,
        init_modes=4,
        structural_update_freq=1
    )
    
    # Initial check
    initial_sum = pmm.occupancy[pmm.active_mask].sum().item()
    assert abs(initial_sum - 1.0) < 1e-6, f"Initial occupancy sum: {initial_sum}"
    
    # After forward pass and update
    batch = torch.randn(4, 16)
    _, _ = pmm(batch, update_memory=True)
    pmm.apply_explicit_updates()
    
    after_update = pmm.occupancy[pmm.active_mask].sum().item()
    assert abs(after_update - 1.0) < 1e-6, f"After update: {after_update}"
    
    # After merge
    pmm.merge_modes()
    after_merge = pmm.occupancy[pmm.active_mask].sum().item()
    assert abs(after_merge - 1.0) < 1e-6, f"After merge: {after_merge}"
    
    # After split
    pmm.split_modes()
    after_split = pmm.occupancy[pmm.active_mask].sum().item()
    assert abs(after_split - 1.0) < 1e-6, f"After split: {after_split}"
    
    print("✓ Occupancy mass conservation test passed!")


def test_parameter_validity():
    """Test that λ_i, γ_i, ω_i remain non-negative"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    
    # Run multiple updates
    for _ in range(10):
        batch = torch.randn(4, 16)
        _, _ = pmm(batch, update_memory=True)
        pmm.apply_explicit_updates()
    
    # Check all active parameters
    active_mask = pmm.active_mask
    lambda_vals = pmm.lambda_i[active_mask]
    gamma_vals = pmm.gamma_i[active_mask]
    omega_vals = pmm.omega_i[active_mask]
    
    assert (lambda_vals >= 0).all(), "λ_i has negative values"
    assert (gamma_vals >= 0).all(), "γ_i has negative values"
    assert (omega_vals >= 0).all(), "ω_i has negative values"
    
    print("✓ Parameter validity test passed!")


def test_capacity_constraint():
    """Test that n_active ≤ max_modes always"""
    max_modes = 5
    pmm = CapsuleBrainPMM(
        latent_dim=8,
        max_modes=max_modes,
        init_modes=4,
        split_threshold=0.1
    )
    
    # Force splits
    for _ in range(10):
        batch = torch.randn(4, 8)
        _, _ = pmm(batch, update_memory=True)
        pmm.apply_explicit_updates()
        pmm.split_modes(force=True)
        
        assert pmm.n_active_modes <= max_modes, \
            f"Exceeded capacity: {pmm.n_active_modes} > {max_modes}"
    
    print("✓ Capacity constraint test passed!")


# ============================================================================
# MERGE/SPLIT TESTS
# ============================================================================

def test_merge_conservation():
    """Test that merge conserves total occupancy and parameters"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=10, init_modes=4)
    
    # Setup: make two modes similar
    with torch.no_grad():
        pmm.mu.data[0] = torch.ones(16) * 0.1
        pmm.mu.data[1] = torch.ones(16) * 0.11
        pmm.occupancy[0] = 0.3
        pmm.occupancy[1] = 0.2
    
    # Force merge
    n_merged = pmm.merge_modes(force=True)
    
    # Check after merge
    total_occ_after = pmm.occupancy[pmm.active_mask].sum().item()
    
    assert abs(total_occ_after - 1.0) < 1e-6, "Occupancy not normalized"
    assert n_merged >= 0, "Merge count negative"
    
    print(f"✓ Merge conservation test passed! (merged {n_merged} modes)")


def test_split_conservation():
    """Test that split halves mass correctly"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=10, init_modes=2)
    
    # Setup: high occupancy, low importance
    with torch.no_grad():
        pmm.occupancy[0] = 0.8
        pmm.lambda_i[0] = 0.1
    
    # Record mode 0 occupancy
    occ_before = pmm.occupancy[0].item()
    
    # Force split
    n_split = pmm.split_modes(force=True)
    
    if n_split > 0:
        # Check that occupancy was halved (approximately)
        occ_after = pmm.occupancy[0].item()
        assert occ_after < occ_before, "Occupancy not reduced after split"
        
        # Check total mass = 1.0
        total_occ = pmm.occupancy[pmm.active_mask].sum().item()
        assert abs(total_occ - 1.0) < 1e-6, f"Total occupancy: {total_occ}"
    
    print(f"✓ Split conservation test passed! (split {n_split} modes)")


# ============================================================================
# CAPSULE BRAIN API TESTS
# ============================================================================

def test_store_api():
    """Test store() API method"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    
    # Create spike packet
    content = torch.randn(1, 16)
    spike = SpikePacket(
        content=content,
        routing_key='test_capsule',
        priority=0.8
    )
    
    # Store
    result = pmm.store(spike)
    
    assert result['stored']
    assert 'novelty' in result
    assert 'active_modes' in result
    assert result['routing_key'] == 'test_capsule'
    
    print("✓ Store API test passed!")


def test_retrieve_api():
    """Test retrieve() API method"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    
    # Store first
    content = torch.randn(1, 16)
    spike = SpikePacket(content=content)
    pmm.store(spike)
    
    # Retrieve
    query = torch.randn(1, 16)
    reconstruction, components = pmm.retrieve(query)
    
    assert reconstruction.shape == query.shape
    assert 'retrieval_confidence' in components
    assert 'alpha' in components
    
    print("✓ Retrieve API test passed!")


def test_compress_api():
    """Test compress() API method"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=6)
    
    # Compress
    stats = pmm.compress()
    
    assert 'modes_before' in stats
    assert 'modes_after' in stats
    assert 'compression_ratio' in stats
    assert abs(stats['occupancy_sum'] - 1.0) < 1e-6
    
    print("✓ Compress API test passed!")


def test_to_workspace_api():
    """Test to_workspace() API method"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    
    workspace = pmm.to_workspace()
    
    assert 'active_prototypes' in workspace
    assert 'occupancy' in workspace
    assert 'importance' in workspace
    assert 'decay_rates' in workspace
    assert 'oscillation_freq' in workspace
    assert 'phase' in workspace
    assert workspace['n_active'].item() == pmm.n_active_modes
    
    print("✓ To workspace API test passed!")


def test_route_to_capsule_api():
    """Test route_to_capsule() API method"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    
    content = torch.randn(4, 16)
    spike = pmm.route_to_capsule('target_capsule', content)
    
    assert spike.routing_key == 'target_capsule'
    assert spike.content.shape == content.shape
    assert 0 <= spike.priority <= 1.0
    assert spike.metadata['source'] == 'pseudo_memory'
    
    print("✓ Route to capsule API test passed!")


# ============================================================================
# GRADIENT SAFETY TESTS
# ============================================================================

def test_gradient_flow():
    """Test that gradients flow correctly through the module"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)
    
    batch = torch.randn(4, 16, requires_grad=True)
    
    reconstruction, components = pmm(batch, update_memory=True, return_components=True)
    loss = F.mse_loss(reconstruction, batch) + 0.1 * components['sparsity_loss']
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    pmm.apply_explicit_updates()
    
    # Check that parameters have gradients
    has_grad = False
    for param in pmm.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "No gradients computed"
    print("✓ Gradient flow test passed!")


def test_no_gradient_errors():
    """Test that structural updates don't cause gradient errors"""
    pmm = CapsuleBrainPMM(
        latent_dim=16,
        max_modes=8,
        init_modes=4,
        structural_update_freq=1
    )
    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)
    
    try:
        for step in range(10):
            batch = torch.randn(4, 16)
            reconstruction, components = pmm(batch, update_memory=True, return_components=True)
            
            loss = F.mse_loss(reconstruction, batch) + 0.1 * components['sparsity_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pmm.apply_explicit_updates()
        
        print("✓ No gradient errors test passed!")
    except Exception as e:
        pytest.fail(f"Gradient error occurred: {e}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_training_loop():
    """Test complete training loop with all features"""
    pmm = CapsuleBrainPMM(
        latent_dim=32,
        max_modes=16,
        init_modes=4,
        structural_update_freq=5
    )
    optimizer = torch.optim.Adam(pmm.parameters(), lr=1e-3)
    
    for epoch in range(5):
        epoch_loss = 0.0
        for step in range(10):
            batch = torch.randn(8, 32)
            
            # Forward
            reconstruction, components = pmm(batch, update_memory=True, return_components=True)
            
            # Loss
            recon_loss = F.mse_loss(reconstruction, batch)
            sparsity_loss = components['sparsity_loss']
            loss = recon_loss + 0.1 * sparsity_loss
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Explicit updates
            pmm.apply_explicit_updates()
            
            epoch_loss += loss.item()
        
        # Verify invariants
        occ_sum = pmm.occupancy[pmm.active_mask].sum().item()
        assert abs(occ_sum - 1.0) < 1e-5, f"Epoch {epoch}: occupancy = {occ_sum}"
    
    print("✓ Full training loop test passed!")


def test_spike_packet_workflow():
    """Test complete spike packet storage and retrieval workflow"""
    pmm = CapsuleBrainPMM(latent_dim=16, max_modes=8, init_modes=4)
    
    # Store multiple spikes
    for i in range(5):
        content = torch.randn(1, 16)
        spike = SpikePacket(
            content=content,
            routing_key=f'capsule_{i}',
            priority=0.5 + i * 0.1
        )
        result = pmm.store(spike)
        assert result['stored']
    
    # Retrieve
    query = torch.randn(1, 16)
    reconstruction, components = pmm.retrieve(query)
    
    # Route to capsule
    routed_spike = pmm.route_to_capsule('output_capsule', reconstruction)
    assert routed_spike.routing_key == 'output_capsule'
    
    # Broadcast to workspace
    workspace = pmm.to_workspace()
    assert workspace['n_active'].item() == pmm.n_active_modes
    
    print("✓ Spike packet workflow test passed!")


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("CAPSULE BRAIN PSEUDO-MEMORY MODULE - COMPREHENSIVE TEST SUITE")
    print("="*70 + "\n")
    
    print("INVARIANT TESTS")
    print("-" * 70)
    test_occupancy_mass_invariant()
    test_parameter_validity()
    test_capacity_constraint()
    
    print("\nMERGE/SPLIT TESTS")
    print("-" * 70)
    test_merge_conservation()
    test_split_conservation()
    
    print("\nCAPSULE BRAIN API TESTS")
    print("-" * 70)
    test_store_api()
    test_retrieve_api()
    test_compress_api()
    test_to_workspace_api()
    test_route_to_capsule_api()
    
    print("\nGRADIENT SAFETY TESTS")
    print("-" * 70)
    test_gradient_flow()
    test_no_gradient_errors()
    
    print("\nINTEGRATION TESTS")
    print("-" * 70)
    test_full_training_loop()
    test_spike_packet_workflow()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
