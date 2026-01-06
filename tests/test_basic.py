"""
Basic tests for core infrastructure.

Tests the GraphOptimizer class initialization, tracing, and basic functionality.
"""

import pytest
import torch
import torch.nn as nn
from torch_fx_optimizer.graph_optimizer import GraphOptimizer, VerificationError


def test_can_trace_simple_model(simple_mlp, device):
    """Test that we can trace a simple model without errors."""
    optimizer = GraphOptimizer(simple_mlp, device=device)
    assert optimizer.traced is not None
    assert isinstance(optimizer.traced, torch.fx.GraphModule)
    assert len(list(optimizer.traced.graph.nodes)) > 0


def test_traced_model_matches_original(simple_mlp, test_input_mlp, device, rtol, atol):
    """Test that traced model produces same output as original."""
    optimizer = GraphOptimizer(simple_mlp, device=device)

    with torch.no_grad():
        original_output = simple_mlp(test_input_mlp)
        traced_output = optimizer.traced(test_input_mlp)

    torch.testing.assert_close(original_output, traced_output, rtol=rtol, atol=atol)


def test_empty_optimization_passes(simple_mlp, test_input_mlp, device):
    """Test that applying no passes returns working model."""
    optimizer = GraphOptimizer(simple_mlp, device=device)
    optimized = optimizer.optimize(passes=[])

    # Should still work
    with torch.no_grad():
        output = optimized(test_input_mlp)

    assert output.shape == (2, 10)


def test_verification_passes_for_unoptimized_model(simple_mlp, test_input_mlp, device, rtol, atol):
    """Test that verification passes when comparing traced to original model."""
    optimizer = GraphOptimizer(simple_mlp, device=device)
    optimizer.optimized = optimizer.traced  # Set optimized to traced (no changes)

    # Verification should pass
    result = optimizer.verify_correctness(test_input_mlp, rtol=rtol, atol=atol)
    assert result is True


def test_benchmark_returns_valid_metrics(simple_mlp, test_input_mlp, device):
    """Test that benchmark returns dictionary with expected metrics."""
    optimizer = GraphOptimizer(simple_mlp, device=device)
    optimized = optimizer.optimize(passes=[])  # No optimization

    metrics = optimizer.benchmark(test_input_mlp, num_runs=10, warmup_runs=2)

    # Check all expected keys are present
    expected_keys = [
        'memory_original_mb',
        'memory_optimized_mb',
        'memory_reduction_pct',
        'time_original_ms',
        'time_optimized_ms',
        'time_overhead_pct',
        'graph_nodes_original',
        'graph_nodes_optimized',
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing key: {key}"

    # Check metrics are reasonable
    assert metrics['time_original_ms'] > 0
    assert metrics['time_optimized_ms'] > 0
    assert metrics['graph_nodes_original'] > 0
    assert metrics['graph_nodes_optimized'] > 0


def test_tracing_failure_raises_clear_error():
    """Test that tracing failure raises clear error message."""

    class UntracableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 10)

        def forward(self, x):
            # Dynamic control flow - not traceable
            if x.sum() > 0:
                return self.fc(x)
            else:
                return x

    model = UntracableModel()

    with pytest.raises(ValueError) as exc_info:
        optimizer = GraphOptimizer(model, device='cpu')

    # Check error message is helpful
    error_msg = str(exc_info.value)
    assert "Failed to trace model" in error_msg
    assert "control flow" in error_msg.lower() or "torch.fx.wrap" in error_msg


def test_invalid_device_raises_error(simple_mlp):
    """Test that invalid device raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        optimizer = GraphOptimizer(simple_mlp, device='invalid_device')

    assert "Device must be one of" in str(exc_info.value)


def test_verify_without_optimization_raises_error(simple_mlp, test_input_mlp, device):
    """Test that verify_correctness raises error if optimize() not called."""
    optimizer = GraphOptimizer(simple_mlp, device=device)

    # optimized is None since we haven't called optimize()
    with pytest.raises(RuntimeError) as exc_info:
        optimizer.verify_correctness(test_input_mlp)

    assert "No optimized model available" in str(exc_info.value)


def test_benchmark_without_optimization_raises_error(simple_mlp, test_input_mlp, device):
    """Test that benchmark raises error if optimize() not called."""
    optimizer = GraphOptimizer(simple_mlp, device=device)

    # optimized is None since we haven't called optimize()
    with pytest.raises(RuntimeError) as exc_info:
        optimizer.benchmark(test_input_mlp, num_runs=10)

    assert "No optimized model available" in str(exc_info.value)


def test_cnn_model_can_be_traced(cnn_model, test_input_cnn, device):
    """Test that CNN model can be traced successfully."""
    optimizer = GraphOptimizer(cnn_model, device=device)
    assert optimizer.traced is not None

    # Test it produces correct output
    with torch.no_grad():
        original_output = cnn_model(test_input_cnn)
        traced_output = optimizer.traced(test_input_cnn)

    torch.testing.assert_close(original_output, traced_output, rtol=1e-5, atol=1e-6)


def test_deep_model_can_be_traced(deep_model, test_input_deep, device):
    """Test that deep model can be traced successfully."""
    optimizer = GraphOptimizer(deep_model, device=device)
    assert optimizer.traced is not None

    # Should have many nodes (multiple layers)
    num_nodes = len(list(optimizer.traced.graph.nodes))
    assert num_nodes > 10  # At least one node per layer

    # Test it produces correct output
    with torch.no_grad():
        original_output = deep_model(test_input_deep)
        traced_output = optimizer.traced(test_input_deep)

    torch.testing.assert_close(original_output, traced_output, rtol=1e-5, atol=1e-6)


def test_optimize_with_invalid_pass_name_raises_error(simple_mlp, device):
    """Test that optimize() raises error for unknown pass names."""
    optimizer = GraphOptimizer(simple_mlp, device=device)

    with pytest.raises(ValueError) as exc_info:
        optimizer.optimize(['nonexistent_pass'])

    error_msg = str(exc_info.value)
    assert "Unknown optimization pass" in error_msg or "not found" in error_msg


def test_optimize_with_non_list_raises_error(simple_mlp, device):
    """Test that optimize() raises error if passes is not a list."""
    optimizer = GraphOptimizer(simple_mlp, device=device)

    with pytest.raises(ValueError) as exc_info:
        optimizer.optimize('redundant_ops')  # String instead of list

    assert "must be a list" in str(exc_info.value)
