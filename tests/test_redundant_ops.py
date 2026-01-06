import pytest
import torch
import torch.nn as nn
from torch_fx_optimizer.graph_optimizer import GraphOptimizer
from torch_fx_optimizer.passes.redundant_ops import RedundantOpElimination


class RedundantModel(nn.Module):
    '''Model with intentional redundant operations.'''
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)

    def forward(self, x):
        # Call conv twice with same input - REDUNDANT!
        a = self.conv(x)
        b = self.conv(x)  # Duplicate
        return a + b


class RedundantFunctionModel(nn.Module):
    '''Model with redundant function calls.'''
    def forward(self, x):
        # Call relu twice with same input
        a = torch.relu(x)
        b = torch.relu(x)  # Duplicate
        return a + b


def test_redundancy_detection():
    '''Test that redundant operations are detected.'''
    model = RedundantModel()
    optimizer = GraphOptimizer(model, device='cpu')

    pass_instance = RedundantOpElimination()
    analysis = pass_instance.analyze(optimizer.traced.graph)

    # Should find at least 1 redundant operation
    assert len(analysis['opportunities']) >= 1, "Should detect redundant operations"
    assert analysis['stats']['redundant_nodes'] >= 1, "Stats should show redundant nodes"
    print(f"✅ Detected {len(analysis['opportunities'])} redundant operations")


def test_redundancy_elimination_modules():
    '''Test eliminating redundant module calls.'''
    model = RedundantModel()
    test_input = torch.randn(1, 3, 32, 32)

    optimizer = GraphOptimizer(model, device='cpu')

    # Get original graph size
    original_nodes = len(list(optimizer.traced.graph.nodes))
    print(f"Original graph has {original_nodes} nodes")

    # Apply optimization
    optimized = optimizer.optimize(passes=['redundant_ops'])

    # Graph should have fewer nodes
    optimized_nodes = len(list(optimized.graph.nodes))
    print(f"Optimized graph has {optimized_nodes} nodes")
    assert optimized_nodes < original_nodes, "Graph should have fewer nodes after optimization"

    # Output should be identical
    with torch.no_grad():
        original_output = model(test_input)
        optimized_output = optimized(test_input)

    torch.testing.assert_close(
        original_output,
        optimized_output,
        msg="Optimized model output should match original"
    )
    print("✅ Redundancy eliminated and correctness verified")


def test_redundancy_elimination_functions():
    '''Test eliminating redundant function calls.'''
    model = RedundantFunctionModel()
    test_input = torch.randn(2, 10)

    optimizer = GraphOptimizer(model, device='cpu')

    # Count relu operations before
    original_relu_count = sum(
        1 for node in optimizer.traced.graph.nodes
        if node.op == 'call_function' and 'relu' in str(node.target)
    )
    print(f"Original model has {original_relu_count} relu operations")

    # Apply optimization
    optimized = optimizer.optimize(passes=['redundant_ops'])

    # Count relu operations after
    optimized_relu_count = sum(
        1 for node in optimized.graph.nodes
        if node.op == 'call_function' and 'relu' in str(node.target)
    )
    print(f"Optimized model has {optimized_relu_count} relu operations")

    # Should have fewer relu operations
    assert optimized_relu_count < original_relu_count, "Should eliminate redundant relu calls"

    # Output should be identical
    with torch.no_grad():
        original_output = model(test_input)
        optimized_output = optimized(test_input)

    torch.testing.assert_close(
        original_output,
        optimized_output,
        msg="Optimized function model output should match original"
    )
    print("✅ Function redundancy eliminated successfully")


def test_no_false_positives():
    '''Test that non-redundant operations are not eliminated.'''
    class NonRedundantModel(nn.Module):
        def forward(self, x):
            a = torch.relu(x)
            b = torch.relu(a)  # Different input - NOT redundant
            return a + b

    model = NonRedundantModel()
    optimizer = GraphOptimizer(model, device='cpu')

    original_nodes = len(list(optimizer.traced.graph.nodes))
    optimized = optimizer.optimize(passes=['redundant_ops'])
    optimized_nodes = len(list(optimized.graph.nodes))

    # Should NOT eliminate anything
    assert optimized_nodes == original_nodes, "Should not eliminate non-redundant operations"
    print("✅ No false positives - non-redundant operations preserved")


def test_multiple_duplicates():
    '''Test handling multiple redundant operations.'''
    class MultiRedundantModel(nn.Module):
        def forward(self, x):
            # Three identical relu calls
            a = torch.relu(x)
            b = torch.relu(x)  # Duplicate 1
            c = torch.relu(x)  # Duplicate 2
            return a + b + c

    model = MultiRedundantModel()
    test_input = torch.randn(2, 10)

    optimizer = GraphOptimizer(model, device='cpu')
    optimized = optimizer.optimize(passes=['redundant_ops'])

    # Should eliminate 2 out of 3 relu calls
    relu_count = sum(
        1 for node in optimized.graph.nodes
        if node.op == 'call_function' and 'relu' in str(node.target)
    )
    assert relu_count == 1, f"Expected 1 relu, got {relu_count}"
    print("✅ Multiple duplicates handled correctly")

    # Verify correctness
    with torch.no_grad():
        original_output = model(test_input)
        optimized_output = optimized(test_input)

    torch.testing.assert_close(
        original_output,
        optimized_output,
        msg="Multi-duplicate model should produce correct output"
    )


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_different_batch_sizes(batch_size):
    '''Test that optimization works with different batch sizes.'''
    model = RedundantModel()
    test_input = torch.randn(batch_size, 3, 32, 32)

    optimizer = GraphOptimizer(model, device='cpu')
    optimized = optimizer.optimize(passes=['redundant_ops'])

    with torch.no_grad():
        original_output = model(test_input)
        optimized_output = optimized(test_input)

    torch.testing.assert_close(
        original_output,
        optimized_output,
        msg=f"Should work correctly with batch_size={batch_size}"
    )
    print(f"✅ Batch size {batch_size} works correctly")


def test_pass_returns_optimized_graph():
    '''Test that optimize() returns the optimized GraphModule.'''
    model = RedundantModel()
    optimizer = GraphOptimizer(model, device='cpu')

    result = optimizer.optimize(passes=['redundant_ops'])

    assert result is not None, "optimize() should return a result"
    assert isinstance(result, torch.fx.GraphModule), "Should return a GraphModule"
    assert optimizer.optimized is not None, "Should store optimized model"
    assert optimizer.optimized is result, "Returned value should match stored optimized"
    print("✅ Pass returns correct optimized graph")


def test_invalid_pass_name():
    '''Test that invalid pass names raise appropriate errors.'''
    model = RedundantModel()
    optimizer = GraphOptimizer(model, device='cpu')

    with pytest.raises(ValueError, match="Unknown optimization pass"):
        optimizer.optimize(passes=['nonexistent_pass'])

    print("✅ Invalid pass name raises ValueError")
