"""
torch_fx_optimizer: A PyTorch FX graph optimization framework.

Main Components:
- GraphOptimizer: Main entry point for optimizing models
- OptimizationPass: Base class for creating optimization passes
- Pass registry functions: register_pass, get_pass_by_name, get_available_passes
"""

from torch_fx_optimizer.graph_optimizer import GraphOptimizer, VerificationError
from torch_fx_optimizer.optimization_pass import (
    OptimizationPass,
    register_pass,
    get_pass_by_name,
    get_available_passes,
    unregister_pass,
)

__all__ = [
    'GraphOptimizer',
    'VerificationError',
    'OptimizationPass',
    'register_pass',
    'get_pass_by_name',
    'get_available_passes',
    'unregister_pass',
]
