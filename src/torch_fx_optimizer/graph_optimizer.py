from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.fx as fx
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerificationError(Exception):
    """Raised when optimized model produces incorrect results."""
    pass


class GraphOptimizer:
    """
    Main entry point for graph optimization.

    Orchestrates tracing, optimization passes, verification, and benchmarking.

    Example:
        >>> model = SimpleMLP()
        >>> optimizer = GraphOptimizer(model, device='cpu')
        >>> optimized = optimizer.optimize(passes=['redundant_ops'])
        >>> optimizer.verify_correctness(test_inputs)
        >>> metrics = optimizer.benchmark(test_inputs)
    """

    def __init__(self, model: nn.Module, device: str = 'mps'):
        """
        Initialize optimizer with a PyTorch model.

        Args:
            model: PyTorch nn.Module to optimize
            device: Device to run on ('cpu', 'mps', 'cuda')

        Raises:
            ValueError: If model cannot be traced
            ValueError: If device is not valid
        """
        # Validate device
        valid_devices = ['cpu', 'mps', 'cuda']
        if device not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}, got '{device}'")

        # Check device availability
        if device == 'mps' and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            device = 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'

        self.device = device
        self.original_model = model.to(device)
        self.traced: Optional[fx.GraphModule] = None
        self.optimized: Optional[fx.GraphModule] = None

        # Trace the model
        try:
            logger.info(f"Tracing model {model.__class__.__name__}...")
            self.traced = fx.symbolic_trace(model)
            self.traced.to(device)
            logger.info(f"Successfully traced model with {len(list(self.traced.graph.nodes))} nodes")
        except Exception as e:
            error_msg = f"Failed to trace model: {str(e)}\n"
            error_msg += "Common causes:\n"
            error_msg += "  - Dynamic control flow (if/for statements)\n"
            error_msg += "  - Non-traceable operations\n"
            error_msg += "  - Custom autograd functions\n"
            error_msg += "Try wrapping non-traceable parts with torch.fx.wrap()"
            raise ValueError(error_msg) from e

        # Verify traced model works
        logger.info("Traced model initialized successfully")

    def optimize(self, passes: List[str]) -> fx.GraphModule:
        """
        Apply optimization passes to the model.

        Args:
            passes: List of pass names (e.g., ['redundant_ops', 'recompute'])

        Returns:
            Optimized GraphModule

        Raises:
            ValueError: If pass name not recognized
            VerificationError: If optimized model produces wrong results

        Example:
            >>> optimized = optimizer.optimize(['redundant_ops', 'recompute'])
        """
        if not isinstance(passes, list):
            raise ValueError(f"passes must be a list, got {type(passes)}")

        # Start with traced graph
        current_graph = self.traced

        # If no passes, return traced model
        if len(passes) == 0:
            logger.info("No optimization passes specified, returning traced model")
            self.optimized = current_graph
            return current_graph

        logger.info(f"Applying {len(passes)} optimization passes: {passes}")

        # Import pass registry
        from torch_fx_optimizer.optimization_pass import get_pass_by_name

        for i, pass_name in enumerate(passes):
            try:
                # Load pass
                logger.info(f"[{i+1}/{len(passes)}] Loading pass '{pass_name}'...")
                optimization_pass = get_pass_by_name(pass_name)

                # Analyze
                logger.info(f"  Analyzing graph...")
                analysis = optimization_pass.analyze(current_graph.graph)
                logger.info(f"  Analysis: {analysis}")

                # Transform
                logger.info(f"  Applying transformation...")
                transformed_graph = optimization_pass.transform(current_graph.graph)

                # Verify transformation
                logger.info(f"  Verifying transformation...")
                if not optimization_pass.verify(transformed_graph):
                    raise VerificationError(
                        f"Pass '{pass_name}' produced invalid graph transformation"
                    )

                # Recompile graph module
                current_graph.recompile()
                logger.info(f"  Pass '{pass_name}' completed successfully")

            except KeyError as e:
                raise ValueError(
                    f"Unknown optimization pass: '{pass_name}'\n"
                    f"Available passes: {get_available_passes()}"
                ) from e
            except Exception as e:
                logger.error(f"Pass '{pass_name}' failed: {str(e)}")
                raise

        self.optimized = current_graph
        logger.info(f"All optimization passes completed successfully")
        return current_graph

    def verify_correctness(
        self,
        test_inputs: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8
    ) -> bool:
        """
        Verify optimized model matches original output.

        Args:
            test_inputs: Sample inputs for testing
            rtol: Relative tolerance for comparison
            atol: Absolute tolerance for comparison

        Returns:
            True if outputs match within tolerance

        Raises:
            AssertionError: If outputs don't match
            RuntimeError: If optimized model not available

        Example:
            >>> test_input = torch.randn(2, 10)
            >>> optimizer.verify_correctness(test_input)
            True
        """
        if self.optimized is None:
            raise RuntimeError(
                "No optimized model available. Call optimize() first."
            )

        # Validate inputs
        if not isinstance(test_inputs, torch.Tensor):
            raise ValueError(f"test_inputs must be a torch.Tensor, got {type(test_inputs)}")

        # Move inputs to device
        test_inputs = test_inputs.to(self.device)

        logger.info("Verifying model correctness...")

        # Run both models
        with torch.no_grad():
            try:
                original_output = self.original_model(test_inputs)
                optimized_output = self.optimized(test_inputs)
            except Exception as e:
                logger.error(f"Error running models: {str(e)}")
                raise

        # Compare outputs
        try:
            torch.testing.assert_close(
                optimized_output,
                original_output,
                rtol=rtol,
                atol=atol,
                msg=f"Optimized model output differs from original"
            )
            logger.info("✓ Verification passed - outputs match within tolerance")
            return True
        except AssertionError as e:
            logger.error(f"✗ Verification failed: {str(e)}")
            # Calculate actual difference for debugging
            max_diff = torch.max(torch.abs(optimized_output - original_output)).item()
            rel_diff = (max_diff / torch.max(torch.abs(original_output)).item()) if torch.max(torch.abs(original_output)).item() > 0 else float('inf')
            logger.error(f"  Max absolute difference: {max_diff}")
            logger.error(f"  Max relative difference: {rel_diff}")
            raise

    def benchmark(
        self,
        test_inputs: torch.Tensor,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark original vs optimized model.

        Args:
            test_inputs: Inputs for benchmarking
            num_runs: Number of iterations for timing
            warmup_runs: Number of warmup iterations (not counted)

        Returns:
            Dictionary with metrics:
            - memory_original_mb: float
            - memory_optimized_mb: float
            - memory_reduction_pct: float
            - time_original_ms: float (median)
            - time_optimized_ms: float (median)
            - time_overhead_pct: float
            - graph_nodes_original: int
            - graph_nodes_optimized: int

        Example:
            >>> metrics = optimizer.benchmark(test_inputs, num_runs=100)
            >>> print(f"Memory reduction: {metrics['memory_reduction_pct']:.1f}%")
        """
        if self.optimized is None:
            raise RuntimeError(
                "No optimized model available. Call optimize() first."
            )

        # Validate inputs
        if not isinstance(test_inputs, torch.Tensor):
            raise ValueError(f"test_inputs must be a torch.Tensor, got {type(test_inputs)}")
        if num_runs < 1:
            raise ValueError(f"num_runs must be >= 1, got {num_runs}")

        test_inputs = test_inputs.to(self.device)

        logger.info(f"Benchmarking with {num_runs} runs ({warmup_runs} warmup)...")

        # Initialize metrics
        metrics = {}

        # Count graph nodes
        original_nodes = len(list(self.traced.graph.nodes))
        optimized_nodes = len(list(self.optimized.graph.nodes))
        metrics['graph_nodes_original'] = original_nodes
        metrics['graph_nodes_optimized'] = optimized_nodes

        # Benchmark timing - Original model
        logger.info("  Benchmarking original model timing...")
        timings_original = []

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.original_model(test_inputs)

        # Actual runs
        for _ in range(num_runs):
            if self.device == 'mps':
                torch.mps.synchronize()
            elif self.device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = self.original_model(test_inputs)

            if self.device == 'mps':
                torch.mps.synchronize()
            elif self.device == 'cuda':
                torch.cuda.synchronize()

            timings_original.append((time.perf_counter() - start) * 1000)  # Convert to ms

        # Benchmark timing - Optimized model
        logger.info("  Benchmarking optimized model timing...")
        timings_optimized = []

        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = self.optimized(test_inputs)

        # Actual runs
        for _ in range(num_runs):
            if self.device == 'mps':
                torch.mps.synchronize()
            elif self.device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            with torch.no_grad():
                _ = self.optimized(test_inputs)

            if self.device == 'mps':
                torch.mps.synchronize()
            elif self.device == 'cuda':
                torch.cuda.synchronize()

            timings_optimized.append((time.perf_counter() - start) * 1000)  # Convert to ms

        # Calculate median times
        time_original_ms = torch.median(torch.tensor(timings_original)).item()
        time_optimized_ms = torch.median(torch.tensor(timings_optimized)).item()

        metrics['time_original_ms'] = time_original_ms
        metrics['time_optimized_ms'] = time_optimized_ms
        metrics['time_overhead_pct'] = ((time_optimized_ms - time_original_ms) / time_original_ms) * 100

        # Benchmark memory
        logger.info("  Benchmarking memory usage...")

        if self.device == 'mps':
            # MPS memory tracking
            torch.mps.empty_cache()
            torch.mps.reset_peak_memory_stats()

            with torch.no_grad():
                _ = self.original_model(test_inputs)

            memory_original = torch.mps.driver_allocated_memory() / (1024 ** 2)  # Convert to MB

            torch.mps.empty_cache()
            torch.mps.reset_peak_memory_stats()

            with torch.no_grad():
                _ = self.optimized(test_inputs)

            memory_optimized = torch.mps.driver_allocated_memory() / (1024 ** 2)  # Convert to MB

        elif self.device == 'cuda':
            # CUDA memory tracking
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = self.original_model(test_inputs)

            memory_original = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            with torch.no_grad():
                _ = self.optimized(test_inputs)

            memory_optimized = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        else:
            # CPU doesn't have reliable memory tracking per-operation
            logger.warning("  Memory tracking not available for CPU device")
            memory_original = 0.0
            memory_optimized = 0.0

        metrics['memory_original_mb'] = memory_original
        metrics['memory_optimized_mb'] = memory_optimized

        if memory_original > 0:
            metrics['memory_reduction_pct'] = ((memory_original - memory_optimized) / memory_original) * 100
        else:
            metrics['memory_reduction_pct'] = 0.0

        # Log summary
        logger.info("Benchmark complete:")
        logger.info(f"  Graph nodes: {original_nodes} → {optimized_nodes}")
        logger.info(f"  Median time: {time_original_ms:.3f}ms → {time_optimized_ms:.3f}ms ({metrics['time_overhead_pct']:+.1f}%)")
        if memory_original > 0:
            logger.info(f"  Memory: {memory_original:.1f}MB → {memory_optimized:.1f}MB ({metrics['memory_reduction_pct']:+.1f}%)")

        return metrics


def get_available_passes() -> List[str]:
    """
    Get list of available optimization passes.

    Returns:
        List of pass names
    """
    # This will be implemented when passes are added
    return []
