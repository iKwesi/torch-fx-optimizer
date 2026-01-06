from abc import ABC, abstractmethod
import torch.fx as fx
from typing import Dict, Any, Type
import logging

logger = logging.getLogger(__name__)


class OptimizationPass(ABC):
    """
    Abstract base class for all optimization passes.

    All passes must implement: analyze(), transform(), verify()

    The lifecycle of a pass is:
    1. analyze() - Identify optimization opportunities
    2. transform() - Apply the optimization to the graph
    3. verify() - Ensure the transformation is valid

    Example:
        >>> class MyPass(OptimizationPass):
        ...     def analyze(self, graph):
        ...         return {'opportunities': [], 'stats': {}, 'safe': True}
        ...
        ...     def transform(self, graph):
        ...         # Modify graph here
        ...         return graph
        ...
        ...     def verify(self, graph):
        ...         graph.lint()
        ...         return True
        ...
        ...     @property
        ...     def name(self):
        ...         return 'my_pass'
    """

    @abstractmethod
    def analyze(self, graph: fx.Graph) -> Dict[str, Any]:
        """
        Analyze graph to find optimization opportunities.

        This method should be read-only - it should NOT modify the graph.
        It should identify patterns, nodes, or operations that can be optimized.

        Args:
            graph: torch.fx Graph to analyze

        Returns:
            Dictionary with analysis results:
            {
                'opportunities': List of nodes/patterns to optimize,
                'stats': Statistics about potential savings,
                'safe': Whether optimization is safe to apply
            }

        Example:
            >>> analysis = pass.analyze(graph)
            >>> if analysis['safe'] and len(analysis['opportunities']) > 0:
            ...     pass.transform(graph)
        """
        pass

    @abstractmethod
    def transform(self, graph: fx.Graph) -> fx.Graph:
        """
        Apply optimization transformation to the graph.

        This method modifies the graph in-place by:
        - Removing redundant nodes
        - Inserting new nodes
        - Rewiring connections between nodes
        - Updating metadata

        IMPORTANT: After modifying the graph, you should:
        1. Update node.users for affected nodes
        2. Call graph.eliminate_dead_code() if you created dead nodes
        3. Ensure graph.lint() passes

        Args:
            graph: torch.fx Graph to transform (modified in-place)

        Returns:
            The modified graph (same object, for chaining)

        Side effects:
            Modifies graph in-place

        Example:
            >>> # Remove a redundant node
            >>> for node in graph.nodes:
            ...     if is_redundant(node):
            ...         node.replace_all_uses_with(node.args[0])
            ...         graph.erase_node(node)
            >>> graph.eliminate_dead_code()
            >>> return graph
        """
        pass

    @abstractmethod
    def verify(self, graph: fx.Graph) -> bool:
        """
        Verify the transformation is valid.

        This method checks that the graph is well-formed after transformation.
        It should verify:
        1. No dangling references (nodes that reference deleted nodes)
        2. Graph structure is valid (graph.lint() passes)
        3. Optimization-specific invariants are maintained

        Args:
            graph: Transformed graph to verify

        Returns:
            True if graph is valid after transformation

        Raises:
            May raise exceptions if verification fails (e.g., from graph.lint())

        Checks:
            - No dangling references
            - Graph is well-formed (graph.lint() passes)
            - Optimization invariants maintained

        Example:
            >>> def verify(self, graph):
            ...     try:
            ...         graph.lint()
            ...         # Check custom invariants
            ...         assert all(node in graph.nodes for node in graph.nodes)
            ...         return True
            ...     except Exception as e:
            ...         logger.error(f"Verification failed: {e}")
            ...         return False
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this pass (e.g., 'redundant_ops').

        This name is used to:
        - Identify the pass in logging
        - Register the pass in the pass registry
        - Reference the pass in optimization pipelines

        Returns:
            String identifier for this pass

        Example:
            >>> @property
            ... def name(self):
            ...     return 'constant_folding'
        """
        pass


# Pass registry - maps pass names to pass classes
_PASS_REGISTRY: Dict[str, Type[OptimizationPass]] = {}


def register_pass(pass_class: Type[OptimizationPass]) -> Type[OptimizationPass]:
    """
    Register an optimization pass.

    Use this as a decorator on your pass class:

    Example:
        >>> @register_pass
        ... class MyPass(OptimizationPass):
        ...     @property
        ...     def name(self):
        ...         return 'my_pass'
        ...     ...

    Args:
        pass_class: The pass class to register

    Returns:
        The same class (for decorator chaining)

    Raises:
        ValueError: If a pass with this name already exists
    """
    # Create instance to get name
    try:
        instance = pass_class()
        pass_name = instance.name

        if pass_name in _PASS_REGISTRY:
            logger.warning(
                f"Pass '{pass_name}' already registered, overwriting with {pass_class.__name__}"
            )

        _PASS_REGISTRY[pass_name] = pass_class
        logger.debug(f"Registered optimization pass: '{pass_name}' ({pass_class.__name__})")

    except Exception as e:
        raise ValueError(
            f"Failed to register pass {pass_class.__name__}: {str(e)}"
        ) from e

    return pass_class


def get_pass_by_name(name: str) -> OptimizationPass:
    """
    Get an optimization pass instance by name.

    Args:
        name: The name of the pass (e.g., 'redundant_ops')

    Returns:
        Instance of the requested pass

    Raises:
        KeyError: If pass name is not registered

    Example:
        >>> pass_instance = get_pass_by_name('redundant_ops')
        >>> analysis = pass_instance.analyze(graph)
    """
    if name not in _PASS_REGISTRY:
        available = list(_PASS_REGISTRY.keys())
        raise KeyError(
            f"Optimization pass '{name}' not found.\n"
            f"Available passes: {available}\n"
            f"Did you forget to register it with @register_pass?"
        )

    pass_class = _PASS_REGISTRY[name]
    return pass_class()


def get_available_passes() -> Dict[str, Type[OptimizationPass]]:
    """
    Get all registered optimization passes.

    Returns:
        Dictionary mapping pass names to pass classes

    Example:
        >>> passes = get_available_passes()
        >>> print(f"Available passes: {list(passes.keys())}")
    """
    return _PASS_REGISTRY.copy()


def unregister_pass(name: str) -> None:
    """
    Unregister an optimization pass.

    Useful for testing or dynamically managing passes.

    Args:
        name: The name of the pass to unregister

    Raises:
        KeyError: If pass name is not registered

    Example:
        >>> unregister_pass('my_pass')
    """
    if name not in _PASS_REGISTRY:
        raise KeyError(f"Pass '{name}' not registered")

    del _PASS_REGISTRY[name]
    logger.debug(f"Unregistered optimization pass: '{name}'")
