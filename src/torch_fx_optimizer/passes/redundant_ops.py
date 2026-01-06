import torch
import torch.fx as fx
from typing import Dict, Any, Tuple, List
import logging

from torch_fx_optimizer.optimization_pass import OptimizationPass, register_pass

logger = logging.getLogger(__name__)


@register_pass
class RedundantOpElimination(OptimizationPass):
    '''
    Eliminates redundant operations in the computation graph.

    A redundant operation is one that:
    1. Has the same operation type as another node
    2. Has identical inputs (same node references)
    3. Is a pure operation (deterministic, no side effects)

    Example:
        x = conv(input)
        y = conv(input)  # Redundant!
        z = x + y

    Optimized:
        x = conv(input)
        y = x  # Reuse
        z = x + y
    '''

    # Operations that are safe to deduplicate (pure, no side effects)
    PURE_OPERATIONS = {
        torch.relu,
        torch.nn.functional.relu,
        torch.add,
        torch.mul,
        torch.matmul,
        torch.nn.functional.linear,
        torch.nn.functional.conv2d,
        torch.cat,
        torch.sum,
        torch.mean,
        # Add more as needed
    }

    # Operations that should NOT be deduplicated
    IMPURE_OPERATIONS = {
        torch.nn.functional.dropout,  # Random
        torch.nn.functional.dropout2d,
        'dropout',  # Module name
        'batchnorm',  # Has running stats
        'batch_norm',
    }

    def analyze(self, graph: fx.Graph) -> Dict[str, Any]:
        '''
        Find redundant operations in the graph.

        Returns:
            {
                'opportunities': List[(duplicate_node, original_node)],
                'stats': {
                    'total_nodes': int,
                    'redundant_nodes': int,
                    'potential_savings': str
                },
                'safe': bool
            }
        '''
        opportunities = []
        seen_ops = {}  # operation_key -> node

        for node in graph.nodes:
            # Only analyze function calls and module calls
            if node.op not in ['call_function', 'call_module']:
                continue

            # Check if this is a pure operation
            if not self._is_pure_operation(node):
                logger.debug(f"Skipping impure operation: {node.name}")
                continue

            # Compute operation key (structural hash)
            op_key = self._compute_operation_key(node)

            if op_key in seen_ops:
                # Found a duplicate!
                original_node = seen_ops[op_key]
                opportunities.append((node, original_node))
                logger.info(f"Found redundant operation: {node.name} duplicates {original_node.name}")
            else:
                # First time seeing this operation
                seen_ops[op_key] = node

        return {
            'opportunities': opportunities,
            'stats': {
                'total_nodes': len(list(graph.nodes)),
                'redundant_nodes': len(opportunities),
                'potential_savings': f"{len(opportunities)} operations"
            },
            'safe': True  # This pass is always safe if it detects redundancy
        }

    def transform(self, graph: fx.Graph) -> fx.Graph:
        '''
        Remove redundant operations from the graph.

        For each redundant operation:
        1. Replace all uses with the original node
        2. Remove the redundant node from graph
        '''
        # Run analysis to find redundancies
        analysis = self.analyze(graph)
        opportunities = analysis['opportunities']

        if not opportunities:
            logger.info("No redundant operations found")
            return graph

        logger.info(f"Eliminating {len(opportunities)} redundant operations")

        # Remove duplicates
        for duplicate_node, original_node in opportunities:
            logger.debug(f"Replacing {duplicate_node.name} with {original_node.name}")

            # Replace all uses of duplicate with original
            duplicate_node.replace_all_uses_with(original_node)

            # Remove the now-unused duplicate node
            graph.erase_node(duplicate_node)

        # Clean up the graph
        graph.lint()  # Verify graph is valid

        logger.info(f"Successfully eliminated {len(opportunities)} redundant operations")
        return graph

    def verify(self, graph: fx.Graph) -> bool:
        '''
        Verify the graph is still valid after transformation.

        Checks:
        1. Graph passes lint (no dangling references)
        2. All nodes are reachable
        3. No duplicate operation keys remain (optional sanity check)
        '''
        try:
            # Check graph is well-formed
            graph.lint()

            # Verify all nodes are reachable from placeholders
            reachable = self._check_reachability(graph)
            if not reachable:
                logger.error("Graph has unreachable nodes after transformation")
                return False

            logger.debug("Graph verification passed")
            return True

        except Exception as e:
            logger.error(f"Graph verification failed: {e}")
            return False

    @property
    def name(self) -> str:
        return 'redundant_ops'

    # Helper methods

    def _compute_operation_key(self, node: fx.Node) -> Tuple:
        '''
        Compute a structural key for the operation.

        Two nodes with the same key are considered equivalent.

        Key includes:
        - Operation type (function or module name)
        - Input node names (for structural identity)
        - Keyword arguments

        Returns:
            Tuple that can be used as dict key
        '''
        # Get operation identifier
        if node.op == 'call_function':
            op_id = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
        elif node.op == 'call_module':
            op_id = node.target  # Module name
        else:
            op_id = node.op

        # Get input identifiers (use node names, not values)
        input_ids = tuple(
            arg.name if isinstance(arg, fx.Node) else str(arg)
            for arg in node.args
        )

        # Get keyword arguments
        kwargs_key = tuple(sorted(node.kwargs.items()))

        return (op_id, input_ids, kwargs_key)

    def _is_pure_operation(self, node: fx.Node) -> bool:
        '''
        Check if an operation is pure (safe to deduplicate).

        Pure operations:
        - Deterministic (same inputs → same outputs)
        - No side effects
        - No randomness
        - No mutable state
        '''
        # Check if it's a known impure operation
        if node.op == 'call_function':
            if node.target in self.IMPURE_OPERATIONS:
                return False
            # Check if it's in our whitelist of pure operations
            if node.target in self.PURE_OPERATIONS:
                return True

        elif node.op == 'call_module':
            # Check module name for impure patterns
            target_lower = node.target.lower()
            for impure_name in self.IMPURE_OPERATIONS:
                if isinstance(impure_name, str) and impure_name in target_lower:
                    return False

            # Conservative: assume modules are pure unless known impure
            return True

        # Conservative: if unsure, don't optimize
        return False

    def _is_inplace_operation(self, node: fx.Node) -> bool:
        '''
        Check if operation is in-place (modifies input).

        In-place operations have names ending with '_' in PyTorch.
        Example: add_(), mul_(), etc.
        '''
        if node.op == 'call_method':
            return node.target.endswith('_')
        return False

    def _check_reachability(self, graph: fx.Graph) -> bool:
        '''
        Check that all nodes are reachable from inputs.

        Returns:
            True if graph is fully connected, False if orphaned nodes exist
        '''
        # Find all placeholder nodes (inputs)
        placeholders = [n for n in graph.nodes if n.op == 'placeholder']

        # BFS from placeholders
        visited = set()
        queue = list(placeholders)

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            # Add users (nodes that use this node's output)
            for user in node.users:
                queue.append(user)

        # Check if all non-placeholder nodes were visited
        all_nodes = set(n for n in graph.nodes if n.op != 'placeholder')
        unreachable = all_nodes - visited

        if unreachable:
            logger.warning(f"Found {len(unreachable)} unreachable nodes")
            return False

        return True
