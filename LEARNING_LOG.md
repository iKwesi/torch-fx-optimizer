# Learning Log

## Format
For each concept/bug/breakthrough:
- **Date/Time**: When it happened
- **Concept/Issue**: What I was working on
- **What I Learned**: Key insight
- **Confusion Points**: What I still don't fully understand
- **How I Resolved**: Steps taken or resources used

---

## Day 0 - Crash Course Complete

**Date**: January 3, 2026
**Time Spent**: ~45 minutes

### What Clicked ✅

1. **Computation graphs are just a way to track operations**
   - I always thought of PyTorch as "just running Python code" but now I see it's secretly building a graph of operations behind the scenes
   - Each operation (like matmul, relu, conv2d) becomes a node
   - The connections between nodes show how data flows through the model
   - This was the biggest "aha!" moment - graphs aren't magic, they're just tracking what happens to tensors

2. **torch.fx makes the invisible graph visible**
   - Normal PyTorch: graph exists but you can't easily see or modify it
   - torch.fx: traces the model and gives you an actual Graph object you can inspect and change
   - The symbolic tracing part was confusing at first but the notebook examples made it clear
   - Being able to print the graph and see all the nodes/operations listed out was really helpful

3. **Graph nodes have a clear structure: operation + inputs + outputs**
   - Every node has: `target` (what operation like 'conv2d'), `args` (what inputs), `users` (what uses this node's output)
   - When I printed a node, I could see exactly what it was doing
   - This makes sense now - to modify a graph, you just need to change these relationships

### What's Still Fuzzy ⚠️

1. **When torch.fx CAN'T trace a model**
   - The notebook mentioned "dynamic control flow" and "data-dependent shapes" break tracing
   - I don't fully understand what this means in practice
   - Like, what specific code would fail? Need concrete examples of "un-traceable" models
   - **Question**: How do I know if my model is traceable before trying?

2. **The difference between modifying a graph vs creating a new one**
   - When I replace a node, am I changing the original graph or making a copy?
   - The notebook showed `graph.erase_node()` and replacing nodes, but I'm not sure about side effects
   - **Question**: If I mess up a graph transformation, can I undo it? Or do I need to re-trace?

3. **How graph modifications affect gradients**
   - We only tested forward passes in the exercises
   - Does changing the graph break backpropagation?
   - **Question**: If I remove a node, do I need to manually fix the gradient computation?
   - This feels important for the recomputation optimization later

4. **What makes an operation "pure" vs having "side effects"**
   - The Conv2d example mentioned this briefly
   - I get that `x = x + 1` (in-place) is different from `y = x + 1` (new tensor)
   - But I don't know how to identify which PyTorch operations are pure vs not
   - **Question**: Is there a list of "safe to optimize" operations?

### Self-Assessment

- **Computational graphs**: 1/10 → 6/10
  - Before: No idea what a computation graph was
  - After: Can trace a model, print the graph, understand the basic structure
  - Still can't: Design graph algorithms or handle complex transformations

- **torch.fx API**: 0/10 → 5/10
  - Before: Never heard of it
  - After: Can use `symbolic_trace()`, iterate nodes, print graphs, do basic replacements
  - Still can't: Handle edge cases, debug tracing failures, understand all the internals

- **Graph manipulation**: 0/10 → 4/10
  - Before: No clue how to modify graphs
  - After: Can replace nodes (like ReLU → GELU exercise), understand the pattern
  - Still can't: Do complex transformations, handle dependencies correctly, avoid breaking things

### Key Takeaways

1. **Most Important**: Graphs are not abstract theory - they're a concrete data structure I can inspect and modify programmatically. This makes optimization passes possible.

2. **Second Most Important**: torch.fx gives me the tools to work with graphs, but it's not magic - there are limitations and edge cases I'll hit.

3. **Surprising Discovery**: I thought I'd need to understand backpropagation deeply to do this project, but the crash course showed me the forward pass graph is separate and I can work with it independently (for now).

### Concepts I Want to Explore Deeper (Later)

- How PyTorch's autograd system interacts with graph modifications
- What the "IR" (intermediate representation) actually means
- How production compilers like TorchScript and XLA differ from torch.fx
- Why some operations can't be traced (need concrete examples)

### Mistakes I Made

1. **First attempt at Conv2d exercise**: Forgot to specify `in_channels` and got a shape mismatch error
   - Learned: Input tensor channels must match Conv2d's `in_channels` parameter
   - Fixed by: Looking at the error message, realizing `(batch, 3, H, W)` needs `Conv2d(3, ...)`

2. **Graph node replacement**: First tried to just delete a node without updating users
   - Learned: You can't just remove nodes - you have to redirect their users to something else
   - Fixed by: Following the pattern in the exercise solution

### Questions for Architecture Phase

- How do I systematically find redundant operations in a graph? (Is it just comparing node hashes?)
- How do I know which activations to checkpoint without breaking gradients?
- How do I test that my graph modifications are actually correct? (Beyond just "outputs match")

### Next Phase
Ready for architecture design. I understand graphs well enough to start thinking about optimization algorithms.

### Confidence Level
**6/10** - I can read graphs and do basic modifications, but I'll definitely hit walls when implementing the actual optimizer. That's expected and okay.

---

## Milestone 1 - Core Infrastructure Implementation

**Date**: January 5, 2026
**Time Spent**: ~2 hours
**Status**: ✅ COMPLETE - All 13 tests passing

### What I Built

Implemented the complete core infrastructure for the graph optimizer:

1. **GraphOptimizer class** (src/graph_optimizer.py - 453 lines)
   - Model tracing with torch.fx.symbolic_trace()
   - Pass orchestration system
   - Verification system (compares original vs optimized outputs)
   - Comprehensive benchmarking (memory, time, graph size)

2. **OptimizationPass ABC** (src/optimization_pass.py - 227 lines)
   - Abstract base class defining the pass interface
   - Pass registry system with @register_pass decorator
   - Pass lookup and management functions

3. **Test infrastructure** (tests/conftest.py + tests/test_basic.py)
   - 8 pytest fixtures (MLP, CNN, deep models, various inputs)
   - 13 comprehensive tests covering all core functionality
   - 100% pass rate

### Key Lessons Learned ✅

#### 1. **The Importance of Validation and Error Handling**
   - **What I Learned**: Every public method needs comprehensive input validation
   - **Example**: Device validation catches typos early, graceful fallback when MPS/CUDA unavailable
   - **Pattern**: Validate inputs → Try operation → Catch exception → Provide helpful error message
   - **Code**:
     ```python
     if device not in valid_devices:
         raise ValueError(f"Device must be one of {valid_devices}, got '{device}'")
     ```
   - **Why it matters**: Users will pass invalid inputs - failing fast with clear messages saves debugging time

#### 2. **Python's ABC Pattern for Extensibility**
   - **What I Learned**: Abstract base classes enforce a contract for all subclasses
   - **Before**: Didn't understand why you'd use ABC instead of just documentation
   - **After**: ABC + @abstractmethod guarantees all passes implement required methods
   - **Pattern**: Define interface → Force implementation → Enable polymorphism
   - **Why it matters**: When I implement optimization passes, I can't forget required methods

#### 3. **The Pass Registry Pattern**
   - **What I Learned**: A registry decouples pass names from implementations
   - **Pattern**:
     ```python
     @register_pass
     class MyPass(OptimizationPass):
         @property
         def name(self): return 'my_pass'

     # Later: get_pass_by_name('my_pass')
     ```
   - **Why it matters**: Users can reference passes by string names, passes can be loaded dynamically
   - **Discovery**: This is how PyTorch's optimizer registry works internally

#### 4. **Device-Aware Benchmarking is Tricky**
   - **What I Learned**: Each device (CPU/MPS/CUDA) has different memory tracking APIs
   - **MPS**: `torch.mps.driver_allocated_memory()`, `torch.mps.synchronize()`
   - **CUDA**: `torch.cuda.max_memory_allocated()`, `torch.cuda.synchronize()`
   - **CPU**: No reliable per-operation memory tracking
   - **Why it matters**: Need to synchronize device before timing to get accurate measurements
   - **Gotcha**: Without synchronize(), you measure kernel launch time, not execution time

#### 5. **torch.fx GraphModule Lifecycle**
   - **What I Learned**: After modifying graph.graph, must call graph.recompile()
   - **Pattern**: Get GraphModule → Modify graph → Recompile → Verify
   - **Why**: The graph and the Python code are separate - recompile regenerates the forward() method
   - **Code**: `graph_module.recompile()` after any graph transformation

#### 6. **graph.lint() is Your Friend**
   - **What I Learned**: graph.lint() checks graph validity (no dangling references, well-formed)
   - **When to use**: After every graph transformation in verify() method
   - **What it catches**: Deleted nodes still referenced, malformed node structure
   - **Pattern**: Transform → lint() → If passes, return True

#### 7. **Type Hints + Docstrings = Self-Documenting Code**
   - **What I Learned**: Comprehensive type hints catch bugs before runtime
   - **Pattern**: All parameters and return values typed, docstrings with examples
   - **Tools**: Python's type checker would catch many issues
   - **Why it matters**: When implementing passes, I'll know exactly what each method expects/returns

#### 8. **Pytest Fixtures for DRY Tests**
   - **What I Learned**: Fixtures eliminate duplicate model/input creation across tests
   - **Pattern**: Define fixtures in conftest.py, use as test parameters
   - **Example**: `def test_something(simple_mlp, test_input_mlp, device)`
   - **Why it matters**: Can test multiple model architectures without code duplication

#### 9. **Median vs Mean for Benchmarking**
   - **What I Learned**: Use median time, not mean - more robust to outliers
   - **Why**: First run might include JIT compilation, cache warming
   - **Pattern**: Warmup runs → Multiple measured runs → Report median
   - **Code**: `torch.median(torch.tensor(timings))`

#### 10. **Testing Error Paths is Critical**
   - **What I Learned**: Half my tests check error handling, not success paths
   - **Tests written**:
     - Invalid device raises ValueError
     - Untraceable model provides helpful error
     - Missing optimized model raises RuntimeError
     - Invalid pass name raises ValueError
   - **Why it matters**: Users will trigger error paths - they need to be tested

### Technical Decisions & Trade-offs

#### Decision 1: In-place graph modification vs copying
- **Choice**: Modify graphs in-place
- **Why**: torch.fx graphs are mutable by design, copying is expensive
- **Trade-off**: Can't easily undo transformations, but that's acceptable since we verify after each pass

#### Decision 2: Store both original and traced models
- **Choice**: Keep references to original_model and traced model
- **Why**: Need original for verification (compare outputs)
- **Trade-off**: 2x memory overhead, but necessary for correctness

#### Decision 3: Verify after EACH pass, not just at the end
- **Choice**: Call verify_correctness() after every optimization pass
- **Why**: Easier to debug which pass broke the model
- **Trade-off**: Slower, but correctness > speed during development

#### Decision 4: Make benchmarking optional, verification required
- **Choice**: optimize() calls verify(), user calls benchmark() separately
- **Why**: Verification is safety-critical, benchmarking is optional analysis
- **Pattern**: Fail fast on verification, allow user to skip benchmarking

### What Surprised Me

1. **How much code is error handling**: ~40% of GraphOptimizer is validation and error messages
   - Before: "Error handling is just try/except"
   - After: Good errors require checking all inputs, providing context, suggesting fixes

2. **torch.fx.symbolic_trace can fail silently**: Dynamic control flow causes tracing failures
   - Solution: Wrap in try/except and provide hints about torch.fx.wrap()

3. **Device synchronization is essential**: Without it, timing measurements are meaningless
   - Learned: GPU operations are async - must synchronize before measuring

4. **Test count matters less than test coverage**: 13 tests covering all code paths > 50 superficial tests

### Mistakes I Made & Fixes

#### Mistake 1: Forgot to add src to Python path
- **Problem**: `ModuleNotFoundError: No module named 'src'`
- **Root cause**: pytest couldn't find src package
- **Fix**: Created root conftest.py to add project root to sys.path
- **Lesson**: Always set up import paths correctly before writing tests

#### Mistake 2: Initially didn't call graph.recompile()
- **Problem**: Graph modifications didn't take effect
- **Root cause**: Forgot GraphModule caches the forward() method
- **Fix**: Added graph_module.recompile() after transformations
- **Lesson**: Read torch.fx docs carefully - recompilation is required

#### Mistake 3: Used mean instead of median for timing
- **Problem**: First draft used mean, which is skewed by outliers
- **Fix**: Switched to torch.median()
- **Lesson**: For benchmarking, median is more robust than mean

### Concepts That Clicked

1. **The optimization pass lifecycle**:
   - analyze() → Find opportunities (read-only)
   - transform() → Modify graph (write)
   - verify() → Check correctness (read-only validation)
   - This separation of concerns makes passes easier to reason about

2. **Why verification must compare original model, not traced**:
   - Traced model might have tracing artifacts
   - Original model is ground truth
   - If optimized ≠ original, something is wrong

3. **The registry pattern is elegant**:
   - Decouples pass names from implementations
   - Allows dynamic loading
   - Makes testing easier (can register/unregister mock passes)

### Still Fuzzy / Questions

1. **When does graph.lint() fail?**
   - I know it checks validity, but haven't seen it actually catch an error yet
   - Need to intentionally create an invalid graph to understand what it checks
   - Question: What specific invariants does lint() verify?

2. **Can I optimize the verification process?**
   - Currently running full forward pass through both models
   - Question: Could I verify on a smaller subset of inputs? Or is full verification required?

3. **Memory tracking accuracy on MPS**
   - torch.mps.driver_allocated_memory() seems coarse-grained
   - Question: Is there a more precise way to measure activation memory?

4. **How do I test passes without implementing passes yet?**
   - Current tests use passes=[] (no-op)
   - Next: Need to create a simple mock pass for testing the orchestration

### Updated Self-Assessment

- **Python software engineering**: 6/10 → 8/10
  - Before: Basic Python knowledge
  - After: Understand ABC, decorators, registries, type hints, comprehensive error handling
  - Still learning: Advanced patterns, performance optimization

- **Testing with pytest**: 5/10 → 8/10
  - Before: Could write basic tests
  - After: Understand fixtures, parametrization, error testing, test organization
  - Still learning: Mocking, property-based testing

- **torch.fx internals**: 5/10 → 7/10
  - Before: Could trace and read graphs
  - After: Understand GraphModule lifecycle, recompilation, graph.lint()
  - Still learning: Advanced transformations, handling edge cases

- **System design**: 4/10 → 7/10
  - Before: Could sketch basic architectures
  - After: Implemented extensible system with clear abstractions and error handling
  - Still learning: Scalability, performance optimization patterns

### Key Takeaways

1. **Most Important**: Infrastructure quality determines how easy it is to add features later. Comprehensive error handling and validation pay off immediately.

2. **Second Most Important**: Test-driven development works - writing tests first clarified what the API should be.

3. **Third Most Important**: Type hints + docstrings = self-documenting code. When I come back tomorrow, I won't need to re-learn what each method does.

### What's Next (Milestone 2)

With the infrastructure complete, I can now implement actual optimization passes:

1. **Redundant operation elimination** - Find duplicate computations and reuse results
2. **Recomputation/checkpointing** - Trade memory for computation

The infrastructure handles:
- ✅ Pass orchestration
- ✅ Verification after each pass
- ✅ Benchmarking
- ✅ Error reporting
- ✅ Device management

I just need to implement the analyze/transform/verify logic for each pass.

### Confidence Level

**7/10 → 8/10** - I can now build the optimizer on a solid foundation. The infrastructure is rock-solid, and I understand the patterns I'll use for implementing passes.

---

## Milestone 2 Complete - Redundancy Elimination Pass

**Date**: January 5-6, 2026
**Time Spent**: ~4 hours
**Tests Passing**: 10/10 ✅

### What I Built

**Files Created:**
- `src/torch_fx_optimizer/passes/redundant_ops.py` (279 lines)
- `tests/test_redundant_ops.py` (215 lines)

**Key Achievement:**
First working graph optimization that actually transforms graphs and eliminates redundant operations. This pass detects duplicate operations (same operation with same inputs) and removes them by reusing the result of the first occurrence.

**Core Functionality:**
- Detects redundant module calls (e.g., `conv(x)` called twice with same `x`)
- Detects redundant function calls (e.g., `torch.relu(x)` called twice)
- Safely distinguishes pure operations (safe to deduplicate) from impure operations (random/stateful)
- Handles multiple redundant copies (e.g., 3 identical operations → keep 1, remove 2)
- Verifies graph validity and reachability after transformation

---

### What I Learned

#### 1. Graph Transformation Patterns

**The analyze → transform → verify pattern:**
```python
def analyze(self, graph):
    # Read-only: find opportunities
    # Returns: {'opportunities': [...], 'stats': {...}, 'safe': bool}

def transform(self, graph):
    # Modify graph in-place
    # Remove redundant nodes, rewire connections

def verify(self, graph):
    # Check graph is still valid
    # Run graph.lint(), check reachability
```

**Key insight**: Separation of concerns makes debugging easier. If analyze() finds nothing, transform() does nothing. If transform() modifies the graph, verify() catches errors.

**Why this pattern works:**
- analyze() can be tested independently (just detection logic)
- transform() can be tested with known redundancies
- verify() catches bugs before they cause downstream failures

#### 2. Structural Hashing for Deduplication

**The challenge**: How do you know if two nodes represent the "same" operation?

**Wrong approach** (doesn't work):
```python
# ❌ DON'T USE id() - object identity changes!
if id(node_a) == id(node_b):  # Always False for different nodes
    ...
```

**Correct approach** (structural equality):
```python
def _compute_operation_key(self, node):
    # Hash by: (operation_type, input_names, kwargs)
    op_id = node.target.__name__  # e.g., "relu", "conv2d"
    input_ids = tuple(arg.name for arg in node.args)  # e.g., ("x", "weight")
    kwargs_key = tuple(sorted(node.kwargs.items()))  # e.g., (("stride", 1),)
    return (op_id, input_ids, kwargs_key)
```

**What I learned**:
- Use **structural equality** (compare structure) not **object identity** (compare memory addresses)
- Node names are stable across transformations
- Tuples are hashable and support exact equality comparison
- Two nodes with the same operation key produce identical outputs

**Example:**
```python
# Graph:
# a = relu(x)  -> key = ('relu', ('x',), ())
# b = relu(x)  -> key = ('relu', ('x',), ())  # SAME KEY! Redundant.
# c = relu(a)  -> key = ('relu', ('a',), ())  # Different input, NOT redundant

seen_ops = {}
for node in graph.nodes:
    key = _compute_operation_key(node)
    if key in seen_ops:
        # Found duplicate!
        duplicate_node = node
        original_node = seen_ops[key]
    else:
        seen_ops[key] = node
```

#### 3. Pure vs Impure Operations

**Pure operations** (safe to deduplicate):
- **Deterministic**: Same inputs → same outputs, every time
- **No side effects**: Don't modify global state
- **Examples**:
  - `torch.relu(x)` - always returns same result for same `x`
  - `torch.conv2d(x, weight)` - deterministic convolution
  - `torch.add(a, b)` - simple arithmetic

**Impure operations** (NOT safe to deduplicate):
- **Random**: `torch.nn.functional.dropout(x)` - uses randomness, different outputs each call
- **Stateful**: `BatchNorm(x)` - has running mean/var that updates during training
- **Side effects**: Operations that modify global state

**Why this matters critically:**
```python
# ❌ WRONG: If you deduplicate dropout, behavior changes!
a = dropout(x)  # Random mask 1
b = dropout(x)  # Random mask 2
z = a + b       # Sum of two different random masks

# After wrong deduplication:
a = dropout(x)  # Random mask 1
b = a           # REUSES same mask - NOT what we want!
z = a + b       # Sum of same mask twice - WRONG!
```

**Implementation:**
```python
PURE_OPERATIONS = {
    torch.relu,
    torch.add,
    torch.matmul,
    torch.nn.functional.conv2d,
    # ... more deterministic ops
}

IMPURE_OPERATIONS = {
    torch.nn.functional.dropout,
    'batchnorm',  # String for module names
    # ... more stateful/random ops
}

def _is_pure_operation(self, node):
    if node.target in IMPURE_OPERATIONS:
        return False
    if node.target in PURE_OPERATIONS:
        return True
    # Conservative: if unsure, don't optimize
    return False
```

**Key lesson**: **Be conservative**. Better to miss optimization opportunities than to break correctness. Whitelist known-safe operations.

#### 4. Graph Node Manipulation APIs

**The key methods I learned:**

```python
# 1. Redirect all users to a different node
duplicate_node.replace_all_uses_with(original_node)
# Before: other_nodes use duplicate_node
# After:  other_nodes use original_node instead

# 2. Remove a node from the graph
graph.erase_node(duplicate_node)
# Node must have no users when erased!

# 3. Verify graph is well-formed
graph.lint()
# Checks: no dangling references, valid structure
# Raises exception if graph is malformed

# 4. Regenerate forward() code
graph_module.recompile()
# Must call after modifying graph
# Regenerates the Python forward() method from graph
```

**The pattern for removing redundant nodes:**
```python
# Step 1: Redirect users
duplicate.replace_all_uses_with(original)

# Step 2: Erase node (safe now, no users)
graph.erase_node(duplicate)

# Step 3: Verify
graph.lint()  # Raises exception if broken

# Step 4: Recompile (done by GraphOptimizer)
graph_module.recompile()
```

**What happens under the hood:**
```python
# Before:
# Node: x (op='placeholder')
# Node: a (op='call_function', target=relu, args=(x,), users=[c])
# Node: b (op='call_function', target=relu, args=(x,), users=[c])
# Node: c (op='call_function', target=add, args=(a, b))

# After replace_all_uses_with(a) and erase(b):
# Node: x (op='placeholder')
# Node: a (op='call_function', target=relu, args=(x,), users=[c, c])
# Node: c (op='call_function', target=add, args=(a, a))
# (b is gone)
```

#### 5. The Reachability Check

**What it checks**: After removing nodes, are all remaining nodes reachable from inputs?

**Why it's needed**: Erasing nodes can orphan other nodes that depended on them.

**How it works (BFS from inputs):**
```python
def _check_reachability(self, graph):
    # Start from inputs (placeholder nodes)
    placeholders = [n for n in graph.nodes if n.op == 'placeholder']

    # BFS: visit all reachable nodes
    visited = set()
    queue = list(placeholders)

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        # Add all users (downstream nodes)
        for user in node.users:
            queue.append(user)

    # Check if any nodes are unreachable
    all_nodes = set(graph.nodes)
    unreachable = all_nodes - visited

    return len(unreachable) == 0
```

**Example of orphaned node:**
```python
# Original graph:
# x -> a -> b -> c
#        -> d

# If we accidentally erase 'a':
# x    b -> c  (b is orphaned! No path from x to b)
#      d       (d is orphaned!)

# Reachability check catches this
```

**Why this matters**: graph.lint() doesn't catch orphaned nodes if they're internally consistent. Reachability check is an additional safety measure.

#### 6. Understanding `node.users`

**What confused me initially**: What does "users" mean?

**Analogy**: Think of data flow like water flowing through pipes
- A node produces a value (water source)
- Other nodes consume that value (water consumers)
- `node.users` = list of consumer nodes

**Concrete example:**
```python
# Graph code:
x = input
a = relu(x)
b = relu(x)
c = add(a, b)

# Node relationships:
# x.users = [a, b]      # a and b both use x
# a.users = [c]         # c uses a
# b.users = [c]         # c uses b
# c.users = []          # nothing uses c (output)

# When we do: b.replace_all_uses_with(a)
# Result:
# x.users = [a, b]      # Still connected (b not erased yet)
# a.users = [c, c]      # c now uses a twice!
# b.users = []          # b no longer used
# c.users = []

# After: graph.erase_node(b)
# x.users = [a]         # b removed from graph
# a.users = [c, c]
# c.users = []
```

#### 7. Testing Strategy for Graph Transformations

**The tests I wrote:**

1. **Detection test**: Does analyze() find redundancies?
   ```python
   def test_redundancy_detection():
       # Model with duplicate conv calls
       analysis = pass_instance.analyze(graph)
       assert len(analysis['opportunities']) >= 1
   ```

2. **Elimination test**: Does transform() actually remove nodes?
   ```python
   def test_redundancy_elimination():
       original_nodes = len(list(graph.nodes))
       optimized = optimizer.optimize(['redundant_ops'])
       optimized_nodes = len(list(optimized.graph.nodes))
       assert optimized_nodes < original_nodes
   ```

3. **Correctness test**: Does output match original?
   ```python
   def test_correctness():
       original_output = model(input)
       optimized_output = optimized_model(input)
       torch.testing.assert_close(original_output, optimized_output)
   ```

4. **False positive test**: Are non-redundant ops preserved?
   ```python
   def test_no_false_positives():
       # relu(x) and relu(relu(x)) are NOT redundant
       # Should preserve both
   ```

5. **Multiple duplicates test**: Handle 3+ identical operations?
   ```python
   def test_multiple_duplicates():
       # a = relu(x), b = relu(x), c = relu(x)
       # Should keep only 1 relu
   ```

6. **Parametrized tests**: Works with different inputs?
   ```python
   @pytest.mark.parametrize("batch_size", [1, 4, 8])
   def test_different_batch_sizes(batch_size):
       # Test optimization works for various batch sizes
   ```

**What I learned**: **Test transformations at multiple levels**
- Detection logic (analyze)
- Transformation logic (transform)
- Correctness (output matches)
- Edge cases (false positives, multiple duplicates)
- Different inputs (parametrized tests)

---

### Comparison to Milestone 1

**What was easier:**
- Had infrastructure in place (GraphOptimizer, OptimizationPass, test fixtures)
- Understood the three-step pattern (analyze/transform/verify)
- Tests followed similar structure to M1
- Pass registration was automatic (@register_pass decorator)

**What was harder:**
- **Actual graph manipulation** - Reading graphs is easy, modifying them safely is hard
- **Understanding operational semantics** - When is an operation safe to optimize?
- **Debugging graph transformation errors** - torch.fx error messages can be cryptic
- **Thinking about structural equality** - Had to unlearn object identity intuitions

**Time comparison:**
- M1: ~2 hours (infrastructure, mostly design and testing setup)
- M2: ~4 hours (first real optimization, lots of learning)
- M2 took longer because:
  - First time manipulating graphs (learning curve)
  - Had to research pure vs impure operations
  - Debugging graph transformation errors took time
  - Writing comprehensive tests for correctness

---

### Challenges Faced

#### Challenge 1: Understanding Structural Hashing

**What confused me:**
- Initially tried to use `id(node)` to detect duplicates
- Didn't understand why this failed - they're literally the same operation!
- Error: Every node has a different `id()`, even if they do the same thing

**How I figured it out:**
```python
# Debugging:
a = relu(x)
b = relu(x)
print(id(a))  # 140234123456
print(id(b))  # 140234999888  # Different IDs!
print(a is b)  # False

# Solution: Compare structure, not identity
print(a.target)  # <built-in function relu>
print(b.target)  # <built-in function relu>  # Same!
print(a.args)    # (x,)
print(b.args)    # (x,)  # Same!
# These ARE the same operation!
```

**What I learned:**
- **Object identity** (id, is) checks memory address
- **Structural equality** compares content
- For graph optimization, we care about structure, not identity
- Hash by (operation_type, inputs, kwargs) not by node object

**Mental model shift:**
- Before: "Two nodes are the same if they're the same object"
- After: "Two nodes are the same if they compute the same thing"

#### Challenge 2: Understanding `replace_all_uses_with()`

**What confused me**:
- What does "uses" mean?
- Where do they come from?
- What happens when you replace?

**How I understood it (concrete example):**
```python
# Before:
# node_a = relu(x)     # node_a.users = [node_c]
# node_b = relu(x)     # node_b.users = [node_c]
# node_c = add(node_a, node_b)

# Visual:
#   x
#  / \
# a   b
#  \ /
#   c

# Call: node_b.replace_all_uses_with(node_a)

# After:
# node_a = relu(x)     # node_a.users = [node_c, node_c]
# node_b = relu(x)     # node_b.users = []  (no longer used!)
# node_c = add(node_a, node_a)  # Both args now point to node_a

# Visual:
#   x
#  / \
# a   b (orphaned)
# ||
# c
```

**What I learned:**
- Nodes track their "users" (downstream nodes that use their output)
- `replace_all_uses_with(other)` redirects all users to `other`
- After replacement, original node has no users (can be safely erased)
- Downstream nodes automatically updated (torch.fx handles rewiring)

**Why this is powerful:**
- Don't need to manually find and update all users
- torch.fx maintains the user graph automatically
- Just replace and erase - graph stays consistent

#### Challenge 3: Deciding What's Safe to Optimize

**What confused me:**
- How do I know if an operation is "pure"?
- Is there a list somewhere?
- What if I'm wrong and break something?

**How I solved it:**
1. **Started with obvious pure operations**
   - Math operations: add, mul, matmul (clearly deterministic)
   - Activations: relu, gelu, tanh (no randomness, no state)

2. **Researched PyTorch docs** for operations to avoid
   - Dropout: explicitly documented as random
   - BatchNorm: has running_mean, running_var (stateful)

3. **Adopted conservative whitelist approach**
   ```python
   # Whitelist: only optimize these known-safe ops
   PURE_OPERATIONS = {torch.relu, torch.add, ...}

   # Blacklist: never optimize these
   IMPURE_OPERATIONS = {dropout, batchnorm, ...}

   # Default: if unsure, DON'T optimize
   def _is_pure_operation(self, node):
       if node.target in IMPURE_OPERATIONS:
           return False
       if node.target in PURE_OPERATIONS:
           return True
       return False  # Conservative: skip unknown ops
   ```

**What I learned:**
- **Correctness > Performance**: Better to miss optimizations than break models
- **Whitelist > Blacklist**: Easier to maintain list of safe operations
- **Document assumptions**: Comments explain why each operation is pure/impure
- **Tests catch mistakes**: If I mislabel an operation, tests fail

**Future improvement**: Could add flag to enable "aggressive mode" that optimizes more operations, but default should be conservative.

#### Challenge 4: Writing Tests for Correctness

**What confused me:**
- How do I test that the optimization is correct?
- Just checking output equality feels insufficient
- What edge cases am I missing?

**How I solved it:**

1. **Multiple levels of testing:**
   ```python
   # Level 1: Detection works
   assert len(opportunities) > 0

   # Level 2: Graph actually changes
   assert optimized_nodes < original_nodes

   # Level 3: Output is correct
   torch.testing.assert_close(original_output, optimized_output)

   # Level 4: No false positives
   # (non-redundant ops preserved)
   ```

2. **Constructed adversarial test cases:**
   ```python
   # Test case: Looks redundant, but isn't
   def forward(self, x):
       a = relu(x)
       b = relu(a)  # Different input!
       return a + b

   # Should NOT optimize - must preserve both relus
   ```

3. **Parametrized testing:**
   ```python
   @pytest.mark.parametrize("batch_size", [1, 4, 8])
   def test_different_batch_sizes(batch_size):
       # Ensure optimization works for all input shapes
   ```

**What I learned:**
- **Tests should cover both positive and negative cases**
  - Positive: Redundancy is eliminated
  - Negative: Non-redundancy is preserved
- **Correctness tests are non-negotiable**
  - Output must match original exactly
  - Use `torch.testing.assert_close()` for numerical comparison
- **Edge cases are where bugs hide**
  - Multiple duplicates (3+ identical ops)
  - Different batch sizes
  - Mixed redundant/non-redundant operations

---

### Key Takeaways

1. **Most Important**: **Graph transformations are doable but require careful verification.** The analyze/transform/verify pattern prevents bugs and makes debugging tractable.

2. **Second Most Important**: **Structural equality, not object identity.** This was the biggest mental shift. Two nodes can be functionally identical even if they're different objects in memory.

3. **Third Most Important**: **Conservative optimization is safer than aggressive.** Whitelisting known-safe operations prevents subtle bugs. Missing optimizations is acceptable; breaking correctness is not.

4. **Fourth Important**: **Test at multiple levels.** Detection, transformation, correctness, and edge cases all need separate tests. Integration tests catch what unit tests miss.

---

### Concepts That Clicked

#### 1. Why we use structural hashing

**Before**: "I'll just compare nodes directly"

**After**: "Nodes are objects in memory. I need to compare what they compute, not where they live in memory."

**The "aha!" moment**:
```python
# These are DIFFERENT objects
a = torch.relu(x)
b = torch.relu(x)
assert a is not b  # True! Different objects

# But they compute the SAME thing
assert a.equals(b)  # True! Same values

# For optimization, we care about computation, not identity
# Solution: Hash by (operation, inputs, kwargs)
```

#### 2. Why recompile() is necessary

**Before**: "The graph is the model, right?"

**After**: "The graph is a data structure. GraphModule generates Python code from it."

**The "aha!" moment**:
```python
# GraphModule stores TWO representations:
# 1. graph (fx.Graph object) - the data structure
# 2. forward() method - generated Python code

# When you modify the graph:
graph.erase_node(duplicate)  # Modifies graph (data structure)

# The forward() code is now stale!
# It still references the old node

# Solution: Regenerate forward() from graph
graph_module.recompile()  # Regenerates forward() code
```

**Mental model**: GraphModule = Graph + Generated Code. Modifying graph requires regenerating code.

#### 3. Why we check reachability

**Before**: "graph.lint() checks everything, right?"

**After**: "lint() checks structural validity, but not connectivity."

**The "aha!" moment**:
```python
# Scenario: Accidentally erase a node that other nodes depend on
# a -> b -> c
graph.erase_node(b)
# a    c  (c is orphaned!)

# graph.lint() might pass if c is internally valid
# But c is unreachable from inputs!

# Solution: BFS from inputs to check all nodes reachable
```

**Why it matters**: Orphaned nodes don't cause immediate errors but can break backpropagation or cause silent failures.

#### 4. The power of the registry pattern

**Before**: "Why not just import the pass class directly?"

**After**: "Registry allows dynamic loading and string-based references."

**The "aha!" moment**:
```python
# Without registry:
from passes.redundant_ops import RedundantOpElimination
pass_instance = RedundantOpElimination()

# With registry:
pass_instance = get_pass_by_name('redundant_ops')

# User can specify passes as strings:
optimizer.optimize(passes=['redundant_ops', 'recomputation'])

# Passes can be added without modifying core code
# Just define pass + @register_pass decorator
```

**Why it's elegant**: Decoupling, extensibility, user-friendly API.

---

### Still Fuzzy / Questions

#### 1. Performance impact of this optimization

**Questions:**
- How much time does redundancy elimination actually save?
- What's the overhead of the analysis pass?
- Is the graph transformation overhead worth it?

**Next steps:**
- Need to benchmark on real models (ResNet, BERT, etc.)
- Measure analysis time vs transformation time vs speedup
- Compare memory usage before/after

**Hypothesis**: Most savings on models with repeated blocks (ResNet residual blocks, Transformer layers)

#### 2. Edge cases not handled

**Questions:**
- What about models with control flow? (if/for statements)
- What about in-place operations? (x.add_(1))
- What about custom autograd functions?

**Concern**: My current implementation assumes:
- Models are traceable (no dynamic control flow)
- Operations are out-of-place
- All operations go through standard PyTorch APIs

**Next steps:**
- Document these limitations in docstring
- Add error handling for in-place operations
- Consider Milestone 3 for edge case handling

#### 3. Interaction with PyTorch autograd

**Questions:**
- Does removing nodes break gradient computation?
- Are gradients still correct after deduplication?
- Do I need to test backward pass, not just forward?

**Current assumption**:
- torch.fx preserves autograd information
- If forward outputs match, gradients should match
- But haven't explicitly tested this!

**Next steps:**
- Write tests that compute gradients and compare
- Verify backward pass correctness
- Test with `.backward()` and check parameter gradients

#### 4. When should users apply this pass?

**Questions:**
- Is redundancy elimination always beneficial?
- Are there cases where it hurts performance?
- Which models benefit most?

**Hypothesis:**
- Models with repeated blocks: High benefit (ResNet, Transformers)
- Models with unique operations: Low benefit (might even add overhead)
- Models with dropout/randomness: Must be careful!

**Next steps:**
- Document when to use this pass
- Provide benchmarks on different model types
- Add warnings for models with dropout

---

### Updated Self-Assessment

- **Graph manipulation**: 4/10 → 8/10
  - Before: Could read graphs, do basic replacements
  - After: Can transform graphs, eliminate nodes, verify correctness, understand node relationships
  - Still learning: Complex multi-node transformations, handling control flow, performance optimization

- **Optimization patterns**: 3/10 → 7/10
  - Before: Theoretical understanding of optimizations
  - After: Built a working optimization, understand pure vs impure, structural equality, correctness verification
  - Still learning: Advanced optimizations (fusion, recomputation), performance tuning, larger-scale transformations

- **torch.fx mastery**: 5/10 → 8/10
  - Before: Could trace and read, basic modifications
  - After: Can manipulate graphs, understand GraphModule lifecycle, verify transformations, handle node users/args
  - Still learning: Handling edge cases (control flow, in-place ops), advanced graph patterns, performance optimization

- **Debugging skills**: 5/10 → 8/10
  - Before: Could print graphs, read error messages
  - After: Can use reachability checks, understand structural issues, write targeted tests, debug transformation errors
  - Still learning: Profiling performance issues, debugging gradient problems

- **Confidence**: 6/10 → 8/10
  - Can now implement optimization passes from scratch
  - Understand the patterns and pitfalls
  - Ready for more complex transformations
  - Milestone 4 (recomputation) feels achievable with the foundation I've built

---

### Mistakes I Made & What I Learned

#### Mistake 1: First tried to use node identity for deduplication
- **What I did**: `if node_a is node_b: ...`
- **Why it failed**: Every node is a unique object, even if they compute the same thing
- **Fix**: Use structural hashing `(op_type, inputs, kwargs)`
- **Lesson**: Think about what nodes *compute*, not what they *are*

#### Mistake 2: Forgot to check if operations are pure
- **What I did**: Initially deduplicated ALL operations
- **Why it failed**: Tests broke for models with dropout (different random values each call)
- **Fix**: Added `PURE_OPERATIONS` and `IMPURE_OPERATIONS` whitelists
- **Lesson**: Operations have semantics - random/stateful operations can't be deduplicated

#### Mistake 3: Didn't verify reachability after transformation
- **What I did**: Only called `graph.lint()`
- **Why it was insufficient**: lint() doesn't catch orphaned nodes
- **Fix**: Added `_check_reachability()` using BFS from placeholders
- **Lesson**: Multiple verification layers catch different types of errors

#### Mistake 4: Initial tests only checked node count
- **What I did**: `assert optimized_nodes < original_nodes`
- **Why it was insufficient**: Doesn't verify correctness
- **Fix**: Added `torch.testing.assert_close()` to compare outputs
- **Lesson**: Performance metrics don't guarantee correctness - need explicit correctness tests

---

### Code Snippets I'm Proud Of

#### 1. The structural hashing function
```python
def _compute_operation_key(self, node: fx.Node) -> Tuple:
    '''Compute structural key for operation deduplication.'''
    # Operation identifier
    op_id = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)

    # Input identifiers (stable names, not values)
    input_ids = tuple(
        arg.name if isinstance(arg, fx.Node) else str(arg)
        for arg in node.args
    )

    # Keyword arguments
    kwargs_key = tuple(sorted(node.kwargs.items()))

    return (op_id, input_ids, kwargs_key)
```
**Why I'm proud**: Handles both functions and modules, uses stable names, properly handles kwargs.

#### 2. The reachability check
```python
def _check_reachability(self, graph: fx.Graph) -> bool:
    '''BFS from inputs to verify all nodes reachable.'''
    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    visited = set()
    queue = list(placeholders)

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for user in node.users:
            queue.append(user)

    all_nodes = set(n for n in graph.nodes if n.op != 'placeholder')
    unreachable = all_nodes - visited
    return len(unreachable) == 0
```
**Why I'm proud**: Clean BFS implementation, catches subtle bugs that lint() misses.

#### 3. The transformation loop
```python
for duplicate_node, original_node in opportunities:
    logger.debug(f"Replacing {duplicate_node.name} with {original_node.name}")
    duplicate_node.replace_all_uses_with(original_node)
    graph.erase_node(duplicate_node)

graph.lint()  # Verify graph validity
```
**Why I'm proud**: Simple, clear, safe. Replace → Erase → Verify.

---

### What's Next (Milestone 3 or 4?)

I now have a complete redundancy elimination pass that:
- ✅ Detects redundant operations
- ✅ Safely eliminates them
- ✅ Preserves correctness
- ✅ Handles edge cases (multiple duplicates, pure vs impure)
- ✅ Has comprehensive tests

**Two paths forward:**

### Option A: Milestone 3 - Edge Cases & Robustness
- Handle BatchNorm correctly (stateful but might be optimizable in eval mode)
- Handle in-place operations (detect and skip)
- Test with real models (ResNet, VGG, Transformers)
- More comprehensive edge case coverage
- Performance benchmarking

**Pros**: Makes current pass production-ready, learns about edge cases
**Cons**: Delays the hard problem (recomputation), might be tedious

### Option B: Milestone 4 - Recomputation (Hard Mode) ⚠️
- Now that I understand graph transformations deeply
- Have working verification system
- Understand pure vs impure operations
- Ready to attempt the hard problem: activation checkpointing

**Pros**: Tackles the core technical challenge, builds on momentum
**Cons**: Much harder, might hit walls, could take significantly longer

**Decision factors:**
- How confident am I with graph transformations? (8/10 - high)
- How important is activation checkpointing? (Core project goal)
- Do I have time for edge cases or want to push forward? (Want challenge)

---

### Confidence Level

**Overall: 8/10** - Ready for the next challenge

**Specific assessments:**
- Graph manipulation: 8/10 (comfortable with transformations)
- Verification: 9/10 (understand what to check and how)
- Testing: 8/10 (know how to write comprehensive tests)
- Debugging: 7/10 (can figure out issues, but torch.fx errors still cryptic sometimes)

**Readiness for Milestone 4 (Recomputation):**
- Understand graph transformations: ✅
- Can verify correctness: ✅
- Know how to test: ✅
- Understand pure/impure operations: ✅
- Ready to learn about gradient checkpointing: ✅

**I'm choosing to proceed to Milestone 4 (Recomputation).**

I feel confident enough with graph transformations, and recomputation is the core technical challenge of this project. Milestone 3 can come later if needed for production deployment, but right now I want to tackle the hard problem while I have momentum.

---

### Final Reflections

**What surprised me most:**
- How much careful thought goes into "safe" optimizations
- The importance of structural equality over object identity
- That tests for correctness are more important than tests for performance

**What I underestimated:**
- How tricky it is to decide what's safe to optimize
- The importance of reachability checks (not just lint)
- How much time debugging takes vs coding

**What I'm excited about:**
- I built something that actually works and makes graphs smaller!
- The verification system gives me confidence in correctness
- Ready to tackle the hard problem (recomputation) next

**Lessons for future me:**
- Be conservative with optimizations - correctness first
- Test at multiple levels (detection, transformation, correctness, edge cases)
- Structural equality is your friend for graph deduplication
- Always verify reachability after transformations
- Write tests before implementing complex logic

---
