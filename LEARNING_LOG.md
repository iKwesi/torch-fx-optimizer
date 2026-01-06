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