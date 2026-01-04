# Graph-Level Gradient Checkpointing: Deep Dive

**Purpose**: Prepare you to attempt PhD-level gradient checkpointing at the torch.fx graph level

**Difficulty**: 🔥🔥🔥🔥🔥 (Expert - you might fail, and that's okay)

**Prerequisites**: Understanding of torch.fx basics, computation graphs, backpropagation

---

## Introduction: What You're Attempting

You're trying to implement **automatic gradient checkpointing** by transforming torch.fx graphs. This is research-level work that combines:
- Deep understanding of PyTorch's autograd system
- Graph transformation at a level most engineers never touch
- Bridging static analysis (torch.fx) with dynamic execution (autograd)

**Why this is hard**: torch.fx gives you a static computation graph, but gradient checkpointing requires dynamic runtime behavior during the backward pass. You're essentially trying to rewrite how PyTorch computes gradients for specific parts of the graph.

**Be prepared to**:
- Read PyTorch source code
- Debug cryptic autograd errors
- Potentially hit fundamental limitations

**If successful**, you'll have done something that would impress NVIDIA researchers.

Let's build the foundation you need.

---

## 1. THE FUNDAMENTAL PROBLEM

### What is an Activation?

An **activation** is the output tensor of a layer/operation during the forward pass.

```python
import torch
import torch.nn as nn

# Simple example
x = torch.randn(1, 3, 224, 224)  # Input image
conv = nn.Conv2d(3, 64, kernel_size=3)

activation = conv(x)  # This is an activation
# Shape: (1, 64, 222, 222)
# Memory: 1 * 64 * 222 * 222 * 4 bytes (float32) = ~12.5 MB
```

The activation is **not** the layer itself - it's the **data** flowing through the layer.

### Why Store Activations?

During backpropagation, PyTorch needs activations to compute gradients. Let's see why:

**Example: Simple ReLU**

```python
# Forward pass
x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
y = torch.relu(x)  # y = [1.0, 0.0, 3.0]

# Backward pass
loss = y.sum()
loss.backward()

# To compute dx, PyTorch needs to know:
# - Where x was positive (output = x)
# - Where x was negative (output = 0, gradient = 0)
# This information is in the ACTIVATION y
```

**ReLU backward formula**:
```
dy/dx = 1 if x > 0
dy/dx = 0 if x <= 0
```

To apply this formula, PyTorch must **remember** which values in x were positive. It does this by **storing the activation**.

### Concrete Example: 5-Layer Network Memory

Let's trace memory usage step by step:

```python
import torch
import torch.nn as nn

class FiveLayerNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Each conv outputs 64 channels, 224x224 spatial
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)

    def forward(self, x):
        # Track memory at each step
        print(f"Input x:  {x.shape}, {x.element_size() * x.numel() / 1024**2:.2f} MB")

        x1 = torch.relu(self.conv1(x))  # STORED for backward
        print(f"After layer 1: {x1.shape}, {x1.element_size() * x1.numel() / 1024**2:.2f} MB")

        x2 = torch.relu(self.conv2(x1))  # STORED for backward
        print(f"After layer 2: {x2.shape}, {x2.element_size() * x2.numel() / 1024**2:.2f} MB")

        x3 = torch.relu(self.conv3(x2))  # STORED for backward
        print(f"After layer 3: {x3.shape}, {x3.element_size() * x3.numel() / 1024**2:.2f} MB")

        x4 = torch.relu(self.conv4(x3))  # STORED for backward
        print(f"After layer 4: {x4.shape}, {x4.element_size() * x4.numel() / 1024**2:.2f} MB")

        x5 = torch.relu(self.conv5(x4))  # STORED for backward
        print(f"After layer 5: {x5.shape}, {x5.element_size() * x5.numel() / 1024**2:.2f} MB")

        return x5

# Run it
model = FiveLayerNet()
input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)
output = model(input_tensor)

# Output:
# Input x:  torch.Size([1, 3, 224, 224]), 0.57 MB
# After layer 1: torch.Size([1, 64, 224, 224]), 12.25 MB
# After layer 2: torch.Size([1, 64, 224, 224]), 12.25 MB
# After layer 3: torch.Size([1, 64, 224, 224]), 12.25 MB
# After layer 4: torch.Size([1, 64, 224, 224]), 12.25 MB
# After layer 5: torch.Size([1, 64, 224, 224]), 12.25 MB

# Total memory: 0.57 + 12.25*5 = ~61.82 MB (just for activations!)
```

**Memory accumulation**:
- Input: 0.57 MB
- After layer 1: +12.25 MB (cumulative: 12.82 MB)
- After layer 2: +12.25 MB (cumulative: 25.07 MB)
- After layer 3: +12.25 MB (cumulative: 37.32 MB)
- After layer 4: +12.25 MB (cumulative: 49.57 MB)
- After layer 5: +12.25 MB (cumulative: 61.82 MB)

**The problem**: For a 50-layer ResNet, this becomes **gigabytes** of memory.

---

## 2. HOW PYTORCH'S AUTOGRAD WORKS (Simplified)

### The Backward Graph

When you set `requires_grad=True`, PyTorch builds a **backward graph** (also called the autograd graph) alongside the forward computation.

**Forward graph**: What operations to compute
**Backward graph**: How to compute gradients

```python
# Example: y = x^2
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2

# Forward: y = [4.0, 9.0]

# Backward graph that PyTorch builds internally:
# y.grad_fn = <PowBackward0>
# This grad_fn knows:
#   1. The operation (power)
#   2. The exponent (2)
#   3. The input value (x) - STORED!

print(y.grad_fn)  # <PowBackward0 object>
```

### What Are 'Saved Tensors'?

Saved tensors are inputs/outputs that PyTorch stores to compute gradients later.

**Example: Matrix Multiplication**

```python
# Forward: Y = X @ W
X = torch.randn(10, 20, requires_grad=True)
W = torch.randn(20, 30, requires_grad=True)
Y = X @ W

# Backward formulas:
# dL/dX = dL/dY @ W.T
# dL/dW = X.T @ dL/dY

# What PyTorch must save:
# - For dL/dX: W (to compute dL/dY @ W.T)
# - For dL/dW: X (to compute X.T @ dL/dY)

# Check what was saved:
print(Y.grad_fn.saved_tensors)  # (X, W)
```

**Key insight**: The more complex the operation, the more tensors need to be saved.

### Detailed Example: y = x^2 Backward

Let's trace exactly what happens:

```python
import torch

# Forward pass
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2  # y = [4.0, 9.0]

# PyTorch internally does:
# 1. Compute y = [4.0, 9.0]
# 2. Create PowBackward grad_fn
# 3. Save x = [2.0, 3.0] in grad_fn.saved_tensors
# 4. Set y.grad_fn = PowBackward

# Backward pass
loss = y.sum()  # loss = 13.0
loss.backward()

# PyTorch internally does:
# 1. Start with dL/dloss = 1.0
# 2. Call loss.grad_fn.backward(1.0)
#    -> Computes dL/dy = [1.0, 1.0]
# 3. Call y.grad_fn.backward([1.0, 1.0])
#    -> Uses formula: dy/dx = 2*x
#    -> Retrieves saved x = [2.0, 3.0]
#    -> Computes: dL/dx = dL/dy * dy/dx = [1.0, 1.0] * 2*[2.0, 3.0] = [4.0, 6.0]
# 4. Stores result in x.grad

print(x.grad)  # tensor([4., 6.])
```

**The backward function signature**:
```python
class PowBackward:
    def __init__(self, exponent):
        self.exponent = exponent
        self.saved_tensors = []

    def forward(self, x):
        self.saved_tensors.append(x)  # SAVE INPUT
        return x ** self.exponent

    def backward(self, grad_output):
        x = self.saved_tensors[0]  # RETRIEVE SAVED INPUT
        # dy/dx = exponent * x^(exponent-1)
        grad_input = grad_output * self.exponent * (x ** (self.exponent - 1))
        return grad_input
```

**Memory issue**: `saved_tensors` holds activations until backward completes.

---

## 3. WHAT torch.utils.checkpoint ACTUALLY DOES

### The Standard Usage

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)

    def forward(self, x):
        # Normal: stores layer1 output
        x = self.layer1(x)

        # Checkpointed: does NOT store layer2 output
        x = checkpoint(self.layer2, x)

        # Normal: stores layer3 output
        x = self.layer3(x)
        return x
```

### How checkpoint() Works Internally

Here's the actual mechanism (simplified from PyTorch source):

```python
def checkpoint(function, *args):
    """
    Checkpoint a function to save memory during backward pass.
    """
    # In forward pass:
    # 1. Run function WITHOUT grad tracking
    with torch.no_grad():
        output = function(*args)

    # 2. Detach output (break autograd link)
    output = output.detach()

    # 3. Wrap in custom autograd.Function that will recompute
    output = CheckpointFunction.apply(function, output, *args)

    return output


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, output, *args):
        # Save the function and inputs (NOT the output!)
        ctx.run_function = run_function
        ctx.save_for_backward(*args)

        # Return the precomputed output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved inputs
        args = ctx.saved_tensors

        # RECOMPUTE the forward pass with grad tracking
        with torch.enable_grad():
            # Detach args to avoid double-backprop issues
            detached_args = [arg.detach().requires_grad_(True) for arg in args]
            output = ctx.run_function(*detached_args)

        # Now compute gradients normally
        torch.autograd.backward(output, grad_output)

        # Return gradients for inputs
        grads = [arg.grad for arg in detached_args]
        return (None, None) + tuple(grads)  # None for run_function, output
```

### What Gets Saved vs Discarded

**WITHOUT checkpointing**:
```python
x = layer1(input)  # Saves: x (the activation)
y = layer2(x)      # Saves: y (the activation)
z = layer3(y)      # Saves: z (the activation)

# Memory: input + x + y + z
```

**WITH checkpointing** on layer2:
```python
x = layer1(input)              # Saves: x
y = checkpoint(layer2, x)      # Saves: x (input), NOT y (output)
z = layer3(y)                  # Saves: z

# Memory: input + x + z (y is NOT saved)

# During backward:
# - Need gradient of y to backprop to layer2
# - Recompute: y = layer2(x) using saved x
# - Now can compute gradients
```

**Key trade-off**:
- **Saved**: ~50% memory (only x, not y)
- **Cost**: Recompute layer2 during backward (adds time)

### The Code Flow

```
FORWARD PASS:
1. Compute layer1(input) → x          [saves x]
2. Enter checkpoint:
   a. Compute layer2(x) → y (no grad)
   b. Detach y
   c. Save x in CheckpointFunction
   d. Return y                         [does NOT save y]
3. Compute layer3(y) → z               [saves z]

BACKWARD PASS:
1. Gradient flows back to layer3
2. Need gradient of layer2 output (y):
   a. CheckpointFunction.backward() is called
   b. Retrieve saved x
   c. RECOMPUTE y = layer2(x) with grad
   d. Compute gradients w.r.t. x
3. Gradient flows back to layer1
```

---

## 4. THE HARD PART: Doing This at the Graph Level

### Why This Is Fundamentally Difficult

**Problem 1: Static vs Dynamic**

```python
# torch.fx gives you a STATIC graph:
traced_model = torch.fx.symbolic_trace(model)
print(traced_model.graph)

# Output:
# graph():
#     %x : torch.Tensor [#users=1] = placeholder[target=x]
#     %conv1 : [#users=1] = call_module[target=conv1](args = (%x,))
#     %relu1 : [#users=1] = call_function[target=torch.relu](args = (%conv1,))
#     %conv2 : [#users=1] = call_module[target=conv2](args = (%relu1,))
#     return %conv2

# This is just a LIST of operations!
# NO DYNAMIC BEHAVIOR (like recomputation during backward)
```

**Problem 2: Autograd Lives Outside the Graph**

```python
# The torch.fx graph describes FORWARD computation only
# Backward computation is handled by PyTorch's autograd engine
# torch.fx CANNOT directly modify backward behavior
```

**Problem 3: When Does Recomputation Happen?**

```python
# Recomputation needs to happen DURING BACKWARD PASS
# But torch.fx transformations happen BEFORE FORWARD PASS
# You're trying to schedule future dynamic behavior from static analysis
```

### The Core Challenge

```
You need to:
1. Identify checkpoints in the STATIC graph
2. Inject DYNAMIC behavior (recomputation) into the backward pass
3. Do this WITHOUT breaking PyTorch's autograd system

This requires bridging two worlds that normally don't talk to each other.
```

---

## 5. THREE POSSIBLE APPROACHES

### Approach A: Insert checkpoint() Calls in the Graph

**Idea**: Transform the graph to wrap certain operations in `torch.utils.checkpoint.checkpoint()`.

**How it works**:
```python
# Original graph:
def forward(x):
    x = layer1(x)
    x = layer2(x)  # ← Mark for checkpointing
    x = layer3(x)
    return x

# Transformed graph:
def forward(x):
    x = layer1(x)
    x = checkpoint(layer2, x)  # ← Wrapped!
    x = layer3(x)
    return x
```

**Implementation sketch**:
```python
import torch.fx as fx
from torch.utils.checkpoint import checkpoint

def apply_checkpointing_pass(graph_module: fx.GraphModule):
    graph = graph_module.graph

    # Identify nodes to checkpoint (every other layer, for example)
    checkpoint_nodes = identify_checkpoint_candidates(graph)

    for node in checkpoint_nodes:
        # Create a wrapper function
        def checkpointed_layer(*args):
            # Get the original module
            module = graph_module.get_submodule(node.target)
            return checkpoint(module, *args)

        # Replace node with checkpointed version
        with graph.inserting_after(node):
            new_node = graph.call_function(checkpointed_layer, args=node.args)
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)

    graph_module.recompile()
    return graph_module
```

**Pros**:
- ✅ Uses PyTorch's built-in checkpointing (reliable)
- ✅ Relatively simple to implement
- ✅ Guaranteed to work with autograd

**Cons**:
- ❌ Coarse-grained (checkpoints entire modules, not individual ops)
- ❌ May not trace correctly (checkpoint uses Python control flow)
- ❌ Limited control over checkpointing policy

**Difficulty**: ⭐⭐⭐ (Medium-Hard)

**Likelihood of success**: 70%

---

### Approach B: Custom autograd.Function for Checkpointed Regions

**Idea**: Identify regions in the graph to checkpoint, extract them into custom autograd Functions that handle recomputation.

**How it works**:
```python
# Original graph:
def forward(x):
    x = layer1(x)
    x = layer2(x)  # ← These two
    x = layer3(x)  # ← are one region
    x = layer4(x)
    return x

# Transformed:
class CheckpointedRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer2_params, layer3_params):
        # Run without tracking
        with torch.no_grad():
            x = layer2(x)
            x = layer3(x)

        # Save inputs for recomputation
        ctx.save_for_backward(x, layer2_params, layer3_params)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Recompute and backprop
        ...

def forward(x):
    x = layer1(x)
    x = CheckpointedRegion.apply(x, layer2.parameters(), layer3.parameters())
    x = layer4(x)
    return x
```

**Implementation challenges**:
```python
# 1. How do you extract a subgraph?
# 2. How do you capture all parameters?
# 3. How do you handle module state (BatchNorm running stats)?
# 4. How do you recompile the graph with the custom function?
```

**Pros**:
- ✅ Fine-grained control
- ✅ Can optimize checkpoint placement
- ✅ Full control over forward/backward

**Cons**:
- ❌ Very complex to implement correctly
- ❌ Need to handle parameter passing, state management
- ❌ Easy to introduce subtle bugs
- ❌ May break with certain operations (in-place, BatchNorm, etc.)

**Difficulty**: ⭐⭐⭐⭐⭐ (Expert)

**Likelihood of success**: 30%

---

### Approach C: Modify the Backward Graph Directly

**Idea**: Intercept PyTorch's backward pass and inject recomputation logic.

**How it (theoretically) works**:
```python
# Register backward hooks on specific operations
# During backward, instead of using saved tensors, recompute them

def backward_hook(module, grad_input, grad_output):
    # Detect if this is a checkpointed layer
    if is_checkpointed(module):
        # Recompute forward using saved inputs
        recomputed = recompute_forward(module, saved_inputs)
        # Use recomputed activations for gradient
        ...
```

**Reality**: This is extremely difficult because:
- You don't have direct access to PyTorch's backward engine
- Hooks have limited power
- Very easy to break autograd invariants

**Pros**:
- ✅ Maximum control (in theory)

**Cons**:
- ❌ Requires deep PyTorch internals knowledge
- ❌ Extremely fragile
- ❌ Likely to hit fundamental limitations
- ❌ Not recommended

**Difficulty**: ⭐⭐⭐⭐⭐+ (Research-level)

**Likelihood of success**: <10%

---

## 6. WHAT WE'RE GOING TO ATTEMPT

### Recommended Approach: Hybrid of A + Simplified B

**Strategy**:
1. Start with Approach A (inserting checkpoint() calls)
2. If successful, attempt a simplified version of B for specific patterns

**Phase 1: Basic Checkpointing (Approach A)**

```python
# Goal: Transform torch.fx graph to insert checkpoint() calls

import torch
import torch.fx as fx
from torch.utils.checkpoint import checkpoint

def checkpoint_pass(model: fx.GraphModule, checkpoint_every: int = 2):
    """
    Insert checkpoint() calls into graph for every Nth module.

    Args:
        model: Traced GraphModule
        checkpoint_every: Checkpoint every Nth call_module node

    Returns:
        Modified GraphModule with checkpointing
    """
    graph = model.graph
    module_nodes = [n for n in graph.nodes if n.op == 'call_module']

    for i, node in enumerate(module_nodes):
        if i % checkpoint_every != 0:
            continue  # Only checkpoint every Nth module

        # Get the module
        module = model.get_submodule(node.target)

        # Create checkpoint wrapper
        # CHALLENGE: How to insert this into the graph?

        # Option 1: Replace with call_function to checkpoint
        # Option 2: Modify the module itself to use checkpointing
        # Option 3: Create intermediate function node

        # Implementation details TBD (this is where research begins)

    model.recompile()
    return model
```

**Specific steps**:

1. **Identify checkpointable modules**
   - Must be `call_module` nodes (not `call_function`)
   - Must be stateless or handle state carefully
   - Must not have in-place operations

2. **Create checkpoint wrappers**
   ```python
   def make_checkpoint_wrapper(module):
       def wrapper(*args, **kwargs):
           return checkpoint(module, *args, **kwargs)
       return wrapper
   ```

3. **Insert into graph**
   - This is the tricky part
   - May need to use graph.call_function with custom wrapper
   - Or use graph rewriting techniques

4. **Verify correctness**
   - Test that gradients match
   - Test that memory is reduced
   - Test on various model architectures

**What PyTorch internals you need**:
- `torch.fx.Graph` manipulation (you already know this)
- `torch.fx.GraphModule.get_submodule()` to access modules
- Understanding of how `checkpoint()` integrates with autograd
- Graph recompilation and validation

**What could go wrong**:
1. **Checkpoint doesn't trace**: `checkpoint()` uses Python control flow internally, may break tracing
   - **Solution**: Apply checkpointing AFTER initial tracing

2. **Module state issues**: BatchNorm has running stats that checkpoint doesn't handle well
   - **Solution**: Only checkpoint stateless modules (Conv, Linear, ReLU)

3. **In-place operations**: checkpoint() can break with in-place ops
   - **Solution**: Detect and skip modules with in-place ops

4. **Gradient correctness**: Subtle bugs in graph transformation
   - **Solution**: Extensive testing with gradient comparison

**Phase 2: Advanced (If Phase 1 Works)**

If basic checkpointing works, attempt:
- Custom autograd.Function for specific patterns (e.g., Conv-BN-ReLU blocks)
- Optimal checkpoint placement using graph analysis
- Handle more complex graph patterns (skip connections, etc.)

---

## 7. FALLBACK PLAN

### If Full Implementation Proves Impossible

**What you can still accomplish**:

1. **Proof of Concept for Simple Cases**
   ```python
   # Demonstrate checkpointing works for:
   # - Sequential models (no skip connections)
   # - Specific layer types (Conv2d, Linear)
   # - Show memory reduction on toy examples
   ```

2. **Detailed Analysis Document**
   ```markdown
   # Why Graph-Level Checkpointing Is Hard
   - Technical deep dive into the challenges
   - What you attempted
   - Where you hit limitations
   - Potential solutions (with research references)
   ```

3. **Partial Solution with Manual Annotation**
   ```python
   # Allow users to manually mark layers for checkpointing
   # Your tool automates the graph transformation

   model = MyModel()
   traced = fx.symbolic_trace(model)

   # User specifies which layers to checkpoint
   checkpointed = apply_checkpoints(
       traced,
       checkpoint_layers=['layer2', 'layer4', 'layer6']
   )
   ```

4. **Research-Quality Writeup**
   ```markdown
   # What Would Make This Impressive:
   - Show you understand WHY it's hard (autograd internals)
   - Demonstrate working solution for constrained cases
   - Propose novel approaches (even if unimplemented)
   - Compare to existing work (cite PyTorch source, research papers)
   - Provide reproducible experiments
   ```

### Alternative: Demonstrate Deep Understanding

Even if you don't get full checkpointing working, you can impress by:

**1. Comprehensive Tutorial**
- Write the definitive guide to PyTorch autograd + torch.fx interaction
- Include diagrams, code examples, edge cases
- This alone would be valuable to community

**2. Working Examples of Graph Transformations**
- Simpler optimizations that DO work
- Show mastery of torch.fx
- Demonstrate rigorous testing methodology

**3. Novel Insights**
- Discover new limitations or possibilities
- Propose theoretical solutions
- Identify specific PyTorch issues that need fixing

**4. Clean, Professional Code**
- Even partial implementation with excellent tests
- Clear documentation
- Reproducible benchmarks

---

## 8. LEARNING RESOURCES

### PyTorch Source Code to Read

1. **torch/utils/checkpoint.py**
   - Location: `pytorch/torch/utils/checkpoint.py`
   - Read: `CheckpointFunction` class
   - Understand: How forward/backward are implemented

2. **torch/autograd/function.py**
   - Location: `pytorch/torch/autograd/function.py`
   - Read: `Function` base class
   - Understand: `ctx.save_for_backward()`, `ctx.saved_tensors`

3. **torch/fx/graph.py**
   - Location: `pytorch/torch/fx/graph.py`
   - Read: Graph manipulation methods
   - Understand: `inserting_after`, `node_copy`, `erase_node`

### Research Papers (Optional but Helpful)

1. **"Training Deep Nets with Sublinear Memory Cost"** (Chen et al., 2016)
   - Optimal checkpointing algorithm
   - O(√n) memory instead of O(n)

2. **"Gradient Checkpointing for Deep Learning"** (various authors)
   - Search for recent papers on gradient/activation checkpointing
   - See how researchers implement this

### Debugging Tools

```python
# Print saved tensors for a node
def print_saved_tensors(tensor):
    if tensor.grad_fn:
        print(f"Saved tensors: {tensor.grad_fn.saved_tensors}")

# Hook to monitor memory
def memory_hook(module, input, output):
    print(f"{module.__class__.__name__}: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

model.register_forward_hook(memory_hook)
```

---

## 9. RECOMMENDED TIMELINE

### Week 1: Foundation (where you are now)

**Day 1-2**: Deep dive into PyTorch autograd
- Read checkpoint.py source
- Experiment with custom autograd.Function
- Understand saved_tensors mechanism

**Day 3-4**: Attempt Approach A
- Implement basic graph transformation
- Test on simple sequential models
- Debug tracing issues

**Day 5**: Evaluation
- Does it work? How well?
- Measure memory reduction
- Identify failures

**Day 6-7**: Iterate or pivot
- If working: extend to more cases
- If not working: document why, attempt simplified version

---

## 10. FINAL ADVICE

### Expect to Fail

**This is PhD-level work**. Most people with PyTorch experience couldn't do this. If you:
- Get basic checkpointing working for simple cases: **Success** ✅
- Fully understand why it's hard and document it: **Success** ✅
- Implement manual checkpointing with good UX: **Success** ✅
- Achieve automatic checkpointing for all cases: **Heroic** 🏆

### Focus on Learning

The **real value** is what you learn:
- Deep PyTorch internals knowledge
- Graph transformation skills
- Autograd expertise
- Research methodology

These skills are more valuable than the specific feature.

### Document Everything

Keep a research log:
```markdown
## Day 1: Understanding checkpoint()
- Read source code in pytorch/torch/utils/checkpoint.py
- Key insight: checkpoint() uses custom autograd.Function
- Experiment: Created minimal checkpoint example
- Question: How to integrate with torch.fx?

## Day 2: First attempt at graph transformation
- Tried: Inserting call_function node for checkpoint
- Result: Graph doesn't trace correctly
- Error: [specific error message]
- Next: Try different approach...
```

This log becomes your writeup.

### When to Ask for Help

If you're stuck for >4 hours on the same issue:
1. Search PyTorch issues/forums
2. Read related source code
3. Try a different approach
4. Document the blocker

Don't waste days on a dead end - pivot to something achievable.

---

## 11. YOUR FIRST TASK

To verify your understanding, implement this **minimal checkpointing example**:

```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(100, 100)
        self.layer2 = nn.Linear(100, 100)
        self.layer3 = nn.Linear(100, 100)

    def forward(self, x):
        x = self.layer1(x)
        # TODO: Checkpoint layer2
        x = self.layer2(x)  # ← Make this checkpointed
        x = self.layer3(x)
        return x

# Test that:
# 1. Forward pass works
# 2. Backward pass works
# 3. Gradients are correct (compare with/without checkpoint)
# 4. Memory is reduced

# Measure memory before/after
# Verify gradient correctness
```

**Once you can do this manually**, you understand the basics.

**Then try**: Automate this transformation using torch.fx.

**If you succeed at that**, you're ready for the hard version.

---

## Summary: The Path Forward

1. **Understand the fundamentals** (autograd, checkpointing)
2. **Get manual checkpointing working** perfectly
3. **Attempt torch.fx transformation** for simple cases
4. **Iterate or pivot** based on results
5. **Document deeply** what you learn
6. **Be proud** of how far you get

Even partial success here would be remarkable.

Good luck. 🚀
