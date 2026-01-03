# PyTorch FX Graph Optimizer - Learning Journey

## Project Goal
Build a graph optimization tool for PyTorch models using torch.fx that:
1. Inspects computation graphs
2. Applies real optimizations (recomputation for memory, redundancy removal)
3. Verifies numerical correctness
4. Measures performance improvements

## My Background
- No compiler experience
- No graph theory background  
- No low-level performance work
- Computational graphs: 1/10
- Automatic differentiation: 4/10
- Memory management: 1/10

## Hardware
- MacBook Pro M2 (MPS acceleration available)

## Time Budget
- 7 days total
- 4-5 hours per day
- ~28-35 hours total

## Success Criteria (End of Week)
- [ ] Working optimizer with at least one optimization pass
- [ ] Numerical correctness tests passing
- [ ] Performance benchmarks with real numbers
- [ ] Write-up documenting approach and trade-offs
- [ ] Demo notebook showing the tool in action

## Current Phase
DAY 0: Setup and crash course → DAY 1: Architecture design

## Completed
- [x] uv environment setup
- [x] PyTorch installation verified
- [x] MPS working
- [x] Project structure created
- [x] Crash course notebook completed (notebooks/00_essentials_only.ipynb)
- [x] Conv2d deep dive notebook (notebooks/conv_example.ipynb)

## Current Understanding
- ✅ Can trace models with torch.fx and inspect computation graphs
- ✅ Can iterate over graph nodes and understand graph structure
- ✅ Can perform basic graph manipulations (add/remove/replace nodes)
- ✅ Understand Conv2d shape calculations and layer stacking rules
- ✅ Ready to design the optimizer architecture

## Next Phase: ARCHITECTURE DESIGN
- [ ] Create ARCHITECTURE.md with system design
- [ ] Define optimization pass algorithms
- [ ] Design testing strategy before implementation

## Time Tracking
- **Time Spent**: ~1-1.5 hours
- **Remaining**: ~26-33 hours (out of 28-35 total)
- **Status**: On track ✅
