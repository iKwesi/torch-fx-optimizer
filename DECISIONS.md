# Architecture Decisions & Deviations

## Decision 1: Attempting Hard Mode for Recomputation Pass
**Date**: January 3, 2026
**Decision**: Upgrade from "simple wrapper" to "true graph-level checkpointing"

### Original Plan (from ARCHITECTURE.md)
- Use `torch.utils.checkpoint` to wrap model sections
- Simple every-Nth-layer policy
- Model wrapper approach (easier, less impressive)

### Revised Plan (Hard Mode)
- Attempt direct torch.fx graph modification
- Insert custom checkpoint nodes into the graph
- Implement recomputation at the graph IR level
- This is PhD-level difficulty

### Rationale
- Original plan was too conservative
- Goal is to become "so good they can't ignore me"
- Learning from attempting hard things > completing easy things
- Even partial success demonstrates deeper understanding
- Failure with deep learning > success without learning

### Risk Assessment
**High risk of failure**: This is genuinely hard
**Mitigation**: 
- Deep learning phase (2 days) before implementation
- Systematic debug protocols
- Fallback to hybrid approach if needed
- Document everything (failure is valuable)

### Success Criteria (Revised)
**Best case**: Working graph-level checkpointing, measurable memory reduction
**Good case**: Partial implementation + deep understanding documented
**Acceptable case**: Attempted implementation + comprehensive writeup of why it's hard

All three outcomes are valuable for job hunting.

### Alignment with Original Architecture
- Milestones 1-3: NO CHANGE (core infrastructure, redundancy elimination)
- Milestone 4: UPGRADED (more ambitious recomputation)
- Milestones 5-6: NO CHANGE (benchmarks, documentation)

### Documentation Trail
- Original: ARCHITECTURE.md (Section 3, Pass 2)
- Deep dive: docs/recomputation_deep_dive.md (created today)
- This decision: DECISIONS.md

### Approval
Self-approved. This is a learning project, we're choosing the harder path intentionally.

---
