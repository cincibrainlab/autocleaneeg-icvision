# Plan Log

## 2026-01-15: RFC-001 Strip Layout Integration

**Document**: `multi-tracing-production.qmd`

**Summary**: Created RFC proposing integration of 9-component strip layout as drop-in replacement for single-image classification. The strip layout reduces API calls from N to N/9 by batching multiple ICA components into a single image.

**Key deliverables**:
- Feature comparison table (27 items) mapping current single-image implementation to strip layout equivalents
- 4-phase implementation plan: Core Integration, Output Compatibility, CLI/API Surface, Batch Windowing
- Mermaid flowcharts documenting both architectures
- Callouts identifying 5 pending design decisions requiring stakeholder input

**Pending decisions**:
1. Strip size configuration (fixed vs configurable)
2. PDF report format in strip mode
3. Default layout behavior
4. Error handling strategy for partial batch failures
5. Validation dataset availability for accuracy comparison

**Status**: Awaiting design decision responses before implementation.

**GitHub Issue**: [#9](https://github.com/cincibrainlab/autocleaneeg-icvision/issues/9) - Feature: Strip Layout Integration for Batch ICA Classification

---

## 2026-01-15: Phase 1 Detailed Execution Plan

**Document**: `multi-tracing-production.qmd` (updated)

**Decision resolved**: Strip size fixed at 9 components per image.

**Remainder handling**: Final batch contains 1-8 components when total is not divisible by 9. Same 4-column layout with fewer rows; prompt specifies exact component count.

**Phase 1 steps defined**:
1. Step 1.1: Extract `plot_single_component_subplot()` to `plotting.py`
2. Step 1.2: Extract `create_strip_image()` to `plotting.py`
3. Step 1.3: Add `classify_strip_image()` to `api.py`
4. Step 1.4: Create batch orchestration with windowing logic
5. Step 1.5: Integration dispatch in `classify_components_batch()`

**Pitfalls identified**: Precomputed sources optimization, axis indexing, figure memory management, JSON parsing edge cases, component count validation, DataFrame schema parity.

**Status**: Ready to begin Phase 1 implementation.

---

## 2026-01-15: Phase 1 Implementation Complete

**Summary**: Implemented all Phase 1 steps for strip layout integration.

**Changes made**:

1. **`plotting.py`** - Added strip layout functions:
   - `plot_single_component_subplot()`: Plots a single ICA component into provided axes dict (topo, ts, erp, psd)
   - `create_strip_image()`: Creates a strip image with N components in 4-column layout (topo | ts | erp | psd per row)

2. **`config.py`** - Added strip prompt:
   - `STRIP_PROMPT_TEMPLATE`: Multi-component classification prompt supporting 1-52 components
   - `get_strip_prompt(n)`: Generates formatted prompt with letter labels (A-Z, AA-AZ)

3. **`api.py`** - Added strip classification functions:
   - `classify_strip_image()`: Sends strip image to API, parses JSON array response, maps letter labels to component indices
   - `classify_components_strip_batch()`: Batch orchestration with windowing (processes N components in batches of `strip_size`)
   - Updated `classify_components_batch()` with `layout` and `strip_size` parameters for dispatch

**Key implementation details**:
- Precomputed ICA sources optimization implemented
- Remainder handling: Final batch contains 1-8 components with same layout
- Error handling: Failed batches fall back to "other_artifact" classification
- DataFrame schema parity: Strip results produce identical columns as single-image

**API changes**:
```python
classify_components_batch(
    ...,
    layout="strip",      # NEW: "single" or "strip"
    strip_size=9,        # NEW: components per strip (default: 9)
)
```

**Status**: Phase 1 complete. Ready for testing and Phase 2 (output compatibility).
