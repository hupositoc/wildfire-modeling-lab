# Engineering decision record

## Context

NDVI scaling, travel-step rounding and target-mask provenance affect the meaning of the reported score.

## Decision

Expose alternative equation interpretations and require observational provenance for target-mask evaluation.

## Consequences and evidence

Synthetic returns cannot establish the thesis-reported IoU of 0.726. Original inputs and unresolved methodological choices are still needed.

See [architecture](architecture.md), [validation](validation.md) and the [case study](case-study.md) for implementation details and limits.
