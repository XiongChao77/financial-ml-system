# Does a single asset time series contain tradable alpha?
## Problem P-001: Multi-class vs Multiple Models
Using a single multi-class model (positive / negative / neutral), it is difficult to control the performance of positive and negative predictions independently.
A key issue is that improvements in F1-score or accuracy may be driven by better classification of the neutral class, rather than improvements in positive/negative predictions, which are more critical for trading.

## Hypothesis
This issue may arise from:
1. **Market asymmetry**
   - Long and short are structurally different; long signals are easier to capture.
   - Single multi-class model tends to bias toward the dominant side.
2. **Objective mismatch**
   - Classification metrics (accuracy, F1) are not aligned with trading objectives.
   - Improvements may be driven by neutral class rather than tradable signals.

## Proposed Approaches

To address this, consider moving from a single multi-class model to a multi-model design:
1. **Trigger + Direction**
   - Trigger: decide whether to trade
   - Direction: predict long/short

2. **Long OVR + Short OVR**
   - Separate models for long and short signals

## Next Step
- Compare:
  - Single multi-class model
  - Trigger + Direction
  - Long OVR + Short OVR

- Evaluate:
  - Positive/negative precision
  - Trading performance (PnL / Sharpe)
  - Trade frequency


# Price-Volumen Evnets Research
Interpretation of Price-Volume Patterns, Factor Construction, and Validation

Explain the following market phenomena:

Rising price with rising volume
Rising price on low volume
Rising volume with stagnant price

Interpret their market meanings, construct quantitative factors based on them, and empirically validate their predictive power.

# Price-Volume Events effect

Define discrete market events from specific price-volume patterns rather than studying general price-volume relationships.
Examples include breakout with volume expansion, low-volume advance, and rising volume with stagnant price.
Test whether these events predict subsequent return direction, persistence, and risk characteristics.


