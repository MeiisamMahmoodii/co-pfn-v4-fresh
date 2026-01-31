# CO-PFN Enhanced Audit Report

**Runtime**: 3.2s | **Device**: cuda

## TEST 1: Trust Calibration
- Mean calibration error: **0.2040**
- Samples: 36
- **Expected**: < 0.1 (well-calibrated)

## TEST 2: ATE Accuracy by Trust Level
- [0.0, 0.4): MAE = 0.4651 (n=20)
- [0.4, 0.7): MAE = 0.6828 (n=19)
- [0.7, 1.0): MAE = 0.9262 (n=16)
- **Expected**: MAE decreases with trust

## TEST 3: Claim Type Performance
| Type | Mean Trust | Accuracy | Count |
| --- | --- | --- | --- |
| ADJUSTMENT_SET | 0.4823 | 70.6% | 17 |
| DIRECT_CAUSE | 0.4184 | 95.9% | 49 |
| POST_TREATMENT | 0.5759 | 69.0% | 29 |

## TEST 4: Claim Recovery (Precision@K)
- Precision@1: **90.0%**
- Precision@3: **50.0%**
- Precision@5: **33.3%**
- **Expected**: High precision means top trusted = valid

## TEST 5: Correction Head Contribution
- Mean correction magnitude: **0.9351**
- Median: 0.6298
- **Expected**: Positive and increasing with better trust

