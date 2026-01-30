# CO-PFN Enhanced Audit Report

**Runtime**: 1.0s | **Device**: cuda

## TEST 1: Trust Calibration
- Mean calibration error: **0.1663**
- Samples: 10
- **Expected**: < 0.1 (well-calibrated)

## TEST 2: ATE Accuracy by Trust Level
- [0.0, 0.4): MAE = 0.5415 (n=4)
- [0.4, 0.7): MAE = 0.4847 (n=5)
- [0.7, 1.0): MAE = 0.1069 (n=5)
- **Expected**: MAE decreases with trust

## TEST 3: Claim Type Performance
| Type | Mean Trust | Accuracy | Count |
| --- | --- | --- | --- |
| ADJUSTMENT_SET | 0.5492 | 50.0% | 2 |
| DIRECT_CAUSE | 0.4110 | 100.0% | 12 |
| POST_TREATMENT | 0.6722 | 75.0% | 8 |

## TEST 4: Claim Recovery (Precision@K)
- Precision@1: **100.0%**
- Precision@3: **60.0%**
- Precision@5: **46.7%**
- **Expected**: High precision means top trusted = valid

## TEST 5: Correction Head Contribution
- Mean correction magnitude: **0.0106**
- Median: 0.0091
- **Expected**: Positive and increasing with better trust

