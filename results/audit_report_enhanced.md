# CO-PFN Enhanced Audit Report

**Runtime**: 2.8s | **Device**: cuda

## TEST 1: Trust Calibration
- Mean calibration error: **0.1691**
- Samples: 37
- **Expected**: < 0.1 (well-calibrated)

## TEST 2: ATE Accuracy by Trust Level
- [0.0, 0.4): MAE = 0.7730 (n=19)
- [0.4, 0.7): MAE = 0.5561 (n=18)
- [0.7, 1.0): MAE = 0.9035 (n=20)
- **Expected**: MAE decreases with trust

## TEST 3: Claim Type Performance
| Type | Mean Trust | Accuracy | Count |
| --- | --- | --- | --- |
| ADJUSTMENT_SET | 0.6370 | 57.1% | 14 |
| DIRECT_CAUSE | 0.4721 | 97.7% | 43 |
| POST_TREATMENT | 0.6349 | 50.0% | 28 |

## TEST 4: Claim Recovery (Precision@K)
- Precision@1: **90.0%**
- Precision@3: **46.7%**
- Precision@5: **37.5%**
- **Expected**: High precision means top trusted = valid

## TEST 5: Correction Head Contribution
- Mean correction magnitude: **0.0057**
- Median: 0.0038
- **Expected**: Positive and increasing with better trust

