# Co-PFN Audit Report

## Run Configuration
| Field | Value |
| --- | --- |
| Checkpoint | checkpoints/model_adversarial_v2.pt |
| Device | cuda |
| Seed | 42 |
| Runtime (s) | 10.9 |
| Torch | 2.7.1+cu118 |

## Test Design Notes
- Data efficiency sweep uses random direct claims (not enforced to be valid) to mirror prior scripts.
- Scale sweep and kernel breakdown enforce a direct edge from T to Y to ensure the claim is true.
- Best/Worst cases are hand-constructed graphs (direct edge vs. pure confounding).
- Corruption sweep uses generator claim corruption labels as the ground truth for validity.
- Lalonde audit standardizes columns with mean/std computed from the dataset.

## Data Efficiency Sweep
| N | MAE None | MAE True | Trust True | Efficiency Gain | Trust False |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.7970 | 0.7930 | 0.2804 | 0.51% | 0.3155 |
| 20 | 0.5078 | 0.5075 | 0.3074 | 0.05% | 0.3095 |
| 50 | 0.6275 | 0.6257 | 0.3075 | 0.30% | 0.3089 |
| 100 | 0.9370 | 0.9381 | 0.2939 | -0.12% | 0.3061 |
| 500 | 0.5875 | 0.5865 | 0.3231 | 0.17% | 0.3146 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.6885 | 0.3439 | 0.3446 | 66 | 20 |
| 0.2 | 0.7086 | 0.3317 | 0.3769 | 58 | 30 |
| 0.5 | 0.6390 | 0.4616 | 0.1775 | 56 | 37 |
| 0.8 | 0.6122 | 0.3705 | 0.2417 | 30 | 62 |
| 1.0 | 0.5926 | 0.3470 | 0.2456 | 20 | 74 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.3267 | 0.3228 | 0.3306 |
| 15 | 0.3217 | 0.3196 | 0.3251 |
| 20 | 0.3192 | 0.3153 | 0.3225 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.3166 | 1.0949 | 1.0928 | 0.0021 |
| GAUSS | 0.3168 | 0.0468 | 0.0410 | 0.0058 |
| SIN | 0.2472 | 0.1136 | 0.1129 | 0.0007 |
| CUBE | 0.3175 | 0.7545 | 0.7537 | 0.0008 |
| MIX | 0.3167 | 0.3923 | 0.3900 | 0.0023 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.1485 | 0.6919 |
| Worst (confounding) | 0.1850 | 0.4078 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.3255 |
| Trust (Garbage Data) | 0.0002 |
| Delta | 0.3253 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.000018 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 53 | 45.28% | 54.72% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.8255 | -0.0046 |
| Reverse Causality (re78 -> treat) | 0.2864 | 0.0126 |
| False IV (age -> treat) | 0.8273 | -0.0044 |
