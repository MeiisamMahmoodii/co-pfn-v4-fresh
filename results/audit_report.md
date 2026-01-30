# Co-PFN Audit Report

## Run Configuration
| Field | Value |
| --- | --- |
| Checkpoint | checkpoints/model_adversarial_v2.pt |
| Device | cuda |
| Seed | 123 |
| Runtime (s) | 6.2 |
| Torch | 2.10.0+cu128 |

## Test Design Notes
- Data efficiency sweep uses random direct claims (not enforced to be valid) to mirror prior scripts.
- Scale sweep and kernel breakdown enforce a direct edge from T to Y to ensure the claim is true.
- Best/Worst cases are hand-constructed graphs (direct edge vs. pure confounding).
- Corruption sweep uses generator claim corruption labels as the ground truth for validity.
- Lalonde audit standardizes columns with mean/std computed from the dataset.

## Data Efficiency Sweep
| N | MAE None | MAE True | Trust True | Efficiency Gain | Trust False |
| --- | --- | --- | --- | --- | --- |
| 10 | 0.4029 | 0.4042 | 0.2269 | -0.32% | 0.2194 |
| 20 | 0.8457 | 0.8416 | 0.2238 | 0.49% | 0.1960 |
| 50 | 0.5364 | 0.5352 | 0.2293 | 0.21% | 0.2140 |
| 100 | 0.5895 | 0.5871 | 0.2901 | 0.41% | 0.2309 |
| 500 | 0.5569 | 0.5573 | 0.2456 | -0.07% | 0.2636 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.6683 | 0.2676 | 0.4007 | 75 | 20 |
| 0.2 | 0.6935 | 0.2169 | 0.4765 | 57 | 32 |
| 0.5 | 0.7317 | 0.1747 | 0.5570 | 42 | 42 |
| 0.8 | 0.6730 | 0.2664 | 0.4066 | 32 | 60 |
| 1.0 | 0.6879 | 0.2286 | 0.4593 | 20 | 75 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.2346 | 0.2004 | 0.2701 |
| 15 | 0.1906 | 0.1688 | 0.2250 |
| 20 | 0.2546 | 0.2122 | 0.2923 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.2511 | 0.8390 | 0.8387 | 0.0004 |
| GAUSS | 0.2615 | 0.1263 | 0.1342 | -0.0079 |
| SIN | 0.1484 | 0.1387 | 0.1392 | -0.0005 |
| CUBE | 0.2495 | 0.6249 | 0.6253 | -0.0004 |
| MIX | 0.2403 | 0.3059 | 0.3083 | -0.0024 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.1778 | 0.5631 |
| Worst (confounding) | 0.1418 | 0.2087 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.2568 |
| Trust (Garbage Data) | 0.0004 |
| Delta | 0.2564 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.000063 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 54 | 44.44% | 55.56% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.0022 | -0.0013 |
| Reverse Causality (re78 -> treat) | 0.0005 | -0.0013 |
| False IV (age -> treat) | 0.0008 | -0.0013 |
