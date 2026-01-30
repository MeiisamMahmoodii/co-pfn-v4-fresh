# Co-PFN Audit Report

## Run Configuration
| Field | Value |
| --- | --- |
| Checkpoint | checkpoints/model_adversarial_v2.pt |
| Device | cuda |
| Seed | 123 |
| Runtime (s) | 8.8 |
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
| 10 | 0.4478 | 0.4199 | 0.3392 | 6.24% | 0.3542 |
| 20 | 0.8836 | 0.8554 | 0.3152 | 3.20% | 0.3309 |
| 50 | 0.5636 | 0.5458 | 0.3432 | 3.15% | 0.3266 |
| 100 | 0.6078 | 0.5963 | 0.3231 | 1.88% | 0.3134 |
| 500 | 0.5690 | 0.5606 | 0.3182 | 1.48% | 0.2914 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.6725 | 0.2681 | 0.4044 | 75 | 20 |
| 0.2 | 0.6891 | 0.2281 | 0.4610 | 57 | 32 |
| 0.5 | 0.7304 | 0.1691 | 0.5612 | 42 | 42 |
| 0.8 | 0.6703 | 0.2530 | 0.4173 | 32 | 60 |
| 1.0 | 0.6729 | 0.2494 | 0.4235 | 20 | 75 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.2049 | 0.1693 | 0.2384 |
| 15 | 0.3527 | 0.3081 | 0.3822 |
| 20 | 0.4202 | 0.4045 | 0.4405 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.4270 | 0.8697 | 0.8457 | 0.0240 |
| GAUSS | 0.4070 | 0.1373 | 0.1173 | 0.0199 |
| SIN | 0.3512 | 0.1840 | 0.1451 | 0.0389 |
| CUBE | 0.4242 | 0.6332 | 0.6264 | 0.0069 |
| MIX | 0.4146 | 0.2966 | 0.3000 | -0.0034 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.3405 | 0.6018 |
| Worst (confounding) | 0.3046 | 0.2301 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.3289 |
| Trust (Garbage Data) | 0.0011 |
| Delta | 0.3279 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.001639 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 54 | 44.44% | 55.56% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.0011 | 0.0022 |
| Reverse Causality (re78 -> treat) | 0.0010 | 0.0022 |
| False IV (age -> treat) | 0.0011 | 0.0022 |
