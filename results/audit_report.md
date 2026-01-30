# Co-PFN Audit Report

## Run Configuration
| Field | Value |
| --- | --- |
| Checkpoint | checkpoints/model_adversarial_v2.pt |
| Device | cuda |
| Seed | 123 |
| Runtime (s) | 6.6 |
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
| 10 | 0.4271 | 0.4138 | 0.3092 | 3.12% | 0.3230 |
| 20 | 0.8430 | 0.8407 | 0.3057 | 0.28% | 0.3245 |
| 50 | 0.5447 | 0.5374 | 0.3312 | 1.34% | 0.3186 |
| 100 | 0.5882 | 0.5851 | 0.3449 | 0.53% | 0.3232 |
| 500 | 0.5721 | 0.5560 | 0.3426 | 2.82% | 0.3414 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.6739 | 0.2498 | 0.4242 | 75 | 20 |
| 0.2 | 0.6869 | 0.2482 | 0.4387 | 57 | 32 |
| 0.5 | 0.7372 | 0.1629 | 0.5742 | 42 | 42 |
| 0.8 | 0.6683 | 0.2704 | 0.3978 | 32 | 60 |
| 1.0 | 0.6891 | 0.2529 | 0.4362 | 20 | 75 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.2917 | 0.2655 | 0.3020 |
| 15 | 0.2369 | 0.2057 | 0.2604 |
| 20 | 0.3660 | 0.3438 | 0.3865 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.3643 | 0.8331 | 0.8369 | -0.0037 |
| GAUSS | 0.3745 | 0.1871 | 0.1475 | 0.0396 |
| SIN | 0.3281 | 0.1805 | 0.1525 | 0.0279 |
| CUBE | 0.3658 | 0.6298 | 0.6252 | 0.0046 |
| MIX | 0.3739 | 0.3233 | 0.3110 | 0.0123 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.2082 | 0.5414 |
| Worst (confounding) | 0.1459 | 0.2251 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.3473 |
| Trust (Garbage Data) | 0.0007 |
| Delta | 0.3466 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.000431 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 54 | 44.44% | 55.56% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.0009 | 0.0012 |
| Reverse Causality (re78 -> treat) | 0.0008 | 0.0012 |
| False IV (age -> treat) | 0.0008 | 0.0012 |
