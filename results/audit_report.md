# Co-PFN Audit Report

## Run Configuration
| Field | Value |
| --- | --- |
| Checkpoint | checkpoints/model_adversarial_v2.pt |
| Device | cuda |
| Seed | 123 |
| Runtime (s) | 6.5 |
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
| 10 | 0.3331 | 0.3331 | 0.2787 | -0.00% | 0.2989 |
| 20 | 0.8335 | 0.8339 | 0.2770 | -0.04% | 0.2719 |
| 50 | 0.6232 | 0.6229 | 0.2845 | 0.05% | 0.3211 |
| 100 | 0.5232 | 0.5234 | 0.3274 | -0.05% | 0.3042 |
| 500 | 0.5210 | 0.5209 | 0.3039 | 0.01% | 0.2882 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.7132 | 0.1240 | 0.5892 | 72 | 20 |
| 0.2 | 0.6970 | 0.2948 | 0.4022 | 60 | 31 |
| 0.5 | 0.6743 | 0.2335 | 0.4409 | 43 | 41 |
| 0.8 | 0.6634 | 0.2397 | 0.4236 | 27 | 57 |
| 1.0 | 0.6618 | 0.2957 | 0.3661 | 20 | 63 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.1828 | 0.1693 | 0.1998 |
| 15 | 0.2164 | 0.1939 | 0.2260 |
| 20 | 0.2592 | 0.2078 | 0.2744 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.2629 | 0.6572 | 0.6571 | 0.0000 |
| GAUSS | 0.2634 | 0.0666 | 0.0666 | -0.0000 |
| SIN | 0.2605 | 0.1091 | 0.1073 | 0.0018 |
| CUBE | 0.2660 | 0.8818 | 0.8818 | -0.0000 |
| MIX | 0.2745 | 0.4220 | 0.4220 | 0.0001 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.2063 | 0.7302 |
| Worst (confounding) | 0.1681 | 0.1122 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.2963 |
| Trust (Garbage Data) | 0.0002 |
| Delta | 0.2961 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.000018 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 47 | 46.81% | 53.19% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.3606 | -0.0130 |
| Reverse Causality (re78 -> treat) | 0.1791 | -0.0054 |
| False IV (age -> treat) | 0.1482 | -0.0057 |
