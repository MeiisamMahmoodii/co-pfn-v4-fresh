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
| 10 | 0.3342 | 0.3342 | 0.2746 | 0.00% | 0.2856 |
| 20 | 0.8306 | 0.8308 | 0.2576 | -0.02% | 0.2979 |
| 50 | 0.6171 | 0.6170 | 0.3119 | 0.02% | 0.2832 |
| 100 | 0.5237 | 0.5225 | 0.2557 | 0.22% | 0.2716 |
| 500 | 0.5276 | 0.5286 | 0.3160 | -0.19% | 0.2811 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.7286 | 0.1244 | 0.6042 | 72 | 20 |
| 0.2 | 0.7161 | 0.2763 | 0.4398 | 60 | 31 |
| 0.5 | 0.7045 | 0.2490 | 0.4555 | 43 | 41 |
| 0.8 | 0.6592 | 0.2583 | 0.4010 | 27 | 57 |
| 1.0 | 0.6592 | 0.2823 | 0.3769 | 20 | 63 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.2986 | 0.2836 | 0.3179 |
| 15 | 0.2955 | 0.2740 | 0.3105 |
| 20 | 0.2963 | 0.2819 | 0.3094 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.2968 | 0.6570 | 0.6565 | 0.0005 |
| GAUSS | 0.2874 | 0.0756 | 0.0767 | -0.0011 |
| SIN | 0.2453 | 0.1128 | 0.1097 | 0.0032 |
| CUBE | 0.2983 | 0.8829 | 0.8822 | 0.0006 |
| MIX | 0.2992 | 0.4216 | 0.4213 | 0.0003 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.1825 | 0.7021 |
| Worst (confounding) | 0.1720 | 0.1318 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.2890 |
| Trust (Garbage Data) | 0.0002 |
| Delta | 0.2889 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.000003 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 47 | 46.81% | 53.19% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.0002 | 0.0018 |
| Reverse Causality (re78 -> treat) | 0.0002 | 0.0018 |
| False IV (age -> treat) | 0.0002 | 0.0018 |
