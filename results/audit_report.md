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
| 10 | 0.3376 | 0.3380 | 0.3142 | -0.10% | 0.3012 |
| 20 | 0.8355 | 0.8355 | 0.2929 | -0.01% | 0.3087 |
| 50 | 0.6229 | 0.6228 | 0.2858 | 0.01% | 0.2817 |
| 100 | 0.5237 | 0.5239 | 0.2824 | -0.03% | 0.2866 |
| 500 | 0.5164 | 0.5161 | 0.3122 | 0.06% | 0.2955 |

## Corruption Sensitivity
Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.
| Corruption | Trust True | Trust False | Gap | N True | N False |
| --- | --- | --- | --- | --- | --- |
| 0.0 | 0.7360 | 0.1195 | 0.6164 | 72 | 20 |
| 0.2 | 0.7255 | 0.2894 | 0.4361 | 60 | 31 |
| 0.5 | 0.6923 | 0.2455 | 0.4467 | 43 | 41 |
| 0.8 | 0.6434 | 0.2648 | 0.3787 | 27 | 57 |
| 1.0 | 0.6690 | 0.2949 | 0.3741 | 20 | 63 |

## Scale Sweep (V=10,15,20)
| Vars | Trust Mean | Trust Min | Trust Max |
| --- | --- | --- | --- |
| 10 | 0.3152 | 0.2939 | 0.3321 |
| 15 | 0.3039 | 0.2911 | 0.3101 |
| 20 | 0.3068 | 0.2987 | 0.3165 |

## Kernel Breakdown (N=100, Direct Edge Enforced)
| Kernel | Trust | MAE None | MAE True | Gain |
| --- | --- | --- | --- | --- |
| QUAD | 0.3041 | 0.6580 | 0.6579 | 0.0000 |
| GAUSS | 0.3036 | 0.0633 | 0.0631 | 0.0002 |
| SIN | 0.2060 | 0.1251 | 0.1250 | 0.0001 |
| CUBE | 0.3060 | 0.8809 | 0.8810 | -0.0001 |
| MIX | 0.3077 | 0.4212 | 0.4211 | 0.0000 |

## Best/Worst Case Scenarios
| Scenario | Trust | MAE |
| --- | --- | --- |
| Best (direct edge) | 0.2245 | 0.7561 |
| Worst (confounding) | 0.1825 | 0.1232 |

## Garbage Data Audit
| Metric | Value |
| --- | --- |
| Trust (Real Data) | 0.2768 |
| Trust (Garbage Data) | 0.0002 |
| Delta | 0.2766 |

## Stability Across Claim Sets
| Metric | Value |
| --- | --- |
| Mean ATE Variance | 0.000000 |

## Index-Sorting Check (True DIRECT claims)
| Total | Pct src<target | Pct src>target |
| --- | --- | --- |
| 47 | 46.81% | 53.19% |

## Lalonde Audit (Real-World)
| Claim | Trust | ATE |
| --- | --- | --- |
| Adjustment (re74, re75, age, educ) | 0.5976 | -0.0203 |
| Reverse Causality (re78 -> treat) | 0.0712 | -0.0162 |
| False IV (age -> treat) | 0.3646 | -0.0146 |
