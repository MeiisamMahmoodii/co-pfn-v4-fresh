# CO-PFN v4 Comprehensive Testing Guide

## Overview

The CO-PFN v4 project includes **two complementary audit suites** for rigorous evaluation:

1. **`audit_suite.py`** - Original comprehensive tests (9 tests)
2. **`audit_suite_enhanced.py`** - NEW: Deeper behavioral analysis (5 additional tests)

Together, they test **all critical aspects** of the causal inference model.

---

## Original Audit Suite (`audit_suite.py`)

### Test Coverage

| Test | Purpose | Metrics |
|------|---------|---------|
| **Data Efficiency Sweep** | Does trust help with small N? | MAE, Efficiency Gain %, Trust |
| **Corruption Sensitivity** | Trust discrimination under noise | Trust True/False Gap, Corruption rate |
| **Scale Sweep** | Robustness to # variables | Trust across 10, 15, 20 variables |
| **Kernel Breakdown** | Performance by functional form | Trust, MAE for each kernel type |
| **Best/Worst Cases** | Extremes of causal structure | Direct edge vs. pure confounding |
| **Garbage Data Audit** | Rejection of fake/random data | Trust on real vs. garbage (Delta) |
| **Stability Test** | Consistency across claim sets | ATE variance |
| **Index Sorting Check** | No positional bias | src<target vs src>target distribution |
| **Lalonde Real-World** | Generalization to real data | Trust scores on causal adjustment claims |

### Running Original Suite

```bash
# Full audit (30 min)
PYTHONPATH=. python src/audit_suite.py

# Quick mode (5 min, reduced reps)
PYTHONPATH=. python src/audit_suite.py --quick

# Custom checkpoint
PYTHONPATH=. python src/audit_suite.py --checkpoint checkpoints/model_adv_e100.pt
```

### Key Output: `results/audit_report.md`

```markdown
# Corruption Sensitivity (PRIMARY METRIC)
| Corruption | Trust True | Trust False | Gap |
| 0% | 0.7132 | 0.1240 | 0.5892 |  ← Main discrimination test

# Data Efficiency (EFFECTIVENESS METRIC)  
| N=100 | MAE None | MAE True | Gain |
| | 0.5232 | 0.5234 | -0.05% |  ← How much trust helps

# Garbage Delta (SAFETY METRIC)
| Real Data | Garbage Data | Delta |
| 0.2963 | 0.0002 | 0.2961 |  ← Rejection capability
```

---

## Enhanced Audit Suite (`audit_suite_enhanced.py`)

### NEW Test Coverage

| Test | Purpose | Metrics |
|------|---------|---------|
| **Trust Calibration** | Is P(true\|trust=t) ≈ t? | Calibration error (target < 0.1) |
| **ATE by Trust Level** | High trust → better ATE? | MAE [0,0.4) vs [0.7,1.0) |
| **Claim Type Analysis** | Performance per type | Mean trust & accuracy by type |
| **Precision@K** | Top-K trustworthiness | P@1, P@3, P@5 (recall-like) |
| **Correction Analysis** | How much does correction work? | Magnitude, contribution % |

### Running Enhanced Suite

```bash
# Full enhanced audit (10 min)
PYTHONPATH=. python src/audit_suite_enhanced.py

# Quick mode (2 min)
PYTHONPATH=. python src/audit_suite_enhanced.py --quick

# Custom checkpoint
PYTHONPATH=. python src/audit_suite_enhanced.py --checkpoint checkpoints/model_adv_e80.pt
```

### Key Output: `results/audit_report_enhanced.md`

```markdown
## TEST 2: ATE Accuracy by Trust Level
- [0.0, 0.4): MAE = 0.5415 (n=4)
- [0.4, 0.7): MAE = 0.4847 (n=5)
- [0.7, 1.0): MAE = 0.1069 (n=5)  ← 5x better at high trust!

## TEST 4: Precision@K
- Precision@1: 100.0%  ← Single top claim is always valid
- Precision@3: 60.0%   ← Top 3 are 60% valid
- Precision@5: 46.7%   ← Top 5 approach baseline

## TEST 1: Trust Calibration
- Mean calibration error: 0.1663
- Status: Slightly miscalibrated (target < 0.1)
```

---

## Complete Testing Workflow

### 1. Train Model
```bash
PYTHONPATH=. python src/train.py
# Saves: checkpoints/model_adversarial_v2.pt
```

### 2. Run Original Audit
```bash
PYTHONPATH=. python src/audit_suite.py --checkpoint checkpoints/model_adversarial_v2.pt
# Output: results/audit_report.md
```

### 3. Run Enhanced Audit
```bash
PYTHONPATH=. python src/audit_suite_enhanced.py --checkpoint checkpoints/model_adversarial_v2.pt
# Output: results/audit_report_enhanced.md
```

### 4. Analyze Results
- **Corruption Gap** (original): Target ≥ 0.55
- **Garbage Delta** (original): Target ≥ 0.25
- **ATE Gain** (original): Target > 0.5% at N=100
- **Calibration Error** (enhanced): Target < 0.1
- **Precision@1** (enhanced): Target > 0.8

---

## What Each Test Actually Checks

### Original Suite Deep Dive

**Corruption Sensitivity** ⭐ MOST IMPORTANT
- Generates claims with varying corruption (0-100% false)
- Measures if trust discrimination survives noise
- Target: Gap ≥ 0.55 at 0% corruption

**Data Efficiency Sweep** ⭐ EFFECTIVENESS METRIC
- Tests ATE accuracy with N=10,20,50,100,500 samples
- Checks if knowing the true claim improves ATE
- Target: Efficiency gain > 0.5% at N=100

**Garbage Data Audit** ⭐ SAFETY METRIC
- Generates random numbers as "data"
- Checks if model rejects garbage (low trust)
- Target: Delta (real_trust - garbage_trust) > 0.25

**Scale Sweep**
- Tests robustness as variables increase
- Should maintain consistent trust ≈ 0.30

**Kernel Breakdown**
- Tests different functional forms (LINEAR, QUAD, SIN, GAUSS, etc.)
- Ensures model doesn't overfit to one type

**Lalonde Real-World**
- Tests on actual econometric dataset
- Measures real-world causal inference quality

### Enhanced Suite Deep Dive

**Trust Calibration**
- Checks: If I say trust=0.7, are 70% of those claims true?
- Measures model's uncertainty quantification
- < 0.1 error = well-calibrated

**ATE by Trust Level**
- Segments samples by trust score
- Checks if high-trust claims have better ATE
- 5x improvement (0.54 → 0.11 MAE) = strong signal

**Claim Type Analysis**
- Breaks down by DIRECT_CAUSE, ADJUSTMENT_SET, etc.
- Some types may be harder than others
- Identifies weak points in reasoning

**Precision@K**
- If I rank claims by trust, how many top-K are valid?
- Precision@1 = 100% means never misrank the best claim
- Precision@5 dropping to 47% means ranking gets harder below top-3

**Correction Analysis**
- Measures how much the correction head contributes
- If correction is near zero, trust isn't being used
- Validates that trust gates the ATE update

---

## Interpreting Results

### Ideal Performance

| Metric | Ideal | Acceptable | Poor |
|--------|-------|-----------|------|
| **Corruption Gap (0%)** | > 0.60 | 0.50-0.60 | < 0.50 |
| **Garbage Delta** | > 0.30 | 0.20-0.30 | < 0.20 |
| **Data Efficiency %** | > 1.0% | 0.5-1.0% | < 0.5% |
| **Calibration Error** | < 0.08 | 0.08-0.15 | > 0.15 |
| **Precision@1** | > 0.90 | 0.70-0.90 | < 0.70 |
| **ATE MAE (high trust)** | < 0.15 | 0.15-0.25 | > 0.25 |

### Red Flags 🚩

- **Corruption Gap < 0.50**: Model not discriminating true/false claims
- **Garbage Delta < 0.20**: Model doesn't reject nonsense data
- **Precision@1 < 0.70**: Top-ranked claims are often wrong
- **ATE doesn't improve with trust**: Correction head not working
- **Calibration Error > 0.20**: Trust scores are unreliable

### Green Lights ✅

- **Corruption Gap 0.55-0.65**: Strong discrimination across noise levels
- **Garbage Delta 0.25-0.35**: Effective safety mechanism
- **Precision@1 > 0.85**: Ranking is meaningful
- **ATE MAE drops 2-5x at high trust**: Trust is actionable
- **Calibration Error < 0.12**: Reasonable uncertainty estimates

---

## Quick Diagnostics

### If corruption gap is low (< 0.50):
```bash
# Check: Are true claims actually being learned?
grep "Trust True" results/audit_report.md
# If < 0.50, model isn't learning to identify true claims
```

### If efficiency gain is zero:
```bash
# Check: Is correction head contributing?
grep "correction" results/audit_report_enhanced.md
# If < 0.01, correction head is dead or not gated by trust
```

### If garbage delta is low (< 0.20):
```bash
# Check: Trust on real data
grep "Trust (Real" results/audit_report.md
# If also low (< 0.30), model is overall conservative
```

### If ATE doesn't improve with trust:
```bash
# Check: Precision@K
grep "Precision@" results/audit_report_enhanced.md
# If Precision@1 is low, ranking isn't working
# If it's high but ATE doesn't improve, loss weighting issue
```

---

## Continuous Monitoring

### For Each Training Iteration:
1. **After epoch 50**: Run original suite (full)
2. **After epoch 100**: Run both suites (full)
3. **If changes made**: Run enhanced suite (quick mode first)

### Tracking Progress:
```bash
# Create a comparison log
echo "Model: e100" > audit_history.txt
grep "Gap\|Delta\|Efficiency" results/audit_report.md >> audit_history.txt
echo "" >> audit_history.txt
```

---

## Future Test Recommendations

- **Adversarial robustness**: Perturb data, check trust stability
- **Sensitivity analysis**: Ablate model components, measure impact
- **Generalization**: Test on different SCM structures
- **Computational efficiency**: Memory, speed benchmarks
- **Inference confidence**: Uncertainty quantification evaluation

---

## Running Tests in CI/CD

```yaml
# Example GitHub Actions workflow
- name: Run Audit Suite
  run: |
    PYTHONPATH=. python src/audit_suite.py --quick
    PYTHONPATH=. python src/audit_suite_enhanced.py --quick
    
- name: Check Results
  run: |
    grep "Gap" results/audit_report.md | grep -q "0.5" || exit 1
    grep "Delta" results/audit_report.md | grep -q "0.2" || exit 1
```

---

## Troubleshooting

**Q: Audit hangs on GPU**
```bash
# Run on CPU to debug
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python src/audit_suite.py --quick
```

**Q: Out of memory**
```bash
# Reduce reps in config (in the script)
cfg.efficiency_reps = 5  # was 20
```

**Q: Trust scores all 0.5**
```bash
# Model probably not trained, check checkpoint file size
ls -lh checkpoints/model_adversarial_v2.pt
# Should be ~14MB, not <1MB
```

---

## Summary

✅ **Original Suite**: 9 tests covering discrimination, efficiency, safety, stability
✅ **Enhanced Suite**: 5 tests covering calibration, ranking, types, correction
✅ **Together**: 14 comprehensive tests of all model aspects
✅ **Total runtime**: ~40 min (full) or ~5 min (quick)

**Every test serves a purpose**—no tests are redundant. Each catches different failure modes that could be invisible to the others.
