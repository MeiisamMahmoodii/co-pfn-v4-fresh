# Quick Reference: Testing CO-PFN v4

## In 30 Seconds

```bash
# Train the model
PYTHONPATH=. python src/train.py

# Run all audits (40 min)
PYTHONPATH=. python src/audit_suite.py
PYTHONPATH=. python src/audit_suite_enhanced.py

# Check results
cat results/audit_report.md
cat results/audit_report_enhanced.md
```

## Key Metrics to Watch

| Metric | File | Ideal | Red Flag |
|--------|------|-------|----------|
| **Corruption Gap** | audit_report.md | > 0.55 | < 0.40 |
| **Garbage Delta** | audit_report.md | > 0.25 | < 0.15 |
| **Data Efficiency %** | audit_report.md | > 0.5% | < 0.1% |
| **Calibration Error** | audit_report_enhanced.md | < 0.12 | > 0.20 |
| **Precision@1** | audit_report_enhanced.md | > 0.85 | < 0.60 |
| **ATE MAE (high trust)** | audit_report_enhanced.md | < 0.15 | > 0.30 |

## Tests Overview

### 9 Original Tests (`audit_suite.py`)
1. **Efficiency Sweep** - Small N performance
2. **Corruption Sensitivity** ⭐ - Main discrimination test
3. **Scale Sweep** - Robustness (10-20 vars)
4. **Kernel Breakdown** - Functional form robustness
5. **Best/Worst Cases** - Extremes
6. **Garbage Audit** ⭐ - Safety/rejection
7. **Stability Test** - ATE variance
8. **Index Sorting** - No positional bias
9. **Lalonde** - Real-world validation

### 5 Enhanced Tests (`audit_suite_enhanced.py`)
10. **Calibration** - Confidence accuracy
11. **ATE by Trust** ⭐ - Effectiveness by level
12. **Claim Types** - Type-specific performance
13. **Precision@K** - Ranking quality
14. **Correction** - Mechanism working?

## Debugging Guide

```
❌ Corruption gap < 0.40?
→ Model not discriminating true/false
→ Check: Trust True & Trust False scores in audit_report.md

❌ Garbage Delta < 0.15?
→ Model accepts nonsense data
→ Check: Trust(Garbage) in audit_report.md (should be ~0)

❌ Efficiency gain ~0%?
→ Trust not helping ATE
→ Check: Correction magnitude in audit_report_enhanced.md

❌ Precision@1 < 0.70?
→ Ranking is broken
→ Check: Top claim types in audit_report_enhanced.md

❌ Calibration > 0.20?
→ Confidence estimates are unreliable
→ Check: Trust distribution in both reports
```

## Quick Mode (5 min total)

```bash
PYTHONPATH=. python src/audit_suite.py --quick
PYTHONPATH=. python src/audit_suite_enhanced.py --quick
```

Runs with reduced repetitions:
- 5 reps instead of 20
- Same tests, lower computational cost
- Good for iteration during development

## Files to Know

| File | Purpose |
|------|---------|
| `src/audit_suite.py` | 9 original comprehensive tests |
| `src/audit_suite_enhanced.py` | 5 new behavioral tests |
| `results/audit_report.md` | Original test results |
| `results/audit_report_enhanced.md` | Enhanced test results |
| `TESTING_GUIDE.md` | Full documentation & interpretation |
| `TESTING_SUMMARY.md` | This summary with context |

## Expected Workflow

```
Train (30 min)
    ↓
Original Audit (30 min) → Check corruption gap & garbage delta
    ↓
Enhanced Audit (10 min) → Check calibration & precision@K
    ↓
Interpret (5 min) → Read TESTING_GUIDE.md for metrics
    ↓
Iterate → Retrain if needed
```

## Pro Tips

- **First time?** Read TESTING_GUIDE.md entirely (10 min)
- **Iterating fast?** Use `--quick` mode until ready for full validation
- **Debugging?** Run enhanced suite only to drill down
- **Confident?** Run full both suites before finalizing
- **Publishing?** Include both audit_report files in results

## Failure Mode Detection

### Can Original Suite Detect?
- ✅ Model learns nothing (gap < 0.3)
- ✅ Garbage acceptance (delta < 0.2)
- ✅ Scale degradation
- ✅ Real-world failure
- ❌ Uncalibrated confidence
- ❌ Broken ranking
- ❌ Dead mechanisms

### Can Enhanced Suite Detect?
- ✅ Uncalibrated confidence
- ✅ Ranking failures
- ✅ Mechanism problems
- ✅ Type-specific issues
- ✅ Trust not improving ATE
- ❌ Garbage acceptance (original catches this)
- ❌ Real-world failure (original catches this)

### Together = Full Coverage ✅

## One-Liner Commands

```bash
# Everything: train → audit original → audit enhanced
PYTHONPATH=. python src/train.py && PYTHONPATH=. python src/audit_suite.py && PYTHONPATH=. python src/audit_suite_enhanced.py

# Just audit current model
PYTHONPATH=. python src/audit_suite.py && PYTHONPATH=. python src/audit_suite_enhanced.py

# Quick check
PYTHONPATH=. python src/audit_suite_enhanced.py --quick

# Check specific model
PYTHONPATH=. python src/audit_suite.py --checkpoint checkpoints/model_adv_e50.pt
```

## Timeline Estimates

| Task | Time |
|------|------|
| Full Training | 30-40 min |
| Original Audit (full) | 25-30 min |
| Original Audit (quick) | 3-5 min |
| Enhanced Audit (full) | 8-10 min |
| Enhanced Audit (quick) | 1-2 min |
| **Both (full)** | **40 min** |
| **Both (quick)** | **5 min** |

## Key Insight

The **original suite** catches "did the model learn?" failures.  
The **enhanced suite** catches "does the mechanism work?" failures.  
**Together** they catch everything important.
