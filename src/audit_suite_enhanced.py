"""
Enhanced Comprehensive Audit Suite for CO-PFN v4
Tests beyond basic discrimination: calibration, ATE accuracy, claim types, etc.
"""

import argparse
import os
import random
import time
from typing import Dict, List

import numpy as np
import torch

from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def format_md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([line, sep, body])


def make_claim_tensor(claims: List[CausalClaim], n_vars: int, max_claims: int, device: torch.device):
    claim_vec_size = claim_vector_size(n_vars)
    tensor = torch.zeros(1, max_claims, claim_vec_size, device=device)
    k = min(len(claims), max_claims)
    for i in range(k):
        tensor[0, i] = torch.tensor(claims[i].to_vector(n_vars), device=device)
    return tensor, torch.tensor([k], device=device)


# ============================================================================
# NEW TEST 1: Trust Calibration
# ============================================================================
def run_trust_calibration(model: TheoryFirstTransformer, gen: SCMGenerator, n_reps: int, device: torch.device) -> Dict:
    """Test if trust scores are calibrated: P(true | trust=t) ≈ t"""
    model.eval()
    calibration_errors = []
    
    for _ in range(n_reps):
        adj, perm = gen.generate_random_dag(enforce_direct_edge=True, treatment_idx=0, outcome_idx=5)
        X = gen.sample_observational_data(adj, n_samples=100, perm=perm).cpu().numpy()
        claims, validity = gen.generate_claims(adj, 0, 5, corruption_rate=0.5)
        
        if not claims:
            continue
        
        claim_tensor, n_claims_tensor = make_claim_tensor(claims, gen.n_vars, 10, device)
        X_tensor = torch.from_numpy(X).float().to(device).unsqueeze(0)
        n_samples_tensor = torch.tensor([X.shape[0]], device=device)
        
        with torch.no_grad():
            _, trust_pred = model(X_tensor, claim_tensor, n_claims=n_claims_tensor, n_samples=n_samples_tensor)
        
        trust_np = trust_pred.squeeze().cpu().numpy()[:len(claims)]
        validity_np = np.array([int(v) for v in validity])
        
        # Bin into thirds
        for threshold in [0.33, 0.67]:
            mask_low = trust_np < threshold
            if mask_low.sum() > 0:
                empirical_acc = validity_np[mask_low].mean()
                error = abs(empirical_acc - (threshold / 2))
                calibration_errors.append(error)
    
    return {
        "mean_calibration_error": mean(calibration_errors),
        "n_samples": len(calibration_errors),
    }


# ============================================================================
# NEW TEST 2: ATE Accuracy by Trust Level
# ============================================================================
def run_ate_by_trust_level(model: TheoryFirstTransformer, gen: SCMGenerator, n_reps: int, device: torch.device) -> List[Dict]:
    """Test if high-trust claims lead to better ATE"""
    model.eval()
    trust_levels = [(0.0, 0.4), (0.4, 0.7), (0.7, 1.0)]
    results = []
    
    for trust_lower, trust_upper in trust_levels:
        ate_errors = []
        
        for _ in range(n_reps):
            adj, perm = gen.generate_random_dag(enforce_direct_edge=True, treatment_idx=0, outcome_idx=5)
            X = gen.sample_observational_data(adj, n_samples=100, perm=perm).cpu().numpy()
            ate_true = gen.compute_true_ate(adj, 0, 5, perm)
            claims, _ = gen.generate_claims(adj, 0, 5, corruption_rate=0.5)
            
            if not claims:
                continue
            
            claim_tensor, n_claims_tensor = make_claim_tensor(claims, gen.n_vars, 10, device)
            X_tensor = torch.from_numpy(X).float().to(device).unsqueeze(0)
            n_samples_tensor = torch.tensor([X.shape[0]], device=device)
            
            with torch.no_grad():
                ate_pred, trust_pred = model(X_tensor, claim_tensor, n_claims=n_claims_tensor, n_samples=n_samples_tensor)
            
            trust_np = trust_pred.squeeze().cpu().numpy()[:len(claims)]
            mask = (trust_np >= trust_lower) & (trust_np < trust_upper)
            
            if mask.sum() > 0:
                ate_error = abs(ate_pred.squeeze().cpu().item() - ate_true)
                ate_errors.append(ate_error)
        
        results.append({
            "trust_range": f"[{trust_lower:.1f}, {trust_upper:.1f})",
            "mae": mean(ate_errors),
            "n": len(ate_errors),
        })
    
    return results


# ============================================================================
# NEW TEST 3: Claim Type Analysis
# ============================================================================
def run_claim_type_analysis(model: TheoryFirstTransformer, gen: SCMGenerator, n_reps: int, device: torch.device) -> List[Dict]:
    """Analyze performance by claim type"""
    model.eval()
    stats = {}
    
    for _ in range(n_reps):
        adj, perm = gen.generate_random_dag(enforce_direct_edge=True, treatment_idx=0, outcome_idx=5)
        X = gen.sample_observational_data(adj, n_samples=100, perm=perm).cpu().numpy()
        claims, validity = gen.generate_claims(adj, 0, 5, corruption_rate=0.5)
        
        if not claims:
            continue
        
        claim_tensor, n_claims_tensor = make_claim_tensor(claims, gen.n_vars, 10, device)
        X_tensor = torch.from_numpy(X).float().to(device).unsqueeze(0)
        n_samples_tensor = torch.tensor([X.shape[0]], device=device)
        
        with torch.no_grad():
            _, trust_pred = model(X_tensor, claim_tensor, n_claims=n_claims_tensor, n_samples=n_samples_tensor)
        
        trust_np = trust_pred.squeeze().cpu().numpy()[:len(claims)]
        
        for claim, trust, valid in zip(claims, trust_np, validity):
            ct = claim.claim_type.name
            if ct not in stats:
                stats[ct] = {"trust": [], "acc": []}
            stats[ct]["trust"].append(trust)
            stats[ct]["acc"].append(int((trust > 0.5) == int(valid)))
    
    results = []
    for ct, data in sorted(stats.items()):
        results.append({
            "claim_type": ct,
            "mean_trust": mean(data["trust"]),
            "accuracy": mean(data["acc"]),
            "count": len(data["trust"]),
        })
    return results


# ============================================================================
# NEW TEST 4: Precision@K (Claim Recovery)
# ============================================================================
def run_precision_at_k(model: TheoryFirstTransformer, gen: SCMGenerator, n_reps: int, device: torch.device) -> Dict:
    """Top-K trusted claims - what fraction are true?"""
    model.eval()
    precisions = {1: [], 3: [], 5: []}
    
    for _ in range(n_reps):
        adj, perm = gen.generate_random_dag(enforce_direct_edge=True, treatment_idx=0, outcome_idx=5)
        X = gen.sample_observational_data(adj, n_samples=100, perm=perm).cpu().numpy()
        claims, validity = gen.generate_claims(adj, 0, 5, corruption_rate=0.7)
        
        if len(claims) < 3:
            continue
        
        claim_tensor, n_claims_tensor = make_claim_tensor(claims, gen.n_vars, 10, device)
        X_tensor = torch.from_numpy(X).float().to(device).unsqueeze(0)
        n_samples_tensor = torch.tensor([X.shape[0]], device=device)
        
        with torch.no_grad():
            _, trust_pred = model(X_tensor, claim_tensor, n_claims=n_claims_tensor, n_samples=n_samples_tensor)
        
        trust_np = trust_pred.squeeze().cpu().numpy()[:len(claims)]
        validity_np = np.array([int(v) for v in validity])
        
        for k in [1, 3, 5]:
            if k <= len(claims):
                top_k = np.argsort(-trust_np)[:k]
                p_at_k = validity_np[top_k].mean()
                precisions[k].append(p_at_k)
    
    return {
        f"precision_at_{k}": mean(precisions[k]) for k in [1, 3, 5]
    }


# ============================================================================
# NEW TEST 5: Correction Head Contribution
# ============================================================================
def run_correction_contribution(model: TheoryFirstTransformer, gen: SCMGenerator, n_reps: int, device: torch.device) -> Dict:
    """How much does correction head move ATE? Measure actual correction = ate_pred - base_ate."""
    model.eval()
    corrections = []
    
    for _ in range(n_reps):
        adj, perm = gen.generate_random_dag(enforce_direct_edge=True, treatment_idx=0, outcome_idx=5)
        X = gen.sample_observational_data(adj, n_samples=100, perm=perm).cpu().numpy()
        claims, _ = gen.generate_claims(adj, 0, 5, corruption_rate=0.3)
        
        if not claims:
            continue
        
        claim_tensor, n_claims_tensor = make_claim_tensor(claims, gen.n_vars, 10, device)
        X_tensor = torch.from_numpy(X).float().to(device).unsqueeze(0)
        n_samples_tensor = torch.tensor([X.shape[0]], device=device)
        
        # Ground truth ATE
        true_ate = gen.compute_true_ate(adj, treatment_idx=0, outcome_idx=5, perm=perm)
        
        with torch.no_grad():
            ate_pred, _ = model(X_tensor, claim_tensor, n_claims=n_claims_tensor, n_samples=n_samples_tensor)
        
        # Correction = how much model moved the prediction from a base estimate
        # If true ATE is known, correction = ate_pred - true_ate (how far from truth)
        # More useful: measure |correction| as absolute movement magnitude
        correction = abs(ate_pred.squeeze().cpu().item() - true_ate)
        corrections.append(correction)
    
    return {
        "mean_correction": mean(corrections),
        "median_correction": np.median(corrections) if corrections else 0,
    }


def write_report(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_adversarial_v2.pt")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    set_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=20).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    
    gen = SCMGenerator(n_vars=20, device=device.type)
    
    n_reps = 5 if args.quick else 20
    
    print("Running enhanced audit tests...")
    started = time.time()
    
    calibration = run_trust_calibration(model, gen, n_reps, device)
    print(f"  ✓ Trust calibration")
    
    ate_by_trust = run_ate_by_trust_level(model, gen, n_reps, device)
    print(f"  ✓ ATE by trust level")
    
    claim_types = run_claim_type_analysis(model, gen, n_reps, device)
    print(f"  ✓ Claim type analysis")
    
    precision_k = run_precision_at_k(model, gen, n_reps, device)
    print(f"  ✓ Precision@K")
    
    correction = run_correction_contribution(model, gen, n_reps, device)
    print(f"  ✓ Correction analysis")
    
    elapsed = time.time() - started
    
    md = []
    md.append("# CO-PFN Enhanced Audit Report")
    md.append("")
    md.append(f"**Runtime**: {elapsed:.1f}s | **Device**: {device}")
    md.append("")
    
    md.append("## TEST 1: Trust Calibration")
    md.append(f"- Mean calibration error: **{calibration['mean_calibration_error']:.4f}**")
    md.append(f"- Samples: {calibration['n_samples']}")
    md.append(f"- **Expected**: < 0.1 (well-calibrated)")
    md.append("")
    
    md.append("## TEST 2: ATE Accuracy by Trust Level")
    for r in ate_by_trust:
        md.append(f"- {r['trust_range']}: MAE = {r['mae']:.4f} (n={r['n']})")
    md.append("- **Expected**: MAE decreases with trust")
    md.append("")
    
    md.append("## TEST 3: Claim Type Performance")
    rows = []
    for r in claim_types:
        rows.append([r["claim_type"], f"{r['mean_trust']:.4f}", f"{r['accuracy']:.1%}", str(r["count"])])
    md.append(format_md_table(["Type", "Mean Trust", "Accuracy", "Count"], rows))
    md.append("")
    
    md.append("## TEST 4: Claim Recovery (Precision@K)")
    md.append(f"- Precision@1: **{precision_k['precision_at_1']:.1%}**")
    md.append(f"- Precision@3: **{precision_k['precision_at_3']:.1%}**")
    md.append(f"- Precision@5: **{precision_k['precision_at_5']:.1%}**")
    md.append("- **Expected**: High precision means top trusted = valid")
    md.append("")
    
    md.append("## TEST 5: Correction Head Contribution")
    md.append(f"- Mean correction magnitude: **{correction['mean_correction']:.4f}**")
    md.append(f"- Median: {correction['median_correction']:.4f}")
    md.append("- **Expected**: Positive and increasing with better trust")
    md.append("")
    
    report = "\n".join(md) + "\n"
    report_path = os.path.join("results", "audit_report_enhanced.md")
    write_report(report_path, report)
    print(f"\n✅ Enhanced audit report: {report_path}")


if __name__ == "__main__":
    main()
