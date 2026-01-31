import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator, KernelType
from src.data.claims import CausalClaim, ClaimType, claim_vector_size
from src.lalonde_audit import run_lalonde_audit


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def var(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / len(vals)


def format_md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([line, sep, body])


def make_claim_tensor(
    claims: List[CausalClaim],
    n_vars: int,
    max_claims: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    claim_vec_size = claim_vector_size(n_vars)
    tensor = torch.zeros(1, max_claims, claim_vec_size, device=device)
    k = min(len(claims), max_claims)
    for i in range(k):
        tensor[0, i] = torch.tensor(claims[i].to_vector(n_vars), device=device)
    return tensor, torch.tensor([k], device=device)


def ensure_direct_edge(gen: SCMGenerator, t: int, y: int, max_tries: int = 50) -> Tuple[torch.Tensor, np.ndarray]:
    adj, perm = gen.generate_random_dag()
    for _ in range(max_tries):
        if adj[y, t].item() != 0:
            return adj, perm
        adj, perm = gen.generate_random_dag()
    return adj, perm


@dataclass
class AuditConfig:
    checkpoint: str
    n_vars: int = 20
    max_claims: int = 10
    seed: int = 123
    efficiency_samples: Tuple[int, ...] = (10, 20, 50, 100, 500)
    efficiency_reps: int = 20
    corruption_rates: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.8, 1.0)
    corruption_reps: int = 20
    scale_vars: Tuple[int, ...] = (10, 15, 20)
    scale_reps: int = 20
    kernel_reps: int = 20
    stability_worlds: int = 5
    best_worst_reps: int = 10
    garbage_reps: int = 20


def run_efficiency_sweep(model: TheoryFirstTransformer, gen: SCMGenerator, cfg: AuditConfig, device: torch.device) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for n_samples in cfg.efficiency_samples:
        mae_none = []
        mae_true = []
        mae_false = []
        trust_true = []
        trust_false = []
        for _ in range(cfg.efficiency_reps):
            adj, perm = gen.generate_random_dag()
            t_idx = np.random.randint(0, cfg.n_vars)
            y_idx = np.random.randint(0, cfg.n_vars)
            while y_idx == t_idx:
                y_idx = np.random.randint(0, cfg.n_vars)

            data = gen.sample_observational_data(adj, n_samples, perm=perm).unsqueeze(0).to(device)
            true_ate = gen.compute_true_ate(adj, t_idx, y_idx, perm=perm)

            # No claims
            c_none, n_claims_none = make_claim_tensor([], cfg.n_vars, cfg.max_claims, device)
            with torch.no_grad():
                ate_none, _ = model(data, c_none, n_claims=n_claims_none)
            mae_none.append(abs(ate_none.item() - true_ate))

            # Direct claim (not enforced to be correct)
            c_true, n_claims_true = make_claim_tensor(
                [CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)],
                cfg.n_vars,
                cfg.max_claims,
                device,
            )
            with torch.no_grad():
                ate_true, trust_t = model(data, c_true, n_claims=n_claims_true)
            mae_true.append(abs(ate_true.item() - true_ate))
            trust_true.append(trust_t[0, 0, 0].item())

            # Reverse claim
            c_false, n_claims_false = make_claim_tensor(
                [CausalClaim(ClaimType.DIRECT_CAUSE, [y_idx], t_idx)],
                cfg.n_vars,
                cfg.max_claims,
                device,
            )
            with torch.no_grad():
                ate_false, trust_f = model(data, c_false, n_claims=n_claims_false)
            mae_false.append(abs(ate_false.item() - true_ate))
            trust_false.append(trust_f[0, 0, 0].item())

        results[n_samples] = {
            "mae_none": mean(mae_none),
            "mae_true": mean(mae_true),
            "mae_false": mean(mae_false),
            "trust_true": mean(trust_true),
            "trust_false": mean(trust_false),
            "eff_gain_pct": 100.0 * (mean(mae_none) - mean(mae_true)) / mean(mae_none) if mean(mae_none) else float("nan"),
        }
    return results


def run_corruption_sweep(model: TheoryFirstTransformer, gen: SCMGenerator, cfg: AuditConfig, device: torch.device) -> List[Dict[str, float]]:
    results = []
    for rate in cfg.corruption_rates:
        true_trusts = []
        false_trusts = []
        for _ in range(cfg.corruption_reps):
            adj, perm = gen.generate_random_dag()
            t_idx = np.random.randint(0, cfg.n_vars)
            y_idx = np.random.randint(0, cfg.n_vars)
            while y_idx == t_idx:
                y_idx = np.random.randint(0, cfg.n_vars)

            claims, validity = gen.generate_claims(adj, t_idx, y_idx, corruption_rate=rate)
            if not claims:
                continue

            data = gen.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)
            c_tensor, n_claims_t = make_claim_tensor(claims, cfg.n_vars, cfg.max_claims, device)
            with torch.no_grad():
                _, trust_scores = model(data, c_tensor, n_claims=n_claims_t)

            trust_vals = trust_scores[0, : len(claims), 0].tolist()
            for t_val, is_valid in zip(trust_vals, validity):
                if is_valid:
                    true_trusts.append(float(t_val))
                else:
                    false_trusts.append(float(t_val))

        results.append({
            "rate": rate,
            "trust_true": mean(true_trusts),
            "trust_false": mean(false_trusts),
            "gap": mean(true_trusts) - mean(false_trusts),
            "n_true": len(true_trusts),
            "n_false": len(false_trusts),
        })
    return results


def run_scale_sweep(model: TheoryFirstTransformer, cfg: AuditConfig, device: torch.device) -> List[Dict[str, float]]:
    results = []
    for v in cfg.scale_vars:
        gen_v = SCMGenerator(n_vars=v, device=device.type)
        trusts = []
        for _ in range(cfg.scale_reps):
            adj, perm = ensure_direct_edge(gen_v, 0, v - 1)
            data_raw = gen_v.sample_observational_data(adj, 100, perm=perm)
            data = torch.zeros(1, 100, cfg.n_vars, device=device)
            data[0, :, :v] = data_raw

            claim = CausalClaim(ClaimType.DIRECT_CAUSE, [0], v - 1)
            c_tensor, n_claims_t = make_claim_tensor([claim], cfg.n_vars, cfg.max_claims, device)

            with torch.no_grad():
                _, trust_scores = model(data, c_tensor, n_claims=n_claims_t)
            trusts.append(trust_scores[0, 0, 0].item())

        results.append({
            "vars": v,
            "trust_mean": mean(trusts),
            "trust_min": min(trusts) if trusts else float("nan"),
            "trust_max": max(trusts) if trusts else float("nan"),
        })
    return results


def run_kernel_breakdown(model: TheoryFirstTransformer, cfg: AuditConfig, device: torch.device) -> List[Dict[str, float]]:
    kernels = [KernelType.QUAD, KernelType.GAUSS, KernelType.SIN, KernelType.CUBE, KernelType.MIX]
    results = []
    for k in kernels:
        gen_k = SCMGenerator(n_vars=cfg.n_vars, device=device.type, kernel_pool=[k])
        mae_none = []
        mae_true = []
        trusts = []
        for _ in range(cfg.kernel_reps):
            adj, perm = ensure_direct_edge(gen_k, 0, cfg.n_vars - 1)
            data = gen_k.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)
            true_ate = gen_k.compute_true_ate(adj, 0, cfg.n_vars - 1, perm=perm)

            c_none, n_none = make_claim_tensor([], cfg.n_vars, cfg.max_claims, device)
            with torch.no_grad():
                ate_none, _ = model(data, c_none, n_claims=n_none)

            claim = CausalClaim(ClaimType.DIRECT_CAUSE, [0], cfg.n_vars - 1)
            c_true, n_true = make_claim_tensor([claim], cfg.n_vars, cfg.max_claims, device)
            with torch.no_grad():
                ate_true, trust_scores = model(data, c_true, n_claims=n_true)

            mae_none.append(abs(ate_none.item() - true_ate))
            mae_true.append(abs(ate_true.item() - true_ate))
            trusts.append(trust_scores[0, 0, 0].item())

        results.append({
            "kernel": k.value,
            "mae_none": mean(mae_none),
            "mae_true": mean(mae_true),
            "trust": mean(trusts),
            "gain": mean(mae_none) - mean(mae_true),
        })
    return results


def run_best_worst_case(model: TheoryFirstTransformer, cfg: AuditConfig, device: torch.device) -> Dict[str, Dict[str, float]]:
    gen = SCMGenerator(n_vars=cfg.n_vars, device=device.type)
    best_trusts, worst_trusts = [], []
    best_ates, worst_ates = [], []

    for _ in range(cfg.best_worst_reps):
        # Best case: direct edge only
        adj_best = torch.zeros(cfg.n_vars, cfg.n_vars, device=device)
        t_idx, y_idx = 0, 1
        adj_best[y_idx, t_idx] = 1.0
        perm = np.arange(cfg.n_vars)
        data = gen.sample_observational_data(adj_best, 200, perm=perm).unsqueeze(0).to(device)
        true_ate = gen.compute_true_ate(adj_best, t_idx, y_idx, perm=perm)

        claim = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
        c_tensor, n_claims_t = make_claim_tensor([claim], cfg.n_vars, cfg.max_claims, device)
        with torch.no_grad():
            ate_pred, trust_scores = model(data, c_tensor, n_claims=n_claims_t)
        best_trusts.append(trust_scores[0, 0, 0].item())
        best_ates.append(abs(ate_pred.item() - true_ate))

        # Worst case: pure confounding, no direct edge
        adj_worst = torch.zeros(cfg.n_vars, cfg.n_vars, device=device)
        z_idx = 2
        adj_worst[t_idx, z_idx] = 1.0
        adj_worst[y_idx, z_idx] = 1.0
        data_w = gen.sample_observational_data(adj_worst, 200, perm=perm).unsqueeze(0).to(device)
        true_ate_w = gen.compute_true_ate(adj_worst, t_idx, y_idx, perm=perm)

        claim_w = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
        c_tensor_w, n_claims_w = make_claim_tensor([claim_w], cfg.n_vars, cfg.max_claims, device)
        with torch.no_grad():
            ate_pred_w, trust_scores_w = model(data_w, c_tensor_w, n_claims=n_claims_w)
        worst_trusts.append(trust_scores_w[0, 0, 0].item())
        worst_ates.append(abs(ate_pred_w.item() - true_ate_w))

    return {
        "best": {"trust": mean(best_trusts), "mae": mean(best_ates)},
        "worst": {"trust": mean(worst_trusts), "mae": mean(worst_ates)},
    }


def run_garbage_audit(model: TheoryFirstTransformer, gen: SCMGenerator, cfg: AuditConfig, device: torch.device) -> Dict[str, float]:
    trust_real = []
    trust_garbage = []
    for _ in range(cfg.garbage_reps):
        adj, perm = gen.generate_random_dag()
        t_idx = np.random.randint(0, cfg.n_vars)
        y_idx = np.random.randint(0, cfg.n_vars)
        while y_idx == t_idx:
            y_idx = np.random.randint(0, cfg.n_vars)

        data = gen.sample_observational_data(adj, 200, perm=perm).unsqueeze(0).to(device)
        garbage = torch.randn_like(data)

        claim = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
        c_tensor, n_claims_t = make_claim_tensor([claim], cfg.n_vars, cfg.max_claims, device)

        with torch.no_grad():
            _, trust_scores = model(data, c_tensor, n_claims=n_claims_t)
            _, trust_scores_g = model(garbage, c_tensor, n_claims=n_claims_t)

        trust_real.append(trust_scores[0, 0, 0].item())
        trust_garbage.append(trust_scores_g[0, 0, 0].item())

    return {
        "trust_real": mean(trust_real),
        "trust_garbage": mean(trust_garbage),
        "delta": mean(trust_real) - mean(trust_garbage),
    }


def run_stability_test(model: TheoryFirstTransformer, cfg: AuditConfig, device: torch.device) -> Dict[str, float]:
    gen = SCMGenerator(n_vars=cfg.n_vars, device=device.type)
    variances = []
    for _ in range(cfg.stability_worlds):
        adj, perm = ensure_direct_edge(gen, 0, cfg.n_vars - 1)
        data = gen.sample_observational_data(adj, 200, perm=perm).unsqueeze(0).to(device)
        true_claim = CausalClaim(ClaimType.DIRECT_CAUSE, [0], cfg.n_vars - 1)
        lie_claim = CausalClaim(ClaimType.DIRECT_CAUSE, [cfg.n_vars - 1], 0)

        scenarios = [
            [],
            [true_claim],
            [lie_claim],
            [true_claim, lie_claim],
        ]

        ates = []
        for claims in scenarios:
            c_tensor, n_claims_t = make_claim_tensor(claims, cfg.n_vars, cfg.max_claims, device)
            with torch.no_grad():
                ate_pred, _ = model(data, c_tensor, n_claims=n_claims_t)
            ates.append(ate_pred.item())
        variances.append(var(ates))

    return {"mean_ate_variance": mean(variances)}


def run_index_sorting_check(gen: SCMGenerator, cfg: AuditConfig) -> Dict[str, float]:
    forward = 0
    reverse = 0
    total = 0
    for _ in range(200):
        adj, _ = gen.generate_random_dag()
        t_idx = np.random.randint(0, cfg.n_vars)
        y_idx = np.random.randint(0, cfg.n_vars)
        while y_idx == t_idx:
            y_idx = np.random.randint(0, cfg.n_vars)

        claims, validity = gen.generate_claims(adj, t_idx, y_idx, corruption_rate=0.0)
        for c, v in zip(claims, validity):
            if c.claim_type == ClaimType.DIRECT_CAUSE and v:
                src = c.variables[0]
                tgt = c.target
                if src < tgt:
                    forward += 1
                elif src > tgt:
                    reverse += 1
                total += 1
    return {
        "total_true_direct": total,
        "pct_src_lt_tgt": 100.0 * forward / total if total else 0.0,
        "pct_src_gt_tgt": 100.0 * reverse / total if total else 0.0,
    }
    return {
        "efficiency": efficiency,
        "corruption": corruption,
        "scale": scale,
        "kernels": kernels,
        "best_worst": best_worst,
        "garbage": garbage,
        "stability": stability,
        "index_sorting": index_sorting,
        "lalonde": lalonde,
        "elapsed": elapsed
    }


def write_report(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def generate_report_md(cfg: AuditConfig, metrics: Dict) -> str:
    elapsed = metrics["elapsed"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    efficiency = metrics["efficiency"]
    corruption = metrics["corruption"]
    scale = metrics["scale"]
    kernels = metrics["kernels"]
    best_worst = metrics["best_worst"]
    garbage = metrics["garbage"]
    stability = metrics["stability"]
    index_sorting = metrics["index_sorting"]
    lalonde = metrics["lalonde"]
    
    md = []
    md.append("# Co-PFN Audit Report")
    md.append("")
    md.append("## Run Configuration")
    md.append(format_md_table(
        ["Field", "Value"],
        [
            ["Checkpoint", cfg.checkpoint],
            ["Device", str(device)],
            ["Seed", str(cfg.seed)],
            ["Runtime (s)", f"{elapsed:.1f}"],
            ["Torch", torch.__version__],
        ],
    ))

    md.append("")
    md.append("## Test Design Notes")
    md.append("- Data efficiency sweep uses random direct claims (not enforced to be valid) to mirror prior scripts.")
    md.append("- Scale sweep and kernel breakdown enforce a direct edge from T to Y to ensure the claim is true.")
    md.append("- Best/Worst cases are hand-constructed graphs (direct edge vs. pure confounding).")
    md.append("- Corruption sweep uses generator claim corruption labels as the ground truth for validity.")
    md.append("- Lalonde audit standardizes columns with mean/std computed from the dataset.")
    md.append("")
    md.append("## Data Efficiency Sweep")
    rows = []
    for n in cfg.efficiency_samples:
        r = efficiency[n]
        rows.append([
            str(n),
            f"{r['mae_none']:.4f}",
            f"{r['mae_true']:.4f}",
            f"{r['trust_true']:.4f}",
            f"{r['eff_gain_pct']:.2f}%",
            f"{r['trust_false']:.4f}",
        ])
    md.append(format_md_table(
        ["N", "MAE None", "MAE True", "Trust True", "Efficiency Gain", "Trust False"],
        rows,
    ))

    md.append("")
    md.append("## Corruption Sensitivity")
    md.append("Note: When corruption=1.0 there may be zero valid claims, so Trust True can be NaN.")
    rows = []
    for r in corruption:
        rows.append([
            f"{r['rate']:.1f}",
            f"{r['trust_true']:.4f}",
            f"{r['trust_false']:.4f}",
            f"{r['gap']:.4f}",
            f"{r['n_true']}",
            f"{r['n_false']}",
        ])
    md.append(format_md_table(
        ["Corruption", "Trust True", "Trust False", "Gap", "N True", "N False"],
        rows,
    ))

    md.append("")
    md.append("## Scale Sweep (V=10,15,20)")
    rows = []
    for r in scale:
        rows.append([
            str(r["vars"]),
            f"{r['trust_mean']:.4f}",
            f"{r['trust_min']:.4f}",
            f"{r['trust_max']:.4f}",
        ])
    md.append(format_md_table(
        ["Vars", "Trust Mean", "Trust Min", "Trust Max"],
        rows,
    ))

    md.append("")
    md.append("## Kernel Breakdown (N=100, Direct Edge Enforced)")
    rows = []
    for r in kernels:
        rows.append([
            r["kernel"],
            f"{r['trust']:.4f}",
            f"{r['mae_none']:.4f}",
            f"{r['mae_true']:.4f}",
            f"{r['gain']:.4f}",
        ])
    md.append(format_md_table(
        ["Kernel", "Trust", "MAE None", "MAE True", "Gain"],
        rows,
    ))

    md.append("")
    md.append("## Best/Worst Case Scenarios")
    md.append(format_md_table(
        ["Scenario", "Trust", "MAE"],
        [
            ["Best (direct edge)", f"{best_worst['best']['trust']:.4f}", f"{best_worst['best']['mae']:.4f}"],
            ["Worst (confounding)", f"{best_worst['worst']['trust']:.4f}", f"{best_worst['worst']['mae']:.4f}"],
        ],
    ))

    md.append("")
    md.append("## Garbage Data Audit")
    md.append(format_md_table(
        ["Metric", "Value"],
        [
            ["Trust (Real Data)", f"{garbage['trust_real']:.4f}"],
            ["Trust (Garbage Data)", f"{garbage['trust_garbage']:.4f}"],
            ["Delta", f"{garbage['delta']:.4f}"],
        ],
    ))

    md.append("")
    md.append("## Stability Across Claim Sets")
    md.append(format_md_table(
        ["Metric", "Value"],
        [["Mean ATE Variance", f"{stability['mean_ate_variance']:.6f}"]],
    ))

    md.append("")
    md.append("## Index-Sorting Check (True DIRECT claims)")
    md.append(format_md_table(
        ["Total", "Pct src<target", "Pct src>target"],
        [[
            str(index_sorting["total_true_direct"]),
            f"{index_sorting['pct_src_lt_tgt']:.2f}%",
            f"{index_sorting['pct_src_gt_tgt']:.2f}%",
        ]],
    ))

    md.append("")
    md.append("## Lalonde Audit (Real-World)")
    rows = []
    for r in lalonde:
        rows.append([r["claim"], f"{r['trust']:.4f}", f"{r['ate']:.4f}"])
    md.append(format_md_table(
        ["Claim", "Trust", "ATE"],
        rows,
    ))

    return "\n".join(md) + "\n"


def print_terminal_summary(metrics: Dict) -> None:
    print("\n" + "="*80)
    print("FINAL AUDIT SUMMARY (TRUST & ROBUSTNESS)")
    print("="*80)
    
    # Extract Key Metrics
    eff_gain_100 = metrics["efficiency"][100]["eff_gain_pct"]
    
    # Corruption Gaps
    corr_gap_0 = next(r["gap"] for r in metrics["corruption"] if r["rate"] == 0.0)
    corr_gap_50 = next(r["gap"] for r in metrics["corruption"] if r["rate"] == 0.5)
    
    # Garbage
    garbage_delta = metrics["garbage"]["delta"]
    
    # Lalonde (Mean trust for adjusted vs unadjusted/bad claims)
    # We don't have labeled Lalonde in metrics easily, just list. 
    # But usually first claim is the valid adjustment.
    lalonde_trust = metrics["lalonde"][0]["trust"] if metrics["lalonde"] else 0.0
    
    print(f"{'Metric':<30} | {'Value':<10} | {'Status':<15}")
    print("-" * 60)
    
    # Efficiency
    status = "✅ PASS" if eff_gain_100 > 0.5 else "❌ FAIL"
    print(f"{'Efficiency Gain (N=100)':<30} | {eff_gain_100:>9.2f}% | {status}")
    
    # Corruption
    status = "✅ PASS" if corr_gap_0 > 0.3 else "⚠️ LOW"
    print(f"{'Corruption Gap @ 0%':<30} | {corr_gap_0:>10.4f} | {status}")
    
    status = "✅ PASS" if corr_gap_50 > 0.4 else "⚠️ LOW"
    print(f"{'Corruption Gap @ 50%':<30} | {corr_gap_50:>10.4f} | {status}")
    
    # Garbage
    status = "✅ PASS" if garbage_delta > 0.25 else "❌ FAIL"
    print(f"{'Garbage Delta':<30} | {garbage_delta:>10.4f} | {status}")
    
    # Lalonde
    status = "✅ PASS" if lalonde_trust > 0.1 else "⚠️ LOW"
    print(f"{'Lalonde Trust (Adjust)':<30} | {lalonde_trust:>10.4f} | {status}")
    
    print("="*80 + "\n")


def run_audit(cfg: AuditConfig) -> Tuple[str, Dict]:
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=cfg.n_vars).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=cfg.n_vars, device=device.type)

    started = time.time()

    metrics = {
        "efficiency": run_efficiency_sweep(model, gen, cfg, device),
        "corruption": run_corruption_sweep(model, gen, cfg, device),
        "scale": run_scale_sweep(model, cfg, device),
        "kernels": run_kernel_breakdown(model, cfg, device),
        "best_worst": run_best_worst_case(model, cfg, device),
        "garbage": run_garbage_audit(model, gen, cfg, device),
        "stability": run_stability_test(model, cfg, device),
        "index_sorting": run_index_sorting_check(gen, cfg),
        "lalonde": run_lalonde_audit(cfg.checkpoint, 'data/raw/lalonde.csv', n_vars=cfg.n_vars, max_claims=cfg.max_claims),
    }
    metrics["elapsed"] = time.time() - started
    
    report_md = generate_report_md(cfg, metrics)
    return report_md, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/model_adversarial_v2.pt")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    cfg = AuditConfig(checkpoint=args.checkpoint)
    if args.quick:
        cfg.efficiency_reps = 5
        cfg.corruption_reps = 5
        cfg.scale_reps = 5
        cfg.kernel_reps = 5
        cfg.stability_worlds = 2
        cfg.best_worst_reps = 3
        cfg.garbage_reps = 5

        cfg.garbage_reps = 5

    report, metrics = run_audit(cfg)
    print_terminal_summary(metrics)
    report_path = os.path.join("results", "audit_report.md")
    write_report(report_path, report)
    print(f"Audit report written to {report_path}")


if __name__ == "__main__":
    main()
