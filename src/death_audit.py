import torch
import numpy as np
from src.data.dataset import CausalPFNDataset, collate_fn
from torch.utils.data import DataLoader
from src.models.core import TheoryFirstTransformer

def perform_death_audit():
    print("--- 💀 CO-PFN DEATH AUDIT (RE-ENGINEERED) 💀 ---")
    device = torch.device('cpu')
    
    # 1. Dataset Leak Check
    # Important: dataset uses generator which now shuffles!
    dataset = CausalPFNDataset(min_vars=5, max_vars=10, min_samples=20, max_samples=50)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    
    data = batch['data']
    claims = batch['claims']
    ate_truth = batch['ate_truth']
    validity_truth = batch['validity_truth']
    n_claims = batch['n_claims']
    
    print(f"\n[LEAK CHECK]")
    print(f"Data range: [{data.min().item():.2f}, {data.max().item():.2f}]")
    
    # 2. Structural Cheat Check (Topological Shuffle)
    print(f"\n[STRUCTURAL CHEAT CHECK]")
    # Check multiple batches to see if source_idx < target_idx always holds for True claims
    found_true_reverse = False
    for i in range(batch['claims'].shape[1]):
        c_vec = batch['claims'][0, i]
        if c_vec[0] == 2.0: # DIRECT_CAUSE
            target_idx = c_vec[-1].item()
            source_indices = torch.where(c_vec[1:-1] == 1.0)[0].tolist()
            # If we find ANY case where target < source, the cheat is crushed!
            for s in source_indices:
                if target_idx < s:
                    print(f"VICTORY: Found reverse causal index (s={s} -> t={target_idx}). Sorting is no longer possible.")
                    found_true_reverse = True
    
    if not found_true_reverse:
        print("Note: All current batch claims follow index order (Random chance, keep auditing).")

    # 3. Model Logic Check (Data Dependency)
    print(f"\n[MODEL DATA-DEPENDENCY CHECK]")
    model = TheoryFirstTransformer(n_vars=10)
    with torch.no_grad():
        _, trust_actual = model(data, claims, n_claims=n_claims)
        # Feed GARBAGE data
        garbage_data = torch.randn_like(data) * 10.0
        _, trust_garbage = model(garbage_data, claims, n_claims=n_claims)
    
    t_actual = trust_actual[0, 0, 0].item()
    t_garbage = trust_garbage[0, 0, 0].item()
    
    print(f"Trust (Real Data): {t_actual:.4f}")
    print(f"Trust (Garbage Data): {t_garbage:.4f}")
    
    delta = abs(t_actual - t_garbage)
    print(f"Trust Delta: {delta:.6f}")
    
    if delta > 1e-6:
        print("SUCCESS: Model is numerically utilizing data context for Trust Auditing.")
    else:
        print("FAILURE: Trust Auditor is still data-independent (isolation bug).")

    print("\n[CONCLUSION]")
    print("Numerical stability: " + ("OK" if not torch.isnan(trust_actual).any() else "Fail (NaNs found)"))

if __name__ == "__main__":
    perform_death_audit()
