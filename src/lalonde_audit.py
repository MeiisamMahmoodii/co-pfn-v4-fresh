import csv
import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def run_lalonde_audit(model_path: str, data_path: str, n_vars: int = 20, max_claims: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 1. Load data
    rows = []
    with open(data_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError("Lalonde CSV appears empty.")
    
    # 2. Map variables to 20-dim input
    # Original: treat, age, educ, black, hispan, married, nodegree, re74, re75, re78
    # We'll map:
    # 0: treat (T)
    # 1: age
    # 2: educ
    # 3: black
    # 4: hispan
    # 5: married
    # 6: nodegree
    # 7: re74
    # 8: re75
    # 19: re78 (Y)
    
    X_raw = np.zeros((len(rows), n_vars))
    cols_to_map = ['treat', 'age', 'educ', 'black', 'hispan', 'married', 'nodegree', 're74', 're75']
    for i, col in enumerate(cols_to_map):
        X_raw[:, i] = [float(r[col]) for r in rows]
    X_raw[:, 19] = [float(r['re78']) for r in rows]
    
    # Normalize (PFNs expect normalized/standardized data)
    mean = X_raw.mean(axis=0)
    std = X_raw.std(axis=0)
    std[std == 0] = 1.0
    X_norm = (X_raw - mean) / std
    data_tensor = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(0).to(device)

    # 3. Define Claims
    print("--- LALONDE REAL-WORLD AUDIT ---")
    
    # Claim 1: True Adjustment (Pre-program earnings and demographics)
    # Adjustment set for T (index 0)
    # Indices: 1 (age), 2 (educ), 7 (re74), 8 (re75)
    true_adj_indices = [1, 2, 7, 8]
    claim_true = CausalClaim(ClaimType.ADJUSTMENT_SET, true_adj_indices, 0)
    
    # Claim 2: False Reverse Causality (re78 causes treat)
    claim_false = CausalClaim(ClaimType.DIRECT_CAUSE, [19], 0)
    
    # Claim 3: False Instrumental Variable (age is an IV for treat)
    # (Actually age is a confounder, not an IV)
    claim_iv_lie = CausalClaim(ClaimType.INSTRUMENTAL_VARIABLE, [1], 0)

    test_claims = [
        ("Adjustment (re74, re75, age, educ)", claim_true),
        ("Reverse Causality (re78 -> treat)", claim_false),
        ("False IV (age -> treat)", claim_iv_lie),
    ]
    claim_vec_size = claim_vector_size(n_vars)
    results = []

    for name, c_obj in test_claims:
        claim_tensor = torch.zeros(1, max_claims, claim_vec_size).to(device)
        claim_tensor[0, 0] = torch.tensor(c_obj.to_vector(n_vars))
        
        with torch.no_grad():
            ate_pred, trust_scores = model(data_tensor, claim_tensor, n_claims=torch.tensor([1]))
        
        print(f"\nEvaluating: {name}")
        print(f"  Trust Score: {trust_scores[0,0,0]:.4f}")
        print(f"  Estimated ATE: {ate_pred.item():.4f}")
        results.append({
            'claim': name,
            'trust': trust_scores[0,0,0].item(),
            'ate': ate_pred.item(),
        })
    return results

if __name__ == "__main__":
    run_lalonde_audit('checkpoints/model_adv_e20.pt', 'data/raw/lalonde.csv')
