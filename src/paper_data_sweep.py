import torch
import numpy as np
import csv
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def generate_exhaustive_paper_data(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=20).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen_mid = SCMGenerator(n_vars=20, device=device.type)
    claim_vec_size_20 = claim_vector_size(20)
    
    sample_sizes = [10, 20, 50, 100, 500, 1000]
    n_repetitions = 30 # Increased for statistical significance
    
    results = []
    
    print("Starting Comprehensive Empirical Sweep...")
    
    # Experiment 1: Data Efficiency (N-Sweep)
    for N in sample_sizes:
        for i in range(n_repetitions):
            adj, perm = gen_mid.generate_random_dag()
            t, y = 0, 19
            data = gen_mid.sample_observational_data(adj, N, perm=perm).unsqueeze(0).to(device)
            true_ate = gen_mid.compute_true_ate(adj, t, y, perm=perm)
            
            # Pad/Prepare tensors
            c_none = torch.zeros(1, 10, claim_vec_size_20).to(device)
            c_true_obj = CausalClaim(ClaimType.DIRECT_CAUSE, [t], y)
            c_true = torch.zeros(1, 10, claim_vec_size_20).to(device)
            c_true[0, 0] = torch.tensor(c_true_obj.to_vector(20)).to(device)
            
            with torch.no_grad():
                ate_none, _ = model(data, c_none, n_claims=torch.tensor([0]))
                ate_true, trust_true = model(data, c_true, n_claims=torch.tensor([1]))
            
            results.append({
                'Experiment': 'Efficiency',
                'N': N,
                'Vars': 20,
                'Iter': i,
                'MAE_None': abs(ate_none.item() - true_ate),
                'MAE_True': abs(ate_true.item() - true_ate),
                'Trust_True': trust_true[0, 0, 0].item(),
                'True_ATE': true_ate
            })

    # Experiment 2: Variable Scalability (Vars-Sweep)
    # We'll use N=100 and vary complexity
    for V in [10, 15, 20]:
        gen_v = SCMGenerator(n_vars=V, device=device.type)
        c_vec_size = claim_vec_size_20
        for i in range(n_repetitions):
            adj, perm = gen_v.generate_random_dag()
            t, y = 0, V-1
            data_raw = gen_v.sample_observational_data(adj, 100, perm=perm)
            # Pad to 20 vars for model
            data = torch.zeros(1, 100, 20).to(device)
            data[0, :, :V] = data_raw
            
            true_ate = gen_v.compute_true_ate(adj, t, y, perm=perm)
            
            c_true_obj = CausalClaim(ClaimType.DIRECT_CAUSE, [t], y)
            c_true = torch.zeros(1, 10, c_vec_size).to(device)
            c_true[0, 0] = torch.tensor(c_true_obj.to_vector(20)).to(device)
            
            with torch.no_grad():
                _, trust = model(data, c_true, n_claims=torch.tensor([1]))
            
            results.append({
                'Experiment': 'Scalability',
                'N': 100,
                'Vars': V,
                'Iter': i,
                'Trust_True': trust[0,0,0].item()
            })

    headers = sorted(results[0].keys())
    with open('academic_raw_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print("Academic data sweep complete. results saved to academic_raw_data.csv")
    
    # Print summary table for my own planning
    eff = [r for r in results if r['Experiment'] == 'Efficiency']
    by_n = {}
    for r in eff:
        n = r['N']
        by_n.setdefault(n, {'MAE_None': [], 'MAE_True': [], 'Trust_True': []})
        by_n[n]['MAE_None'].append(r['MAE_None'])
        by_n[n]['MAE_True'].append(r['MAE_True'])
        by_n[n]['Trust_True'].append(r['Trust_True'])
    print("\nEfficiency Summary:")
    print("N | MAE_None | MAE_True | Trust_True")
    for n in sorted(by_n):
        vals = by_n[n]
        mean_none = sum(vals['MAE_None']) / len(vals['MAE_None'])
        mean_true = sum(vals['MAE_True']) / len(vals['MAE_True'])
        mean_trust = sum(vals['Trust_True']) / len(vals['Trust_True'])
        print(f"{n} | {mean_none:.4f} | {mean_true:.4f} | {mean_trust:.4f}")

if __name__ == "__main__":
    generate_exhaustive_paper_data('checkpoints/model_adv_e20.pt')
