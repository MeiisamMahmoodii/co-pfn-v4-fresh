import torch
import numpy as np
import csv
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def generate_paper_data(model_path: str, n_vars: int = 20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=n_vars, device=device.type)
    
    sample_sizes = [20, 50, 100, 500, 1000]
    claim_vec_size = claim_vector_size(n_vars)
    all_results = []

    print(f"Sweep starting on {model_path}...")

    for N in sample_sizes:
        batch_mae_none = []
        batch_mae_true = []
        batch_trust_true = []
        batch_mae_false = []
        batch_trust_false = []
        
        for _ in range(20): # More iterations for paper stability
            adj, perm = gen.generate_random_dag()
            t_idx = np.random.randint(0, n_vars)
            y_idx = np.random.randint(0, n_vars)
            while y_idx == t_idx: y_idx = np.random.randint(0, n_vars)
            
            data = gen.sample_observational_data(adj, N, perm=perm).unsqueeze(0).to(device)
            true_ate = gen.compute_true_ate(adj, t_idx, y_idx, perm=perm)
            
            # 1. No Claims
            with torch.no_grad():
                ate_none, _ = model(data, torch.zeros(1, 10, claim_vec_size).to(device), n_claims=torch.tensor([0]))
                batch_mae_none.append(abs(ate_none.item() - true_ate))
            
            # 2. True Claim
            c_true_obj = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
            c_true = torch.zeros(1, 10, claim_vec_size).to(device)
            c_true[0, 0] = torch.tensor(c_true_obj.to_vector(n_vars)).to(device)
            with torch.no_grad():
                ate_true, trust_true = model(data, c_true, n_claims=torch.tensor([1]))
                batch_mae_true.append(abs(ate_true.item() - true_ate))
                batch_trust_true.append(trust_true[0, 0, 0].item())
                
            # 3. False Claim
            c_false_obj = CausalClaim(ClaimType.DIRECT_CAUSE, [y_idx], t_idx)
            c_false = torch.zeros(1, 10, claim_vec_size).to(device)
            c_false[0, 0] = torch.tensor(c_false_obj.to_vector(n_vars)).to(device)
            with torch.no_grad():
                ate_false, trust_false = model(data, c_false, n_claims=torch.tensor([1]))
                batch_mae_false.append(abs(ate_false.item() - true_ate))
                batch_trust_false.append(trust_false[0, 0, 0].item())

        all_results.append({
            'N': N,
            'MAE_None': np.mean(batch_mae_none),
            'MAE_True': np.mean(batch_mae_true),
            'Trust_True': np.mean(batch_trust_true),
            'MAE_False': np.mean(batch_mae_false),
            'Trust_False': np.mean(batch_trust_false)
        })

    print("\n--- FINAL PAPER DATA TABLES ---")
    headers = ['N', 'MAE_None', 'MAE_True', 'Trust_True', 'MAE_False', 'Trust_False']
    print(" | ".join(headers))
    for row in all_results:
        print(f"{row['N']} | {row['MAE_None']:.4f} | {row['MAE_True']:.4f} | {row['Trust_True']:.4f} | "
              f"{row['MAE_False']:.4f} | {row['Trust_False']:.4f}")
    with open('paper_results_v6.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

if __name__ == "__main__":
    generate_paper_data('checkpoints/model_adv_e20.pt')
