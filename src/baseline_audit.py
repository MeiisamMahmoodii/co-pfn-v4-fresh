import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size
import argparse

def audit_baseline_vs_claims(model_path: str, n_vars: int = 20, max_claims: int = 10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=n_vars, device=device.type)
    claim_vec_size = claim_vector_size(n_vars)
    
    sample_sizes = [20, 50, 100, 500]
    results = []

    for n_samples in sample_sizes:
        batch_errors_none = []
        batch_errors_true = []
        batch_trust_true = []
        batch_errors_false = []
        batch_trust_false = []
        batch_trust_garbage = []
        
        for _ in range(10): 
            adj, perm = gen.generate_random_dag()
            t_idx = np.random.randint(0, n_vars)
            y_idx = np.random.randint(0, n_vars)
            while y_idx == t_idx:
                y_idx = np.random.randint(0, n_vars)
            
            data_full = gen.sample_observational_data(adj, n_samples, perm=perm).unsqueeze(0).to(device)
            true_ate = gen.compute_true_ate(adj, t_idx, y_idx, perm=perm)
            
            # Scenario 1: Zero Claims (Baseline)
            c_none = torch.zeros(1, max_claims, claim_vec_size).to(device)
            with torch.no_grad():
                ate_none, _ = model(data_full, c_none, n_claims=torch.tensor([0]))
                batch_errors_none.append(abs(ate_none.item() - true_ate))

            # Scenario 2: One True Claim
            true_claim = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
            c_true = torch.zeros(1, max_claims, claim_vec_size).to(device)
            c_true[0, 0] = torch.tensor(true_claim.to_vector(n_vars))
            with torch.no_grad():
                ate_true, trust_true = model(data_full, c_true, n_claims=torch.tensor([1]))
                batch_errors_true.append(abs(ate_true.item() - true_ate))
                batch_trust_true.append(trust_true[0, 0, 0].item())

            # Scenario 3: One False Claim
            # Pick a random lie
            false_claim = CausalClaim(ClaimType.DIRECT_CAUSE, [y_idx], t_idx)
            c_false = torch.zeros(1, max_claims, claim_vec_size).to(device)
            c_false[0, 0] = torch.tensor(false_claim.to_vector(n_vars))
            with torch.no_grad():
                ate_false, trust_false = model(data_full, c_false, n_claims=torch.tensor([1]))
                batch_errors_false.append(abs(ate_false.item() - true_ate))
                batch_trust_false.append(trust_false[0, 0, 0].item())

            # Scenario 4: True Claim + Garbage Data
            garbage_data = torch.randn_like(data_full) * 5.0 
            with torch.no_grad():
                _, trust_garbage = model(garbage_data, c_true, n_claims=torch.tensor([1]))
                batch_trust_garbage.append(trust_garbage[0, 0, 0].item())

        results.append({
            'N': n_samples,
            'MAE None': np.mean(batch_errors_none),
            'MAE True': np.mean(batch_errors_true),
            'Trust True': np.mean(batch_trust_true),
            'MAE False': np.mean(batch_errors_false),
            'Trust False': np.mean(batch_trust_false),
            'Trust Garbage': np.mean(batch_trust_garbage)
        })

    print(f"\nAudit results for: {model_path}")
    headers = ["N", "MAE None", "MAE True", "Trust True", "MAE False", "Trust False", "Trust Garbage"]
    print(" | ".join(headers))
    for row in results:
        print(f"{row['N']} | {row['MAE None']:.4f} | {row['MAE True']:.4f} | {row['Trust True']:.4f} | "
              f"{row['MAE False']:.4f} | {row['Trust False']:.4f} | {row['Trust Garbage']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    audit_baseline_vs_claims(args.model)
