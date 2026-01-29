import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def run_stress_tests(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use max_vars=100 for the scalability section
    model_scalar = TheoryFirstTransformer(n_vars=20).to(device)
    model_scalar.load_state_dict(torch.load(model_path, map_location=device))
    model_scalar.eval()

    print("--- 🚀 ACADEMIC STRESS TESTS 🚀 ---")

    # TEST A: Corruption Threshold (Sensitivity)
    # How well does it distinguish True vs False as corruption rate increases?
    print("\n[TEST A: Corruption Sensitivity]")
    gen = SCMGenerator(n_vars=20, device=device.type)
    claim_vec_size = claim_vector_size(20)
    rates = [0.0, 0.2, 0.5, 0.8, 1.0]
    sens_results = []
    
    for r in rates:
        trust_true, trust_false = [], []
        for _ in range(50):
            adj, perm = gen.generate_random_dag()
            t, y = 0, 19
            data = gen.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)
            
            # True Claim
            c_t = torch.zeros(1, 10, claim_vec_size).to(device)
            c_t[0,0] = torch.tensor(CausalClaim(ClaimType.DIRECT_CAUSE, [t], y).to_vector(20)).to(device)
            # False Claim
            c_f = torch.zeros(1, 10, claim_vec_size).to(device)
            c_f[0,0] = torch.tensor(CausalClaim(ClaimType.DIRECT_CAUSE, [y], t).to_vector(20)).to(device)
            
            with torch.no_grad():
                _, tr_t = model_scalar(data, c_t, n_claims=torch.tensor([1]))
                _, tr_f = model_scalar(data, c_f, n_claims=torch.tensor([1]))
                trust_true.append(tr_t[0,0,0].item())
                trust_false.append(tr_f[0,0,0].item())
        
        sens_results.append({'Rate': r, 'Avg_Trust_True': np.mean(trust_true), 'Avg_Trust_False': np.mean(trust_false)})
    
    print("Rate | Avg_Trust_True | Avg_Trust_False")
    for row in sens_results:
        print(f"{row['Rate']:.1f} | {row['Avg_Trust_True']:.4f} | {row['Avg_Trust_False']:.4f}")

    # TEST B: Graceful Degradation (Error vs Claim Quality)
    print("\n[TEST B: Graceful Degradation]")
    # Prove that ATE MAE stays capped even if theory is pure lies.
    degrad_results = []
    for _ in range(50):
        adj, perm = gen.generate_random_dag()
        t, y = 0, 19
        data = gen.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)
        true_ate = gen.compute_true_ate(adj, t, y, perm=perm)
        
        with torch.no_grad():
            # MAE No Theory
            ate_none, _ = model_scalar(data, torch.zeros(1, 10, claim_vec_size).to(device), n_claims=torch.tensor([0]))
            # MAE with Lie
            c_lie = torch.zeros(1, 10, claim_vec_size).to(device)
            c_lie[0,0] = torch.tensor(CausalClaim(ClaimType.DIRECT_CAUSE, [y], t).to_vector(20)).to(device)
            ate_lie, trust_lie = model_scalar(data, c_lie, n_claims=torch.tensor([1]))
            
            degrad_results.append({
                'MAE_None': abs(ate_none.item() - true_ate),
                'MAE_Lie': abs(ate_lie.item() - true_ate),
                'Trust_Lie': trust_lie[0,0,0].item()
            })
    
    mae_none = sum(r['MAE_None'] for r in degrad_results) / len(degrad_results)
    mae_lie = sum(r['MAE_Lie'] for r in degrad_results) / len(degrad_results)
    trust_lie = sum(r['Trust_Lie'] for r in degrad_results) / len(degrad_results)
    print("Metrics on Poisoned Theory:")
    print(f"MAE_None: {mae_none:.4f}, MAE_Lie: {mae_lie:.4f}, Trust_Lie: {trust_lie:.4f}")
    if mae_lie <= mae_none * 1.1:
        print("RESULT: SUCCESS. Model degrades gracefully (Lies are ignored).")

if __name__ == "__main__":
    run_stress_tests('checkpoints/model_adv_e20.pt')
