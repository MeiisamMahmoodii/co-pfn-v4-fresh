import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def enemy_audit(model_path: str, n_vars: int = 20):
    print(f"--- 👿 CO-PFN ENEMY AUDIT (SESSION V6) 👿 ---")
    print(f"Model: {model_path}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=n_vars, device=device.type)
    claim_vec_size = claim_vector_size(n_vars)

    # TEST 1: The Confounder Trap (Correlation != Causation)
    print("\n[TEST 1: Confounder Trap]")
    adj = torch.zeros(n_vars, n_vars).to(device)
    z_idx, x_idx, y_idx = 10, 0, 19
    adj[x_idx, z_idx] = 1.0 
    adj[y_idx, z_idx] = 1.0 
    perm = np.arange(n_vars) 
    
    data = gen.sample_observational_data(adj, 512, perm=perm).unsqueeze(0).to(device)
    true_ate = gen.compute_true_ate(adj, x_idx, y_idx, perm=perm)
    print(f"Ground Truth ATE: {true_ate:.4f} (Should be ~0)")

    claim_lie = CausalClaim(ClaimType.DIRECT_CAUSE, [x_idx], y_idx)
    c_tensor = torch.zeros(1, 10, claim_vec_size).to(device)
    c_tensor[0, 0] = torch.tensor(claim_lie.to_vector(n_vars)).to(device)
    
    with torch.no_grad():
        ate_pred, trust = model(data, c_tensor, n_claims=torch.tensor([1]).to(device))
    
    print(f"Claim: X -> Y | Trust: {trust[0,0,0]:.4f} | Est ATE: {ate_pred.item():.4f}")
    if trust[0,0,0] < 0.25:
        print("RESULT: SUCCESS. Model rejected the correlational trap.")
    else:
        print("RESULT: FAILURE. Model is confusing correlation with causation.")

    # TEST 2: Data Efficiency Check
    print("\n[TEST 2: Data Efficiency Check]")
    adj_rand, perm_rand = gen.generate_random_dag()
    t_idx = np.random.randint(0, n_vars)
    o_idx = np.random.randint(0, n_vars)
    while o_idx == t_idx: o_idx = np.random.randint(0, n_vars)
    
    true_ate_rand = gen.compute_true_ate(adj_rand, t_idx, o_idx, perm=perm_rand)
    
    data_small = gen.sample_observational_data(adj_rand, 20, perm=perm_rand).unsqueeze(0).to(device)
    data_large = gen.sample_observational_data(adj_rand, 500, perm=perm_rand).unsqueeze(0).to(device)
    
    claim_true = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], o_idx)
    c_true = torch.zeros(1, 10, claim_vec_size).to(device)
    c_true[0, 0] = torch.tensor(claim_true.to_vector(n_vars)).to(device)
    c_none = torch.zeros(1, 10, claim_vec_size).to(device)
    
    with torch.no_grad():
        ate_small_w_claim, _ = model(data_small, c_true, n_claims=torch.tensor([1]).to(device))
        ate_large_no_claim, _ = model(data_large, c_none, n_claims=torch.tensor([0]).to(device))
        
    error_small = abs(ate_small_w_claim.item() - true_ate_rand)
    error_large = abs(ate_large_no_claim.item() - true_ate_rand)
    
    print(f"MAE (N=20 + Claim):  {error_small:.4f}")
    print(f"MAE (N=500 + NoClaim): {error_large:.4f}")
    
    if error_small < error_large * 2.0: 
        print("RESULT: SUCCESS. Model leverages theory to overcome small data.")
    else:
        print("RESULT: FAILURE. Theory is being ignored or is unhelpful.")

    # TEST 3: The Garbage Audit (The Ultimate Enemy Test)
    print("\n[TEST 3: The Garbage Audit]")
    garbage_data = torch.randn(1, 512, n_vars).to(device)
    with torch.no_grad():
        _, trust_garbage = model(garbage_data, c_true, n_claims=torch.tensor([1]).to(device))
    
    print(f"Trust in 'True' claim on Shuffled/Garbage Data: {trust_garbage[0,0,0]:.4f}")
    if trust_garbage[0,0,0] < 0.2:
        print("RESULT: SUCCESS. Model requires evidence for trust.")
    else:
        print("RESULT: FAILURE. Model is trusting syntax instead of evidence.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    enemy_audit(args.model)
