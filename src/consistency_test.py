import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def run_consistency_test(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_vars = 20
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=n_vars, device=device.type)
    claim_vec_size = claim_vector_size(n_vars)
    max_claims = 10
    n_worlds = 5
    
    print(f"STRESS TEST: {n_worlds} Worlds Consistency Strategy")
    print("-" * 60)

    all_variances = []
    
    for w in range(n_worlds):
        # 1. Create one fixed world
        adj, perm = gen.generate_random_dag()
        t_idx = 0
        y_idx = n_vars - 1 # Use fixed ends for simplicity, or random if guaranteed path
        # Retry to find a valid path
        def check_path(adj, start, end):
             # Simple BFS
             q = [start]
             visited = {start}
             while q:
                 curr = q.pop(0)
                 if curr == end: return True
                 # adj is [N, N], children are where adj[curr, :] != 0
                 children = torch.where(adj[curr, :] != 0)[0].tolist()
                 for c in children:
                     if c not in visited:
                         visited.add(c)
                         q.append(c)
             return False

        for _ in range(20):
             if check_path(adj, t_idx, y_idx): break
             adj, perm = gen.generate_random_dag()
             
        data = gen.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)
        true_ate = gen.compute_true_ate(adj, t_idx, y_idx, perm=perm)
        
        print(f"\nWORLD {w+1} | Ground Truth ATE: {true_ate:.4f}")

        scenarios = []
        # Scenario A: Baseline
        scenarios.append(("Baseline", []))
        # Scenario B: True Claims
        true_direct = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
        scenarios.append(("True Only", [true_direct]))
        # Scenario C: Lie
        lie_reverse = CausalClaim(ClaimType.DIRECT_CAUSE, [y_idx], t_idx)
        scenarios.append(("Mixed (True+Lie)", [true_direct, lie_reverse]))
        # Scenario D: Hostile
        scenarios.append(("Hostile", [lie_reverse]))

        ates = []
        for name, claim_list in scenarios:
            k = len(claim_list)
            claim_tensor = torch.zeros(1, max_claims, claim_vec_size).to(device)
            # Mask logic
            n_claims_t = torch.tensor([k]).to(device)
            if k == 0: n_claims_t = torch.tensor([0]).to(device) # Transformer needs care

            for i, c in enumerate(claim_list):
                claim_tensor[0, i] = torch.tensor(c.to_vector(n_vars))
            
            with torch.no_grad():
                ate_pred, trust = model(data, claim_tensor, n_claims=n_claims_t)
                
            ates.append(ate_pred.item())
            tr_val = trust[0,0,0].item() if k > 0 else 0.0
            print(f"  {name:15}: ATE {ate_pred.item():.4f} (Trust: {tr_val:.3f})")

        all_variances.append(np.var(ates))

    print("-" * 60)
    print(f"Mean ATE Variance across worlds: {np.mean(all_variances):.6f}")

if __name__ == "__main__":
    run_consistency_test('checkpoints/model_adversarial_v2.pt')
