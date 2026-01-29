import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.claims import CausalClaim, ClaimType, claim_vector_size
from src.data.scm_generator import SCMGenerator

def ghost_claim_test(model_path: str, n_vars: int = 20):
    print("--- 👻 CO-PFN GHOST CLAIM TEST 👻 ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=n_vars, device=device.type)
    adj, perm = gen.generate_random_dag()
    data = gen.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)

    # Claim about a variable index that is OUT OF BOUNDS (e.g., 25)
    # Note: Our ClaimEncoder currently handles up to n_vars.
    # If we pass 25, it might crash or behave weirdly if not caught.
    # However, to_vector caps at max_vars. 
    # Let's try to claim a variable that exists but is IRRELEVANT (Noise node).
    
    # 1. True Claim Baseline
    true_c = CausalClaim(ClaimType.DIRECT_CAUSE, [0], 19)
    # 2. Ghost Claim: Claim index 15 causes 19, but 15 is a disconnected noise node
    # (We can force adj[19, 15] to be 0)
    ghost_c = CausalClaim(ClaimType.DIRECT_CAUSE, [15], 19)
    
    claims = [("Valid", true_c), ("Ghost (Irrelevant)", ghost_c)]
    
    for name, c_obj in claims:
        claim_vec_size = claim_vector_size(n_vars)
        c_tensor = torch.zeros(1, 10, claim_vec_size).to(device)
        c_tensor[0, 0] = torch.tensor(c_obj.to_vector(n_vars))
        
        with torch.no_grad():
            _, trust = model(data, c_tensor, n_claims=torch.tensor([1]))
        
        print(f"[{name}] Trust Score: {trust[0,0,0]:.4f}")

if __name__ == "__main__":
    ghost_claim_test('checkpoints/model_adv_e20.pt')
