import csv
import torch
import numpy as np
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator, KernelType
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def run_kernel_breakdown(model_path: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=20).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # We need to manually force the generator to use specific kernels
    # Since SCMGenerator mixes them by default, we'll hack it or just filter (if exposed), 
    # but actual implementation allows passing kernel_pool? No.
    # Let's interact with the internal _sample_function or similar?
    # Actually, SCMGenerator has `kernel_pool`. We can override it!
    
    kernels = [KernelType.QUAD, KernelType.SIN, KernelType.CUBE, KernelType.GAUSS]
    claim_vec_size = claim_vector_size(20)
    results = []

    print("--- KERNEL BREAKDOWN ANALYSIS ---")
    
    for k in kernels:
        # Create a generator restricted to ONE kernel
        gen = SCMGenerator(n_vars=20, device=device.type, kernel_pool=[k])
        
        batch_mae_none = []
        batch_mae_true = []
        batch_trust = []
        
        for _ in range(50):
            adj, perm = gen.generate_random_dag()
            t, y = 0, 19
            # Retry to ensure path exists for meaningful ATE
            def check_path(adj, start, end):
                 q = [start]; visited = {start}
                 while q:
                     curr = q.pop(0)
                     if curr == end: return True
                     children = torch.where(adj[curr, :] != 0)[0].tolist()
                     for c in children:
                         if c not in visited: visited.add(c); q.append(c)
                 return False

            for _ in range(10): 
                if check_path(adj, t, y): break
                adj, perm = gen.generate_random_dag()
            
            data = gen.sample_observational_data(adj, 100, perm=perm).unsqueeze(0).to(device)
            # Sample function uses the kernel pool we hacked
            
            true_ate = gen.compute_true_ate(adj, t, y, perm=perm)
            
            # 1. No Claims
            with torch.no_grad():
                ate_none, _ = model(data, torch.zeros(1, 10, claim_vec_size).to(device), n_claims=torch.tensor([0]))
            
            # 2. True Claim
            c_true_obj = CausalClaim(ClaimType.DIRECT_CAUSE, [t], y)
            c_true = torch.zeros(1, 10, claim_vec_size).to(device)
            c_true[0, 0] = torch.tensor(c_true_obj.to_vector(20)).to(device)
            
            with torch.no_grad():
                ate_true, trust_out = model(data, c_true, n_claims=torch.tensor([1]))
                
            batch_mae_none.append(abs(ate_none.item() - true_ate))
            batch_mae_true.append(abs(ate_true.item() - true_ate))
            batch_trust.append(trust_out[0,0,0].item())

        results.append({
            'Kernel': k.name,
            'MAE_None': np.mean(batch_mae_none),
            'MAE_True': np.mean(batch_mae_true),
            'Trust': np.mean(batch_trust),
            'Gain': np.mean(batch_mae_none) - np.mean(batch_mae_true)
        })

    # Print table
    headers = ["Kernel", "MAE_None", "MAE_True", "Trust", "Gain"]
    print(" | ".join(headers))
    for row in results:
        print(f"{row['Kernel']} | {row['MAE_None']:.4f} | {row['MAE_True']:.4f} | {row['Trust']:.4f} | {row['Gain']:.4f}")

    with open('kernel_breakdown.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == "__main__":
    run_kernel_breakdown('checkpoints/model_adversarial_v2.pt')
