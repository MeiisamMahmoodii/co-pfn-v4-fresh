import torch
from src.models.core import TheoryFirstTransformer
from src.data.scm_generator import SCMGenerator
from src.data.claims import CausalClaim, ClaimType, claim_vector_size

def evaluate_causal_integrity(model_path: str, n_vars: int = 20, n_tasks: int = 10, n_samples: int = 500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TheoryFirstTransformer(n_vars=n_vars).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gen = SCMGenerator(n_vars=n_vars, device=device.type)
    claim_vec_size = claim_vector_size(n_vars)

    print("--- CAUSAL INTEGRITY TEST ---")

    results = []

    for i in range(n_tasks):
        adj, perm = gen.generate_random_dag()
        t_idx = 0
        y_idx = n_vars - 1
        data = gen.sample_observational_data(adj, n_samples, perm=perm).unsqueeze(0).to(device)
        true_ate = gen.compute_true_ate(adj, t_idx, y_idx, perm=perm)

        true_claim = CausalClaim(ClaimType.DIRECT_CAUSE, [t_idx], y_idx)
        claims_true = torch.zeros(1, 10, claim_vec_size).to(device)
        claims_true[0, 0] = torch.tensor(true_claim.to_vector(n_vars))

        with torch.no_grad():
            ate_true_input, trust_true_input = model(data, claims_true, n_claims=torch.tensor([1]))

        false_claim = CausalClaim(ClaimType.DIRECT_CAUSE, [y_idx], t_idx)
        claims_false = torch.zeros(1, 10, claim_vec_size).to(device)
        claims_false[0, 0] = torch.tensor(false_claim.to_vector(n_vars))

        with torch.no_grad():
            ate_false_input, trust_false_input = model(data, claims_false, n_claims=torch.tensor([1]))

        print(f"\nTask {i+1}: Ground Truth ATE: {true_ate:.4f}")
        print(f"  TRUE Claim -> Trust Score: {trust_true_input[0,0,0]:.4f}, Est ATE: {ate_true_input.item():.4f}")
        print(f"  FALSE Claim -> Trust Score: {trust_false_input[0,0,0]:.4f}, Est ATE: {ate_false_input.item():.4f}")

        results.append({
            'true_trust': trust_true_input[0,0,0].item(),
            'false_trust': trust_false_input[0,0,0].item()
        })

    avg_true = sum(r['true_trust'] for r in results) / len(results)
    avg_false = sum(r['false_trust'] for r in results) / len(results)

    print(f"\nFINAL PERFORMANCE summary:")
    print(f"Average Trust in Correct claims: {avg_true:.4f}")
    print(f"Average Trust in Lying claims:   {avg_false:.4f}")

    if avg_true > avg_false + 0.2:
        print("\nSUCCESS: Model is discriminating between true and false claims!")
    else:
        print("\nWARNING: Model trust scores are too similar. It may be hallucinating or ignoring claims.")

if __name__ == "__main__":
    evaluate_causal_integrity('checkpoints/model_adv_e100.pt')
