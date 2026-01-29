import torch
from torch.utils.data import IterableDataset, DataLoader
from src.data.scm_generator import SCMGenerator
import numpy as np

class CausalPFNDataset(IterableDataset):
    """Infinite dataset of causal tasks with adversarial Null Injection."""
    def __init__(self, min_vars: int = 5, max_vars: int = 20, min_samples: int = 20, max_samples: int = 1000, corruption_rate: float = 0.3, max_claims: int = 10, null_rate: float = 0.2, enforce_edge_rate: float = 0.5):
        super().__init__()
        self.min_vars = min_vars
        self.max_vars = max_vars
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.corruption_rate = corruption_rate
        self.max_claims = max_claims
        self.null_rate = null_rate
        self.enforce_edge_rate = enforce_edge_rate  # Rate at which to enforce direct T->Y edge

    def __iter__(self):
        while True:
            n_vars = np.random.randint(self.min_vars, self.max_vars + 1)
            n_samples = np.random.randint(self.min_samples, self.max_samples + 1)
            
            generator = SCMGenerator(n_vars=n_vars)
            t_idx = np.random.randint(0, n_vars)
            y_idx = np.random.randint(0, n_vars)
            while y_idx == t_idx:
                y_idx = np.random.randint(0, n_vars)
            
            # 1. Decision: Is this a Null Scenario?
            is_null = np.random.rand() < self.null_rate
            
            if is_null:
                # Null Case: Real syntax, Zero Evidence
                adj, perm = generator.generate_random_dag()
                data_raw = torch.randn(n_samples, n_vars)
                ate = 0.0 # No causal relationship in random noise
                # Generate claims that might look true syntactically but are false due to noise
                claims_objs, _ = generator.generate_claims(adj, t_idx, y_idx, corruption_rate=0.0)
                validity = [False] * len(claims_objs) # ALL claims are false if data is noise
            else:
                # Standard Case - decide whether to enforce direct edge
                enforce_edge = np.random.rand() < self.enforce_edge_rate
                adj, perm = generator.generate_random_dag(
                    enforce_direct_edge=enforce_edge,
                    treatment_idx=t_idx,
                    outcome_idx=y_idx
                )
                data_raw = generator.sample_observational_data(adj, n_samples, perm=perm)
                ate = generator.compute_true_ate(adj, t_idx, y_idx, perm=perm)
                claims_objs, validity = generator.generate_claims(adj, t_idx, y_idx, self.corruption_rate)
            
            # Zero-pad data samples to max_samples AND vars to max_vars
            data = torch.zeros(self.max_samples, self.max_vars)
            data[:n_samples, :n_vars] = data_raw
            
            # Pad claims to max_claims
            # Vector size is now 1 (type) + n_vars (vars) + n_vars (target one-hot) = 2*n_vars + 1
            claim_vec_size = 2 * self.max_vars + 1
            claim_vecs = [torch.tensor(c.to_vector(self.max_vars)) for c in claims_objs]
            k_actual = len(claim_vecs)
            
            claim_tensor = torch.zeros(self.max_claims, claim_vec_size)
            validity_tensor = torch.zeros(self.max_claims)
            
            if k_actual > 0:
                claim_tensor[:k_actual] = torch.stack(claim_vecs)
                validity_tensor[:k_actual] = torch.tensor(validity, dtype=torch.float32)

            yield {
                'data': data,
                'claims': claim_tensor,
                'ate_truth': torch.tensor([ate], dtype=torch.float32),
                'validity_truth': validity_tensor,
                'n_claims': torch.tensor(k_actual, dtype=torch.long),
                'n_samples': torch.tensor(n_samples, dtype=torch.long)
            }

def collate_fn(batch):
    data = torch.stack([item['data'] for item in batch])
    claims = torch.stack([item['claims'] for item in batch])
    ate_truth = torch.stack([item['ate_truth'] for item in batch])
    validity_truth = torch.stack([item['validity_truth'] for item in batch])
    n_claims = torch.stack([item['n_claims'] for item in batch])
    n_samples = torch.stack([item['n_samples'] for item in batch])
    
    return {
        'data': data,
        'claims': claims,
        'ate_truth': ate_truth,
        'validity_truth': validity_truth,
        'n_claims': n_claims,
        'n_samples': n_samples
    }
