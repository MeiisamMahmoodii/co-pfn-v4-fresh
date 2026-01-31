import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from src.data.claims import CausalClaim, ClaimType, CausalWorldMetadata

class KernelType(Enum):
    LINEAR = "LINEAR"
    SIN = "SIN"
    QUAD = "QUAD"
    CUBE = "CUBE"
    GAUSS = "GAUSS"
    RELU = "RELU"
    MIX = "MIX"

class SCMGenerator:
    """Vectorized generator for Structural Causal Models with Shuffled Topology."""
    
    def __init__(self, n_vars: int, device: str = 'cpu', non_linear: bool = True, kernel_pool: Optional[List[KernelType]] = None):
        self.n_vars = n_vars
        self.device = device
        self.non_linear = non_linear
        if kernel_pool is not None:
            self.kernel_pool = kernel_pool
        else:
            self.kernel_pool = [
                KernelType.SIN,
                KernelType.QUAD,
                KernelType.CUBE,
                KernelType.GAUSS,
                KernelType.RELU,
                KernelType.MIX,
            ]

    def generate_random_dag(self, enforce_direct_edge: bool = False, treatment_idx: int = None, outcome_idx: int = None) -> Tuple[torch.Tensor, np.ndarray]:
        """Generate a random DAG with optional enforcement of a direct T->Y edge."""
        adj_base = torch.tril(torch.randn(self.n_vars, self.n_vars), diagonal=-1)
        mask = torch.rand(self.n_vars, self.n_vars) > 0.5
        adj_base = adj_base * mask.float()
        
        perm = np.random.permutation(self.n_vars)
        adj_shuffled = torch.zeros_like(adj_base)
        for i in range(self.n_vars):
            for j in range(self.n_vars):
                adj_shuffled[perm[i], perm[j]] = adj_base[i, j]
        
        # Optionally enforce a direct edge from treatment to outcome with MODERATE effect
        if enforce_direct_edge and treatment_idx is not None and outcome_idx is not None:
            # Moderate effect size: uniform in [0.5, 1.5] with random sign
            # Reduced from [1.5, 3.0] to prevent trust saturation
            effect_magnitude = 0.5 + 1.0 * np.random.rand()
            effect_sign = np.random.choice([-1, 1])
            adj_shuffled[outcome_idx, treatment_idx] = effect_sign * effect_magnitude
                
        return adj_shuffled.to(self.device), perm

    def sample_observational_data(self, adj: torch.Tensor, n_samples: int, perm: np.ndarray, intervention: Optional[Dict[int, float]] = None) -> torch.Tensor:
        data = torch.zeros(n_samples, self.n_vars).to(self.device)
        noise = torch.zeros(n_samples, self.n_vars).to(self.device)
        
        # DISTRIBUTIONAL ROBUSTNESS: Sample noise from diverse distributions per variable
        # This prevents the model from overfitting to Gaussian assumptions (solving Lalonde OOD)
        for i in range(self.n_vars):
            dist_type = np.random.choice(['gauss', 'uniform', 'exp', 'gamma', 'beta'])
            
            if dist_type == 'gauss':
                noise[:, i] = torch.randn(n_samples)
            elif dist_type == 'uniform':
                # Uniform[-sqrt(3), sqrt(3)] has std=1, mean=0
                noise[:, i] = (torch.rand(n_samples) - 0.5) * 3.46 
            elif dist_type == 'exp':
                # Exp(1) has mean=1, std=1. Shift to mean=0.
                noise[:, i] = torch.empty(n_samples).exponential_(1.0) - 1.0
            elif dist_type == 'gamma':
                # Gamma(2.0, 2.0) has mean=1, std=0.7. Normalize to mean=0, std=1
                g = torch.distributions.Gamma(2.0, 2.0).sample((n_samples,))
                noise[:, i] = (g - 1.0) * 1.41
            elif dist_type == 'beta':
                # Beta(0.5, 0.5) is bimodal (Archives). Mean=0.5, Var=0.125 -> Std=0.35.
                b = torch.distributions.Beta(0.5, 0.5).sample((n_samples,))
                noise[:, i] = (b - 0.5) * 2.82
        
        noise = noise.to(self.device)
        
        for base_idx in range(self.n_vars):
            actual_idx = perm[base_idx]
            
            if intervention is not None and actual_idx in intervention:
                data[:, actual_idx] = intervention[actual_idx]
                continue

            parents_contribution = torch.matmul(data, adj[actual_idx, :].unsqueeze(1)).squeeze()
            
            if base_idx == 0:
                data[:, actual_idx] = noise[:, actual_idx]
            else:
                if self.non_linear:
                    # Adversarial Kernel Pool
                    k = self.kernel_pool[np.random.randint(0, len(self.kernel_pool))]
                    if k == KernelType.SIN:
                        f_x = torch.sin(parents_contribution)
                    elif k == KernelType.QUAD:
                        f_x = 0.5 * parents_contribution**2 * torch.sign(parents_contribution)
                    elif k == KernelType.CUBE:
                        f_x = 0.2 * parents_contribution**3
                    elif k == KernelType.GAUSS:
                        f_x = torch.exp(-(parents_contribution**2))
                    elif k == KernelType.RELU:
                        f_x = torch.relu(parents_contribution)
                    elif k == KernelType.MIX:
                        f_x = torch.sin(parents_contribution) + torch.tanh(parents_contribution)
                    elif k == KernelType.LINEAR:
                        f_x = parents_contribution
                    else:
                        f_x = parents_contribution
                else:
                    f_x = parents_contribution
                
                f_x = torch.clamp(f_x, -5, 5)
                data[:, actual_idx] = f_x + noise[:, actual_idx]
        
        return data

    def compute_true_ate(self, adj: torch.Tensor, treatment_idx: int, outcome_idx: int, perm: np.ndarray) -> float:
        n_mc = 10000
        d1 = self.sample_observational_data(adj, n_mc, perm=perm, intervention={treatment_idx: 1.0})
        d0 = self.sample_observational_data(adj, n_mc, perm=perm, intervention={treatment_idx: 0.0})
        return d1[:, outcome_idx].mean().item() - d0[:, outcome_idx].mean().item()

    def _has_causal_path(self, adj: torch.Tensor, source: int, target: int) -> bool:
        """Check if there's a directed path from source to target in the DAG."""
        visited = set()
        stack = [source]
        while stack:
            node = stack.pop()
            if node == target:
                return True
            if node in visited:
                continue
            visited.add(node)
            # Find children: nodes where adj[child, node] != 0 (node is parent of child)
            children = torch.where(adj[:, node] != 0)[0].tolist()
            stack.extend(children)
        return False

    def _get_parents(self, adj: torch.Tensor, node: int) -> List[int]:
        """Get direct parents of a node."""
        return torch.where(adj[node, :] != 0)[0].tolist()

    def _get_children(self, adj: torch.Tensor, node: int) -> List[int]:
        """Get direct children of a node."""
        return torch.where(adj[:, node] != 0)[0].tolist()

    def _is_valid_adjustment_set(self, adj: torch.Tensor, treatment: int, outcome: int, adjust_vars: List[int]) -> bool:
        """
        Strict backdoor criterion check:
        A valid adjustment set Z must:
        1. Not include treatment or outcome
        2. Not include any descendant of treatment (would block causal path or introduce collider bias)
        3. Block all backdoor paths from treatment to outcome
        
        For a single variable to be valid, it should be:
        - A parent of treatment (blocks backdoor), OR
        - A parent of a common ancestor of T and Y
        AND not be a descendant of treatment
        """
        if treatment in adjust_vars or outcome in adjust_vars:
            return False
        
        for v in adjust_vars:
            # Check v is not a descendant of treatment
            if self._has_causal_path(adj, treatment, v):
                return False
            
            # Check v is not on the causal path from T to Y (mediator)
            if self._has_causal_path(adj, treatment, v) and self._has_causal_path(adj, v, outcome):
                return False
            
            # v should be a parent of treatment OR a parent of outcome's ancestors
            # (i.e., it should actually help block backdoor paths)
            is_parent_of_treatment = adj[treatment, v] != 0
            is_ancestor_of_treatment = self._has_causal_path(adj, v, treatment)
            
            # Must be relevant to the backdoor path
            if not (is_parent_of_treatment or is_ancestor_of_treatment):
                # Check if it's a common cause (parent of both T and Y or their ancestors)
                is_ancestor_of_outcome = self._has_causal_path(adj, v, outcome)
                if not (is_ancestor_of_treatment and is_ancestor_of_outcome):
                    return False  # Irrelevant variable - not a valid adjustment
        
        return True

    def generate_claims(self, adj: torch.Tensor, treatment_idx: int, outcome_idx: int, corruption_rate: float = 0.3) -> Tuple[List[CausalClaim], List[bool]]:
        """
        Generate claims with ENFORCED ground truth and HARD NEGATIVES.
        - True claims are verified against the actual causal graph
        - False claims are guaranteed to be false BUT plausible (correlate with Y)
        - DIRECT_TRUE claims are prioritized and frequent
        """
        claims, validity = [], []
        num_claims = np.random.randint(3, 7)  # More claims for better pairwise learning
        
        # Precompute graph properties
        has_direct_edge = adj[outcome_idx, treatment_idx] != 0
        has_causal_path = self._has_causal_path(adj, treatment_idx, outcome_idx)
        treatment_parents = self._get_parents(adj, treatment_idx)
        treatment_children = self._get_children(adj, treatment_idx)
        outcome_parents = self._get_parents(adj, outcome_idx)
        outcome_children = self._get_children(adj, outcome_idx)
        
        # Build pool of TRUE claims (verified)
        # PRIORITIZE DIRECT_TRUE by adding multiple copies
        true_claims_pool = []
        
        # TRUE: Direct cause (only if edge actually exists) - ADD MULTIPLE COPIES FOR PRIORITY
        if has_direct_edge:
            # Add 4 copies to make DIRECT_TRUE claims very frequent
            for _ in range(4):
                true_claims_pool.append(('DIRECT_TRUE', treatment_idx, outcome_idx))
        
        # TRUE: Valid adjustment sets (parents of treatment with strict backdoor check)
        if treatment_parents:
            for p in treatment_parents:
                if self._is_valid_adjustment_set(adj, treatment_idx, outcome_idx, [p]):
                    true_claims_pool.append(('ADJUST_TRUE', p, treatment_idx))
        
        # TRUE: Post-treatment variable (actual children of treatment)
        for c in treatment_children:
            true_claims_pool.append(('POST_TRUE', c, treatment_idx))
        
        # Build pool of FALSE claims - TRULY FALSE (no causal path to Y)
        false_claims_pool = []
        
        # Find all nodes with NO causal path to outcome Y
        # These are truly disconnected from Y - the hardest test
        disconnected_from_y = []
        for v in range(self.n_vars):
            if v != outcome_idx and not self._has_causal_path(adj, v, outcome_idx):
                # v has no causal path to Y - truly unrelated
                disconnected_from_y.append(v)
        
        # FALSE 1: Disconnected nodes claimed as direct causes of Y (truly false)
        for v in disconnected_from_y:
            if v != treatment_idx:
                # Add multiple copies for nodes that are truly disconnected
                for _ in range(2):
                    false_claims_pool.append(('DIRECT_FALSE', v, outcome_idx))
        
        # FALSE 2: Reverse causation Y -> T (only if no edge exists)
        if adj[treatment_idx, outcome_idx] == 0:
            false_claims_pool.append(('DIRECT_FALSE', outcome_idx, treatment_idx))
        
        # FALSE 3: Random non-existent edges between disconnected pairs
        for _ in range(5):
            rand_src = np.random.randint(0, self.n_vars)
            rand_tgt = np.random.randint(0, self.n_vars)
            if rand_src != rand_tgt:
                # Check no direct edge AND no causal path
                if adj[rand_tgt, rand_src] == 0 and not self._has_causal_path(adj, rand_src, rand_tgt):
                    false_claims_pool.append(('DIRECT_FALSE', rand_src, rand_tgt))
        
        # FALSE 4: Invalid adjustment - descendant of T (opens backdoor path)
        for c in treatment_children:
            false_claims_pool.append(('ADJUST_FALSE', c, treatment_idx))
        
        # FALSE 5: Post-treatment on non-children of T
        non_children = [i for i in range(self.n_vars) if i not in treatment_children and i != treatment_idx]
        if non_children:
            for nc in np.random.choice(non_children, min(3, len(non_children)), replace=False):
                false_claims_pool.append(('POST_FALSE', nc, treatment_idx))
        
        # FALSE: Post-treatment on non-children
        non_children = [i for i in range(self.n_vars) if i not in treatment_children and i != treatment_idx]
        if non_children:
            for nc in np.random.choice(non_children, min(2, len(non_children)), replace=False):
                false_claims_pool.append(('POST_FALSE', nc, treatment_idx))
        
        # Shuffle pools
        np.random.shuffle(true_claims_pool)
        np.random.shuffle(false_claims_pool)
        
        # ALWAYS include at least one TRUE and one FALSE for pairwise learning
        # First, add a true claim if available
        if has_direct_edge:
            claims.append(CausalClaim(ClaimType.DIRECT_CAUSE, [treatment_idx], outcome_idx))
            validity.append(True)
            num_claims -= 1
            true_claims_pool = [c for c in true_claims_pool if c[0] != 'DIRECT_TRUE']
        elif true_claims_pool:
            claim_type, v1, v2 = true_claims_pool.pop()
            if 'DIRECT' in claim_type:
                claims.append(CausalClaim(ClaimType.DIRECT_CAUSE, [v1], v2))
            elif 'ADJUST' in claim_type:
                claims.append(CausalClaim(ClaimType.ADJUSTMENT_SET, [v1], v2))
            elif 'POST' in claim_type:
                claims.append(CausalClaim(ClaimType.POST_TREATMENT, [v1], v2))
            validity.append(True)
            num_claims -= 1
        
        # Second, add a false claim if available
        if false_claims_pool:
            claim_type, v1, v2 = false_claims_pool.pop()
            
            if 'DIRECT' in claim_type:
                claims.append(CausalClaim(ClaimType.DIRECT_CAUSE, [v1], v2))
            elif 'ADJUST' in claim_type:
                claims.append(CausalClaim(ClaimType.ADJUSTMENT_SET, [v1], v2))
            elif 'POST' in claim_type:
                claims.append(CausalClaim(ClaimType.POST_TREATMENT, [v1], v2))
            validity.append(False)
            num_claims -= 1
        
        # Fill remaining claims with corruption rate
        for _ in range(num_claims):
            want_false = np.random.rand() < corruption_rate
            
            if want_false and false_claims_pool:
                claim_type, v1, v2 = false_claims_pool.pop()
                is_valid = False
            elif true_claims_pool:
                claim_type, v1, v2 = true_claims_pool.pop()
                is_valid = True
            elif false_claims_pool:
                claim_type, v1, v2 = false_claims_pool.pop()
                is_valid = False
            else:
                break
            
            if 'DIRECT' in claim_type:
                claims.append(CausalClaim(ClaimType.DIRECT_CAUSE, [v1], v2))
            elif 'ADJUST' in claim_type:
                claims.append(CausalClaim(ClaimType.ADJUSTMENT_SET, [v1], v2))
            elif 'POST' in claim_type:
                claims.append(CausalClaim(ClaimType.POST_TREATMENT, [v1], v2))
            
            validity.append(is_valid)
        
        return claims, validity
