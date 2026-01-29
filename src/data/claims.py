import enum
from dataclasses import dataclass
from typing import List, Tuple, Optional

class ClaimType(enum.IntEnum):
    ADJUSTMENT_SET = 0
    INSTRUMENTAL_VARIABLE = 1
    DIRECT_CAUSE = 2
    NO_DIRECT_CAUSE = 3
    POST_TREATMENT = 4

@dataclass
class CausalClaim:
    claim_type: ClaimType
    variables: List[int]  # Indices of variables involved
    target: int           # Usually the treatment (T) or outcome (Y) index
    
    def to_vector(self, max_vars: int) -> List[float]:
        """Convert claim to a fixed-size vector for the Transformer."""
        vec = [float(self.claim_type)]
        # Add a multi-hot encoding of involved variables
        ids = [0.0] * max_vars
        for v in self.variables:
            if v < max_vars:
                ids[v] = 1.0
        vec.extend(ids)
        # Add target index as a one-hot vector
        target_ids = [0.0] * max_vars
        if self.target < max_vars:
            target_ids[self.target] = 1.0
        vec.extend(target_ids)
        return vec

def claim_vector_size(max_vars: int) -> int:
    """Return fixed-size claim vector length for a given variable count."""
    return 2 * max_vars + 1

@dataclass
class CausalWorldMetadata:
    true_ate: float
    ground_truth_validity: List[bool]
    claims: List[CausalClaim]
