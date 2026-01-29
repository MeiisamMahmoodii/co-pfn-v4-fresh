import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class TabPFNEncoder(nn.Module):
    """Refined TabPFN encoder with non-linear mapping."""
    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class TheoryFirstTransformer(nn.Module):
    """
    Adversarial Architecture: 
    Forces Claims to query Data via Cross-Attention.
    Directly splits ATE into Base (Data-only) and Correction (Claim-driven).
    """
    def __init__(self, n_vars: int, embed_dim: int = 128, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Encoders
        self.data_encoder = TabPFNEncoder(n_vars, embed_dim)
        self.claim_encoder = nn.Linear(2 * n_vars + 1, embed_dim)
        
        # 1. Base PFN (Processes data in isolation to get a baseline)
        self.base_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dropout=0.1),
            num_layers=n_layers
        )
        self.base_ate_head = nn.Linear(embed_dim, 1)

        # 2. Theory Auditor (Processes claims)
        self.theory_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dropout=0.1),
            num_layers=2
        )
        self.trust_head = nn.Linear(embed_dim, 1)

        # 3. Cross-Attention: Claims (Q) query Data (K/V)
        # This force-feeds evidence into the theory
        self.cross_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm_cross = nn.LayerNorm(embed_dim)

        # 4. Correction Head
        self.correction_head = nn.Linear(embed_dim, 1)
        
        # 5. Learnable trust amplification scale
        self.trust_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, data: torch.Tensor, claims: torch.Tensor, n_claims: Optional[torch.Tensor] = None, n_samples: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = data.shape[0]
        max_samples = data.shape[1]
        k_claims = claims.shape[1]
        
        # Create claim mask (True = padded/masked position)
        claim_mask = torch.zeros(batch_size, k_claims, dtype=torch.bool, device=data.device)
        if n_claims is not None:
            for i in range(batch_size):
                claim_mask[i, n_claims[i].item():] = True
        
        # Create data padding mask (True = padded/masked position)
        data_mask = torch.zeros(batch_size, max_samples, dtype=torch.bool, device=data.device)
        if n_samples is not None:
            for i in range(batch_size):
                data_mask[i, n_samples[i].item():] = True

        # STEP 1: Process Data alone
        h_data = self.data_encoder(data)   # [B, N, D]
        # Apply data padding mask to base transformer if available
        if n_samples is not None:
            h_data_late = self.base_transformer(h_data, src_key_padding_mask=data_mask)
        else:
            h_data_late = self.base_transformer(h_data)
        
        # Masked mean pooling: only average over real (non-padded) samples
        if n_samples is not None:
            # Create weight mask: 1 for real samples, 0 for padded
            data_weight = (~data_mask).float().unsqueeze(-1)  # [B, N, 1]
            h_data_sum = (h_data_late * data_weight).sum(dim=1)  # [B, D]
            n_real = data_weight.sum(dim=1).clamp(min=1.0)  # [B, 1] avoid div by zero
            h_data_mean = h_data_sum / n_real
        else:
            h_data_mean = h_data_late.mean(dim=1)
        base_ate = self.base_ate_head(h_data_mean)  # [B, 1]

        # STEP 2: Process Claims
        h_claims = self.claim_encoder(claims) # [B, K, D]
        
        # Check if we have at least one valid claim in the batch to avoid Transformer crash
        if n_claims is not None and (n_claims > 0).any():
            h_claims = self.theory_transformer(h_claims, src_key_padding_mask=claim_mask)
        else:
            # Skip theory transformer if no active claims
            pass
        
        # PREVIOUSLY: trust_scores = torch.sigmoid(self.trust_head(h_claims))
        # REMOVED: Redundant and non-data-dependent

        # STEP 3: Forced Interaction (Cross Attention)
        # Claims search data for confirmation
        # We REMOVE the residual for the Audit Head to stop syntax-checking.
        # Apply data padding mask so claims don't attend to padded rows
        theory_evidence, _ = self.cross_attn(
            query=h_claims, 
            key=h_data_late, 
            value=h_data_late,
            key_padding_mask=data_mask if n_samples is not None else None
        )
        
        # STEP 4: Data-Dependent Audit (The Cold Interrogation)
        # We use ONLY the evidence from data to compute trust. 
        # h_claims is NOT allowed here.
        trust_raw = self.trust_head(theory_evidence)  # Raw logits [B, K, 1]
        trust_scores = torch.sigmoid(trust_raw) # [B, K, 1] - for loss computation
        
        # Suppression mask for padding
        m = (1.0 - claim_mask.float()).unsqueeze(-1)
        trust_scores = trust_scores * m

        # STEP 5: Amplified Trust Scaling for Correction Gating
        # Apply non-linear (sqrt) amplification + learnable scale to boost correction when trust is high
        # sqrt: boosts mid-range trust (0.3→0.55, 0.5→0.71, 0.7→0.84)
        # learnable scale: allows model to discover optimal amplification
        # Note: We use sigmoid on amplified raw logits to keep values in [0, 1]
        trust_amplified = torch.sigmoid(trust_raw * torch.relu(self.trust_scale)) * m

        # STEP 6: Weighted Correction (Purified: Evidence ONLY)
        # Prediction must rely solely on the evidence extracted from data.
        theory_full_context = self.norm_cross(theory_evidence)
        
        gated_theory = theory_full_context * trust_amplified
        theory_contribution = self.correction_head(gated_theory.sum(dim=1)) # [B, 1]

        # Final ATE
        final_ate = base_ate + theory_contribution
        
        return final_ate, trust_scores
