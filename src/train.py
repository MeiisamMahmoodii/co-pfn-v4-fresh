import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import CausalPFNDataset, collate_fn
from src.models.core import TheoryFirstTransformer
import os
import time


def pairwise_ranking_loss(trust_pred: torch.Tensor, validity_truth: torch.Tensor, n_claims: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Pairwise ranking loss: Trust(true) should be > Trust(false) + margin within same sample.
    
    Args:
        trust_pred: [B, K, 1] predicted trust scores
        validity_truth: [B, K, 1] ground truth validity (1=true, 0=false)
        n_claims: [B] number of actual claims per sample
        margin: minimum gap between true and false trust scores
    
    Returns:
        Scalar loss encouraging trust separation
    """
    batch_size = trust_pred.shape[0]
    total_loss = 0.0
    n_pairs = 0
    
    for b in range(batch_size):
        k = n_claims[b].item()
        if k < 2:
            continue
            
        trust_b = trust_pred[b, :k, 0]  # [k]
        valid_b = validity_truth[b, :k, 0]  # [k]
        
        # Find true and false indices
        true_mask = valid_b > 0.5
        false_mask = valid_b < 0.5
        
        if not true_mask.any() or not false_mask.any():
            continue
        
        # Get trust scores for true and false claims
        true_trusts = trust_b[true_mask]  # [n_true]
        false_trusts = trust_b[false_mask]  # [n_false]
        
        # For each (true, false) pair, we want: true_trust > false_trust + margin
        # Loss = max(0, margin - (true_trust - false_trust))
        for t_trust in true_trusts:
            for f_trust in false_trusts:
                pair_loss = F.relu(margin - (t_trust - f_trust))
                total_loss += pair_loss
                n_pairs += 1
    
    if n_pairs == 0:
        return torch.tensor(0.0, device=trust_pred.device)
    
    return total_loss / n_pairs


def train():
    # 1. Hyperparameters
    max_vars = 20
    max_samples = 1000
    batch_size = 16
    lr = 1e-4
    n_epochs = 100
    steps_per_epoch = 100
    lambda_ate = 1.0
    lambda_trust = 5.0  # BCE loss weight
    lambda_pairwise = 3.0  # Pairwise ranking loss weight - reduced to prevent trust equalization
    alpha_trust_weight = 0.5  # Trust-weighted ATE loss: weight = 1 + alpha * mean_trust
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- ADVERSARIAL TRAINING WITH PAIRWISE RANKING + TRUST-WEIGHTED ATE ---")
    print(f"Device: {device} | Max Epochs: {n_epochs}")
    print(f"Weights: ATE={lambda_ate}, Trust={lambda_trust}, Pairwise={lambda_pairwise}")
    print(f"Trust-Weighted ATE: alpha={alpha_trust_weight}, weight range [1.0, 1.5]")

    # 2. Data & Model
    dataset = CausalPFNDataset(
        min_vars=5, 
        max_vars=max_vars, 
        min_samples=20, 
        max_samples=max_samples, 
        null_rate=0.15,
        corruption_rate=0.5,  # 50% false claims
        enforce_edge_rate=0.35  # Reduced from 0.6 to prevent trust saturation
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    model = TheoryFirstTransformer(n_vars=max_vars).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    criterion_ate = nn.HuberLoss()
    criterion_trust = nn.BCELoss()

    # 3. Training Loop
    model.train()
    data_iter = iter(dataloader)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_ate_loss = 0.0
        epoch_trust_loss = 0.0
        epoch_pair_loss = 0.0
        
        for step in range(steps_per_epoch):
            batch = next(data_iter)
            
            data = batch['data'].to(device)
            claims = batch['claims'].to(device)
            ate_truth = batch['ate_truth'].to(device)
            validity_truth = batch['validity_truth'].to(device).unsqueeze(-1)
            n_claims = batch['n_claims'].to(device)
            n_samples = batch['n_samples'].to(device)
            
            optimizer.zero_grad()
            
            ate_pred, trust_pred = model(data, claims, n_claims=n_claims, n_samples=n_samples)
            
            loss_trust = criterion_trust(trust_pred, validity_truth)
            loss_pairwise = pairwise_ranking_loss(trust_pred, validity_truth, n_claims, margin=0.3)
            
            # Trust-weighted ATE loss: force ATE to improve when claiming high trust
            # Compute mean trust per sample (over valid claims only)
            batch_size_actual = trust_pred.shape[0]
            ate_weights = []
            for b in range(batch_size_actual):
                k = n_claims[b].item()
                if k > 0:
                    # Mean trust for this sample
                    mean_trust_b = trust_pred[b, :k, 0].mean()
                    # Weight: 1 + alpha * mean_trust, clamped to [1.0, 1.5]
                    weight = (1.0 + alpha_trust_weight * mean_trust_b).clamp(1.0, 1.5)
                    ate_weights.append(weight)
                else:
                    ate_weights.append(torch.tensor(1.0, device=device))
            
            ate_weights = torch.stack(ate_weights)  # [B]
            
            # Compute per-sample ATE loss using Huber loss
            ate_residual = torch.abs(ate_pred.squeeze(-1) - ate_truth)
            huber_threshold = 1.0
            ate_loss_per_sample = torch.where(
                ate_residual <= huber_threshold,
                0.5 * ate_residual ** 2,
                huber_threshold * (ate_residual - 0.5 * huber_threshold)
            )
            loss_ate_weighted = (ate_loss_per_sample * ate_weights).mean()
            
            loss = lambda_ate * loss_ate_weighted + lambda_trust * loss_trust + lambda_pairwise * loss_pairwise
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ate_loss += loss_ate_weighted.item()
            epoch_trust_loss += loss_trust.item()
            epoch_pair_loss += loss_pairwise.item()
        
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"E{epoch+1:03d} | Total: {epoch_loss/steps_per_epoch:.4f} "
            f"| ATE: {epoch_ate_loss/steps_per_epoch:.4f} | Trust: {epoch_trust_loss/steps_per_epoch:.4f} "
            f"| Pair: {epoch_pair_loss/steps_per_epoch:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.1f}s")
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/model_adv_e{epoch+1}.pt')

    # 4. Save Final
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/model_adversarial_v2.pt')
    print("Training complete. Ground-breaking model (maybe) saved.")

if __name__ == "__main__":
    train()
