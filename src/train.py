import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import CausalPFNDataset, collate_fn
from src.models.core import TheoryFirstTransformer
from src.audit_suite import run_audit, AuditConfig, print_terminal_summary, write_report
import time
import random


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


def trust_ate_correlation_loss(trust_pred: torch.Tensor, ate_pred: torch.Tensor, ate_truth: torch.Tensor, n_claims: torch.Tensor) -> torch.Tensor:
    """
    Lightweight correlation loss: Encourage high trust → low ATE error.
    Uses simple linear correlation (faster than Spearman for training).
    
    Args:
        trust_pred: [B, K, 1] predicted trust scores
        ate_pred: [B, 1] predicted ATEs
        ate_truth: [B, 1] ground truth ATEs
        n_claims: [B] number of actual claims per sample
    
    Returns:
        Scalar loss
    """
    batch_size = trust_pred.shape[0]
    correlations = []
    
    for b in range(batch_size):
        k = n_claims[b].item()
        if k < 2:
            continue
        
        # Trust scores for this sample
        trust_b = trust_pred[b, :k, 0]  # [k]
        
        # ATE error: higher error = worse ATE, so trust should be LOW
        ate_pred_scalar = ate_pred[b].squeeze() if ate_pred[b].dim() > 0 else ate_pred[b]
        ate_truth_scalar = ate_truth[b].squeeze() if ate_truth[b].dim() > 0 else ate_truth[b]
        ate_error = torch.abs(ate_pred_scalar - ate_truth_scalar).expand(k)  # [k]
        
        # Linear correlation: corr = cov(trust, error) / (std(trust) * std(error))
        trust_norm = trust_b - trust_b.mean()
        error_norm = ate_error - ate_error.mean()
        
        cov = (trust_norm * error_norm).mean()
        std_trust = trust_norm.std()
        std_error = error_norm.std()
        
        if std_trust > 1e-6 and std_error > 1e-6:
            corr = cov / (std_trust * std_error + 1e-8)
        else:
            corr = torch.tensor(0.0, device=trust_pred.device)
        
        # We want NEGATIVE correlation: high trust → LOW error
        # Loss = max(0, corr) penalizes positive/zero correlations
        correlations.append(F.relu(corr))
    
    if len(correlations) == 0:
        return torch.tensor(0.0, device=trust_pred.device)
    
    return torch.stack(correlations).mean()


def train():
    # Initialize distributed training if running with torchrun
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        is_main_process = rank == 0
    else:
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_main_process = True
    
    # 1. Hyperparameters
    max_vars = 20
    max_samples = 1000
    batch_size = 16
    lr = 1e-4
    n_epochs = 100
    steps_per_epoch = 100
    lambda_ate = 1.0
    lambda_trust = 5.0  # BCE loss weight
    lambda_pairwise = 4.0  # Pairwise ranking loss weight - increased for stronger discrimination
    alpha_trust_weight = 0.3  # Trust-weighted ATE loss: weight = 1 + alpha * mean_trust (reattached)
    alpha_trust_weight = 0.3  # (Phase 8) Optimal balance for correction
    lambda_correlation = 0.0  # Disabled to ensure pure detached trust
    teacher_forcing_ratio = 0.1  # (Phase 8) Target 0.38% Efficiency without breaking robustness
    
    if is_main_process:
        print(f"--- ADVERSARIAL TRAINING (Option A + Detached) ---")
        print(f"Device: {device} | Rank: {rank}/{world_size} | Max Epochs: {n_epochs}")
        print(f"Weights: ATE={lambda_ate}, Trust={lambda_trust}, Pairwise={lambda_pairwise}, Correlation={lambda_correlation}")
        print(f"Trust-Weighted ATE: alpha={alpha_trust_weight}, detached gradients")
        print(f"Goal: Strong discrimination + weak correlation nudge to align trust with ATE without collapse")

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
    
    if distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, 
                              pin_memory=True, num_workers=4, prefetch_factor=2)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn,
                              pin_memory=True, num_workers=4, prefetch_factor=2)
    
    model = TheoryFirstTransformer(n_vars=max_vars).to(device)
    
    if distributed:
        model = DDP(model, device_ids=[rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    criterion_ate = nn.HuberLoss()
    criterion_trust = nn.BCELoss()
    
    # Enable cuDNN benchmarking for faster training
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    # 3. Training Loop
    model.train()
    data_iter = iter(dataloader)
    
    start_time = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_ate_loss = 0.0
        epoch_trust_loss = 0.0
        epoch_pair_loss = 0.0
        epoch_corr_loss = 0.0
        
        for step in range(steps_per_epoch):
            batch = next(data_iter)
            
            data = batch['data'].to(device)
            claims = batch['claims'].to(device)
            ate_truth = batch['ate_truth'].to(device)
            validity_truth = batch['validity_truth'].to(device).unsqueeze(-1)
            n_claims = batch['n_claims'].to(device)
            n_samples = batch['n_samples'].to(device)
            
            optimizer.zero_grad()
            
            # TEACHER FORCING: Occasionally force the gate open for valid claims
            # This solves the "Lazy Mechanic" problem where the Correction Head never gets a signal
            trust_override = None
            if random.random() < teacher_forcing_ratio:
                trust_override = validity_truth # [B, K, 1] - Use ground truth (1.0 or 0.0)

            ate_pred, trust_pred = model(data, claims, n_claims=n_claims, n_samples=n_samples, trust_override=trust_override)
            
            loss_trust = criterion_trust(trust_pred, validity_truth)
            loss_pairwise = pairwise_ranking_loss(trust_pred, validity_truth, n_claims, margin=0.3)
            loss_correlation = trust_ate_correlation_loss(trust_pred, ate_pred, ate_truth, n_claims)
            
            # Trust-weighted ATE loss: force ATE to improve when claiming high trust
            # Compute mean trust per sample (over valid claims only)
            batch_size_actual = trust_pred.shape[0]
            ate_weights = []
            for b in range(batch_size_actual):
                k = n_claims[b].item()
                if k > 0:
                    # Mean trust for this sample - REATTACHED to allow gradients (Fix #6)
                    mean_trust_b = trust_pred[b, :k, 0].detach().mean()  # Detached (Phase 3 style)
                    # Weight: 1 + alpha * mean_trust, clamped to [1.0, 1.4] (softer range)
                    weight = (1.0 + alpha_trust_weight * mean_trust_b).clamp(1.0, 1.4)
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
            
            loss = lambda_ate * loss_ate_weighted + lambda_trust * loss_trust + lambda_pairwise * loss_pairwise + lambda_correlation * loss_correlation
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_ate_loss += loss_ate_weighted.item()
            epoch_trust_loss += loss_trust.item()
            epoch_pair_loss += loss_pairwise.item()
            epoch_corr_loss += loss_correlation.item()
        
        scheduler.step()

        elapsed = time.time() - start_time
        if is_main_process:
            print(f"E{epoch+1:03d} | Total: {epoch_loss/steps_per_epoch:.4f} "
                f"| ATE: {epoch_ate_loss/steps_per_epoch:.4f} | Trust: {epoch_trust_loss/steps_per_epoch:.4f} "
                f"| Pair: {epoch_pair_loss/steps_per_epoch:.4f} | Corr: {epoch_corr_loss/steps_per_epoch:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f} | Time: {elapsed:.1f}s")
        
        # Periodic checkpoint (main process only)
        if is_main_process and (epoch + 1) % 10 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.module.state_dict() if distributed else model.state_dict(), f'checkpoints/model_adv_e{epoch+1}.pt')

    # 4. Save Final (main process only)
    if is_main_process:
        os.makedirs('checkpoints', exist_ok=True)
        final_ckpt_path = 'checkpoints/model_adversarial_v2.pt'
        torch.save(model.module.state_dict() if distributed else model.state_dict(), final_ckpt_path)
        print("Training complete. Ground-breaking model saved.")
        
        # --- BAKED-IN AUDIT (The "Trustable Test Suite") ---
        print("\n" + "="*80)
        print("STARTING AUTOMATED AUDIT PROTOCOL")
        print("="*80)
        
        # Configure audit matching training context
        audit_cfg = AuditConfig(
            checkpoint=final_ckpt_path,
            n_vars=max_vars,
            seed=42  # Fixed seed for trustable comparison
        )
        
        # Run audit and show results
        report_md, metrics = run_audit(audit_cfg)
        print_terminal_summary(metrics)
        
        # Save full report
        report_path = os.path.join("results", "audit_report_final.md")
        write_report(report_path, report_md)
        print(f"Full verified report saved to: {report_path}")

    
    # Cleanup distributed training
    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()
