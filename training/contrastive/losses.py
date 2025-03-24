import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.1, device='cuda', n_views=2):
    """
    InfoNCE loss for contrastive learning.
    
    Args:
        features: Tensor [2B, D] of embeddings (cat(anchor, positives))
        temperature: Scaling factor for similarities.
    
    Returns:
        loss: Scalar loss value.
    """
    batch_size = features.shape[0]//2
    # Create labels indicating which original batch each augmented sample belongs to
    labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # 2B x 2B mask
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    sim = torch.mm(features, features.T)  # [2B, 2B]

    # Remove self-similarity (diagonal entries)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)  # 2B x (2B-1)
    sim = sim[~mask].view(sim.shape[0], -1)  # 2B x (2B-1)

    # Separate positive and negative samples
    positives = sim[labels.bool()].view(labels.shape[0], -1)  # 2B x 1
    negatives = sim[~labels.bool()].view(labels.shape[0], -1)  # 2B x (2B-2)

    # Combine pos/neg for cross entropy: logits[pos, neg...], target index 0
    logits = torch.cat([positives, negatives], dim=1)  # 2B x (2B-1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)  # 2B

    # Apply temperature scaling
    logits /= temperature

    return F.cross_entropy(logits, labels)