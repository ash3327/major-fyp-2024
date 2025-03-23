import torch
import torch.nn.functional as F

def info_nce_loss(anchors, positives, temperature=0.1):
    """
    InfoNCE loss for contrastive learning.
    
    Args:
        anchors: Tensor [B, D] of anchor embeddings.
        positives: Tensor [B, D] of positive embeddings.
        temperature: Scaling factor for similarities.
    
    Returns:
        loss: Scalar loss value.
    """
    sim = torch.mm(anchors, positives.t()) / temperature  # [B, B]
    labels = torch.arange(anchors.size(0)).to(anchors.device)
    return F.cross_entropy(sim, labels)