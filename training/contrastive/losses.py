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


def supcon_loss(features, labels, temperature=0.07, device='cuda'):
    """
    Compute Supervised Contrastive Loss.
    Implementation following: https://arxiv.org/pdf/2004.11362
    Source: https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
    Intermediate:
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.

    """
    batch_size = features.size(0)
    anchor_count = contrast_count = features.size(1)

    labels = labels.contiguous().view(-1, 1) # [B, 1]
    anchor_feature = contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # [B, #views, D]

    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T), temperature
    )

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = torch.eq(labels, labels.T).float().to(device)  # Positive pairs mask
    mask = mask.repeat(anchor_count, contrast_count)  # [B, B] -> [B, B, N, N]
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask


    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

    # compute mean of log-likelihood over positive
    # modified to handle edge cases when there is no positive pair
    # for an anchor point. 
    # Edge case e.g.:- 
    # features of shape: [4,1,...]
    # labels:            [0,1,1,2]
    # loss before mean:  [nan, ..., ..., nan] 
    mask_pos_pairs = mask.sum(1)
    mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

    # loss
    loss = -mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    # mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    # loss = -mean_log_prob_pos.mean()

    return loss