import torch

def compute_spe_loss(features, pseudo_labels, source_centroids, num_classes):
    """
    Calculate SPE (Self-adaptive Pseudo-label Enhancement) Loss

    Args:
        features: Feature vectors from target domain, shape=(batch_size, feature_dim)
        pseudo_labels: Generated pseudo labels, shape=(batch_size,)
        source_centroids: Source domain class centers, shape=(num_classes, feature_dim)
        num_classes: Number of classes

    Returns:
        spe_loss: The calculated SPE loss
    """
    # Calculate target domain centroid
    target_centroid = features.mean(dim=0)

    # Calculate corresponding source domain centroid based on pseudo label
    source_centroid = source_centroids[pseudo_labels[0]]  # Use first pseudo label since they're all same

    # Calculate distribution alignment loss
    alignment_loss = torch.norm(target_centroid - source_centroid) ** 2

    # Calculate feature compactness loss
    compactness_loss = torch.norm(features - target_centroid.unsqueeze(0), dim=1).mean()

    # Total SPE loss
    spe_loss = alignment_loss + compactness_loss / features.size(0)

    return spe_loss