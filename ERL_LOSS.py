def compute_erl_loss(features, labels, num_classes):
    """
    Calculate ERL (Embedding-guided Representation Learning) Loss

    Args:
        features: Feature vectors, shape=(batch_size, feature_dim)
        labels: Ground truth labels, shape=(batch_size,)
        num_classes: Number of classes
    Returns:
        erl_loss: The calculated ERL loss
    """
    # Calculate feature centroids for each class
    centroids = []
    for i in range(num_classes):
        mask = (labels == i)
        if mask.sum() > 0:
            # Mean of features for class i
            centroid = features[mask].mean(dim=0)
            centroids.append(centroid)
    centroids = torch.stack(centroids)

    # Calculate average distance between samples and their class centroid
    distances = []
    for i in range(num_classes):
        mask = (labels == i)
        if mask.sum() > 0:
            # Calculate Euclidean distance and take mean
            dist = torch.norm(features[mask].unsqueeze(1) - centroids[i], dim=2).mean()
            distances.append(dist)

    # Convert distances to ranks (larger distance gets smaller rank)
    distances = torch.stack(distances)
    ranks = torch.argsort(torch.argsort(distances, descending=True))

    # Calculate adaptive weights based on ranks
    k = num_classes
    weights = (k + 1 - ranks).float()
    weights = weights / weights.sum()  # Normalize weights

    # Calculate weighted ERL loss
    erl_loss = (weights * distances).sum()

    return erl_loss