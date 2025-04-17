import torch


def shannon_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calculate Shannon entropy of a probability distribution."""
    # Remove zero entries to avoid log(0)
    probs = probs[probs > 0]
    entropy = -torch.sum(probs * torch.log(probs))

    return entropy


def dkl(p, q):
    # Ensure same length
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]

    # Normalize the distributions
    p = p / torch.sum(p)
    q = q / torch.sum(q)

    p = p[p > 0]
    q = q[q > 0]

    return torch.sum(p * torch.log(p / q))
