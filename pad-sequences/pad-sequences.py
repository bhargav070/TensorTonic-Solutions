import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    seqs = list(seqs)
    N = len(seqs)

    if N == 0:
        L = max_len if max_len is not None else 0
        return np.full((0, L), pad_values, dtype = float)

    if max_len is None:
        L = max(len(s) for s in seqs)
    else:
        L = max_len

    out = np.full((N, L), pad_value, dtype = float)

    for i, seq in enumerate(seqs):
        seq = np.asarray(seq, dtype=float)
        length = min(len(seq), L)
        out[i, :length] = seq[:length]

    return out
    pass