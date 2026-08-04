"""The 3-D rule+exception (Shepard-Hovland type-5) task as Gaussian blobs.

Same task as the two-neuron study: 8 signed (+/-1) cube corners, rule ``y = NOT
x0`` with the pocket ``{x1=+1, x2=-1}`` flipping two of them. Here the corners are
treated as *prototypes*: each training input is a prototype plus fresh Gaussian
jitter of std ``signal_noise_std`` on the 3 coordinates. ``signal_noise_std = 0``
recovers the 8 fixed examples; larger values make the classes Gaussian blobs.

Rule (non-exception) stimuli are "central", exception stimuli are "peripheral",
matching the reference notebook's vocabulary.
"""

import numpy as np
import torch

# 6 central (rule) prototypes and 2 peripheral (exception) prototypes, signed.
CENTRAL_PROTOTYPES = np.array([
    [+1, +1, +1], [+1, -1, -1], [+1, -1, +1],
    [-1, +1, +1], [-1, -1, -1], [-1, -1, +1],
], dtype=np.float32)
CENTRAL_LABELS = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)

PERIPHERAL_PROTOTYPES = np.array([[-1, +1, -1], [+1, +1, -1]], dtype=np.float32)
PERIPHERAL_LABELS = np.array([0, 1], dtype=np.float32)

# Full 8-prototype view (central first, then peripheral) for evaluation.
ALL_PROTOTYPES = np.concatenate([CENTRAL_PROTOTYPES, PERIPHERAL_PROTOTYPES], 0)
ALL_LABELS = np.concatenate([CENTRAL_LABELS, PERIPHERAL_LABELS], 0)
IS_PERIPHERAL = np.array([False] * 6 + [True] * 2)   # exception mask over ALL_*
IS_CENTRAL = ~IS_PERIPHERAL
NUM_FEATURES = 3


class BlobTask:
    """Task prototypes moved onto a torch device (central / peripheral split)."""

    def __init__(self, device: torch.device):
        self.device = device
        to = lambda a: torch.tensor(a, device=device)
        self.central_prototypes = to(CENTRAL_PROTOTYPES)      # (6, 3)
        self.central_labels = to(CENTRAL_LABELS)              # (6,)
        self.peripheral_prototypes = to(PERIPHERAL_PROTOTYPES)  # (2, 3)
        self.peripheral_labels = to(PERIPHERAL_LABELS)          # (2,)
        self.all_prototypes = to(ALL_PROTOTYPES)              # (8, 3)
        self.all_labels = to(ALL_LABELS)                      # (8,)
        self.is_peripheral = torch.tensor(IS_PERIPHERAL, device=device)
        self.is_central = torch.tensor(IS_CENTRAL, device=device)
