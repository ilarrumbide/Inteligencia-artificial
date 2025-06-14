from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StateSchema:
    ctx_bins = ["small", "medium", "large"]
    tool_bins = [0, 1, 2, 3]  # 3 ≡ "≥3"
    resp_bins = ["short", "medium", "long"]

    # --- números rápidos -------------------------------------------------------
    n_ctx = len(ctx_bins)  # 3
    n_tools = len(tool_bins)  # 4
    n_resp = len(resp_bins)  # 3
    n_states = n_ctx * n_tools * n_resp  # 36

    # --- helpers ---------------------------------------------------------------
    def to_one_hot(self, idx_triplet: np.ndarray) -> np.ndarray:  # → (10,)
        ctx, tools, resp = idx_triplet
        vec = np.zeros(10, dtype=np.float32)
        vec[ctx] = 1.0  # 0-2
        vec[3 + tools] = 1.0  # 3-6
        vec[7 + resp] = 1.0  # 7-9
        return vec
    
@dataclass(slots=True)
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray | None
    d: bool
