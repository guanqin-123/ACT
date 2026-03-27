import torch
from typing import Tuple, Optional, Dict
from act.back_end.core import Bounds

# -------- LSTM --------
@torch.no_grad()
def dual_lstm_backward(
    nu: torch.Tensor, weight_ih: torch.Tensor, weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None, bias_hh: Optional[torch.Tensor] = None,
    bounds_dict: Optional[Dict[str, Bounds]] = None, seq_len: int = 1,
    hidden_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LSTM backward through time. Requires gate bounds from forward pass. (Pending)"""
    raise NotImplementedError("dual_lstm_backward: pending")

# -------- GRU --------
@torch.no_grad()
def dual_gru_backward(
    nu: torch.Tensor, weight_ih: torch.Tensor, weight_hh: torch.Tensor,
    bias_ih: Optional[torch.Tensor] = None, bias_hh: Optional[torch.Tensor] = None,
    bounds_dict: Optional[Dict[str, Bounds]] = None, seq_len: int = 1,
    hidden_size: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GRU backward through time. Requires gate bounds from forward pass. (Pending)"""
    raise NotImplementedError("dual_gru_backward: pending")
