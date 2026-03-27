# Core DualTF class
from .dual_tf import DualTF, compute_dual_bound, compute_robust_loss_bound

# MLP backward functions
from .tf_mlp import (
    dual_relu_backward, dual_dense_backward, get_relu_masks,
    dual_bias_backward, dual_scale_backward, dual_bn_backward, dual_identity_backward,
)

# Forward bounds (via IntervalTF)
from .tf_forward import compute_forward_bounds

# CNN backward functions
from .tf_cnn import dual_conv2d_backward, dual_maxpool2d_backward, dual_avgpool2d_backward

# Smooth activation backward functions (Sigmoid, Tanh)
from .tf_smooth import (
    dual_smooth_backward, dual_sigmoid_backward, dual_tanh_backward,
    compute_smooth_relaxation, sigmoid, dsigmoid, tanh, dtanh,
)

# RNN backward functions (placeholders)
from .tf_rnn import dual_lstm_backward, dual_gru_backward

# Transformer backward functions (placeholders)
from .tf_transformer import dual_attention_backward, dual_layernorm_backward, dual_gelu_backward

__all__ = [
    # Core
    'DualTF', 'compute_dual_bound', 'compute_robust_loss_bound',
    # MLP
    'dual_relu_backward', 'dual_dense_backward', 'get_relu_masks',
    'dual_bias_backward', 'dual_scale_backward', 'dual_bn_backward', 'dual_identity_backward',
    # Forward
    'compute_forward_bounds',
    # CNN
    'dual_conv2d_backward', 'dual_maxpool2d_backward', 'dual_avgpool2d_backward',
    # Smooth activations (Sigmoid, Tanh)
    'dual_smooth_backward', 'dual_sigmoid_backward', 'dual_tanh_backward',
    'compute_smooth_relaxation', 'sigmoid', 'dsigmoid', 'tanh', 'dtanh',
    # RNN
    'dual_lstm_backward', 'dual_gru_backward',
    # Transformer
    'dual_attention_backward', 'dual_layernorm_backward', 'dual_gelu_backward',
]
