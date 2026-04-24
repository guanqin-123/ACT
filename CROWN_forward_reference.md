# CROWN Forward Bound Propagation Reference Cheat-Sheet
## α-β-CROWN / auto_LiRPA Implementation Guide

---

## 1. Main Forward CROWN Entry Point

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/forward_bound.py`  
**Function**: `forward_general()` (lines 32–100)

**Purpose**: Top-level orchestrator for forward CROWN bound propagation. Recursively computes linear bounds through the network.

**Key excerpt** (lines 32–65):
```python
def forward_general(self: 'BoundedModule', C=None, node:'Bound'=None, concretize=False,
                    offset=0, from_node=False):
    # Recursively compute bounds for input nodes
    for l_pre in node.inputs:
        if not hasattr(l_pre, 'linear'):
            self.forward_general(node=l_pre, offset=offset, from_node=from_node)
    
    inp = [l_pre.linear for l_pre in node.inputs]
    node._start = '_forward'
    
    # Call operator-specific bound_forward
    linear = node.bound_forward(self.dim_in, *inp)
    
    lw, uw = linear.lw, linear.uw  # Lower/upper A matrices
    lower, upper = linear.lb, linear.ub  # Lower/upper bias vectors
```

**Key insight**: Each node stores its linear bounds in a `LinearBound` object with dual-track coefficients (lw, lb, uw, ub).

---

## 2. Dual-Track State Representation

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/linear_bound.py`  
**Class**: `LinearBound` (lines 17–49)

**Purpose**: Container for linear bounds with separate lower and upper coefficient tracking.

**Key excerpt** (lines 17–38):
```python
class LinearBound:
    def __init__(
            self, lw=None, lb=None, uw=None, ub=None, lower=None, upper=None,
            from_input=None, x_L=None, x_U=None, offset=0, tot_dim=None):
        self.lw = lw  # Lower weight matrix: shape [batch, in_dim, out_dim]
        self.lb = lb  # Lower bias vector: shape [batch, out_dim]
        self.uw = uw  # Upper weight matrix: shape [batch, in_dim, out_dim]
        self.ub = ub  # Upper bias vector: shape [batch, out_dim]
        self.lower = lower  # Concrete lower bounds (after concretization)
        self.upper = upper  # Concrete upper bounds (after concretization)
        self.x_L = x_L  # Input lower bounds (for final extraction)
        self.x_U = x_U  # Input upper bounds (for final extraction)
```

**Shape convention**:
- `lw, uw`: `[batch_size, input_dim, output_dim]` (flattened for matrix ops)
- `lb, ub`: `[batch_size, output_dim]`
- Represents: `y_lb = lw @ x + lb`, `y_ub = uw @ x + ub`

---

## 3. ReLU Forward CROWN Bound

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/relu.py`  
**Methods**: `_forward_relaxation()` (lines 493–519), `bound_forward()` (lines 571–580), `_relu_lower_bound_init()` (lines 430–454)

**Purpose**: Compute CROWN relaxation for ReLU: dual-track linear lower/upper bounds.

**Key excerpt – bound_forward** (lines 571–580):
```python
def bound_forward(self, dim_in, x):
    self._forward_relaxation(x)  # Compute alpha, upper slope/intercept
    lb = self.lw * x.lb  # Lower bound: α·x_lb
    ub = self.uw * x.ub + self.ub  # Upper bound: slope·x_ub + intercept
    lw = (self.lw.unsqueeze(1) * x.lw) if x.lw is not None else None
    uw = (self.uw.unsqueeze(1) * x.uw) if x.uw is not None else None
    return LinearBound(lw, lb, uw, ub)
```

**Key excerpt – _forward_relaxation** (lines 493–519):
```python
def _forward_relaxation(self, x):
    self._init_masks(x)  # Identify positive/negative/unstable neurons
    
    # Upper bound: CROWN relaxation slope and intercept
    upper_k, upper_b = self._relu_upper_bound(x.lower, x.upper, self.leaky_alpha)
    self.uw = self.mask_pos + self.mask_both * upper_k  # Upper slope
    self.ub = self.mask_both * upper_b  # Upper intercept
    
    # Lower bound: adaptive alpha selection (Zhang heuristic)
    if self.opt_stage in ['opt', 'reuse']:
        lower_k = self.alpha['_forward'][0, 0]  # Optimized alpha
    else:
        lower_k = self._relu_lower_bound_init(upper_k)  # Heuristic alpha
    
    self.lw = self.mask_both * lower_k + self.mask_pos  # Lower slope
```

**Key excerpt – _relu_lower_bound_init (adaptive heuristic)** (lines 430–454):
```python
def _relu_lower_bound_init(self, upper_k):
    # Adaptive: α = 1 if upper_k > 0.5 else 0
    if self.relu_options == "adaptive":
        if self.leaky_alpha == 0:
            lower_k = (upper_k > 0.5).to(upper_k)  # Zhang heuristic
        else:
            lower_k = ((upper_k > 0.5).to(upper_k) + 
                       (upper_k <= 0.5).to(upper_k) * self.leaky_alpha)
    return lower_k
```

**Key excerpt – _relu_upper_bound** (lines 584–595):
```python
@staticmethod
@torch.jit.script
def _relu_upper_bound(lb, ub, leaky_alpha: float):
    """Upper bound slope and intercept according to CROWN relaxation."""
    lb_r = lb.clamp(max=0)
    ub_r = ub.clamp(min=0)
    ub_r = torch.max(ub_r, lb_r + 1e-8)
    upper_d = ub_r / (ub_r - lb_r)  # Slope: u/(u-l)
    upper_b = - lb_r * upper_d  # Intercept: -l·slope
    return upper_d, upper_b
```

**Semantics**:
- **Lower bound**: `y >= α·x` where α ∈ {0, 1} (or optimized)
- **Upper bound**: `y <= slope·x + intercept` where slope = u/(u-l), intercept = -l·slope
- **Masks**: `mask_pos` (x > 0), `mask_both` (x ∈ [l, u] unstable), `mask_neg` (x < 0)

---

## 4. Conv2D Forward CROWN (Coefficient Propagation)

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/convolution.py`  
**Method**: `bound_forward()` (lines 426–460)

**Purpose**: Propagate linear coefficients through Conv2D **without resetting to interval bounds**.

**Key excerpt** (lines 426–460):
```python
def bound_forward(self, dim_in, *x):
    weight = x[1].lb  # Conv weight
    bias = x[2].lb if self.has_bias else None
    x = x[0]  # Input LinearBound
    
    # Decompose input bounds into center + deviation
    mid_w = (x.lw + x.uw) / 2
    mid_b = (x.lb + x.ub) / 2
    diff_w = (x.uw - x.lw) / 2
    diff_b = (x.ub - x.lb) / 2
    
    weight_abs = weight.abs()
    shape = mid_w.shape
    shape_wconv = [shape[0] * shape[1]] + list(shape[2:])
    
    # Apply F.conv2d to coefficient tensors (NOT just interval bounds!)
    deviation_w = self.F_conv(
        diff_w.reshape(shape_wconv), weight_abs, None,
        self.stride, self.padding, self.dilation, self.groups)
    center_w = self.F_conv(
        mid_w.reshape(shape_wconv), weight, None,
        self.stride, self.padding, self.dilation, self.groups)
    
    # Reconstruct dual-track bounds
    return LinearBound(
        lw = center_w - deviation_w,
        lb = center_b - deviation_b,
        uw = center_w + deviation_w,
        ub = center_b + deviation_b)
```

**Key insight**: The A matrices (lw, uw) are reshaped and passed through `F.conv2d` directly, preserving the linear structure. No reset to interval bounds!

---

## 5. Bias / BatchNorm / Scale Forward (Affine Composition)

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/normalization.py`  
**Method**: `bound_forward()` (lines 79–110)

**Purpose**: Compose affine transformations (scale + shift) with linear bounds.

**Key excerpt – BatchNorm** (lines 79–110):
```python
def bound_forward(self, dim_in, *x):
    inp = x[0]
    weight, bias = x[1].lower, x[2].lower  # BN scale & shift
    
    # Compute affine transformation: y = (w / sqrt(var + eps)) * x + (b - mean * w / sqrt(var + eps))
    tmp_weight = weight / torch.sqrt(self.current_var + self.eps)
    tmp_bias = bias - self.current_mean * tmp_weight
    
    # Propagate through linear bounds (element-wise multiplication of A matrices)
    tmp_weight = tmp_weight.view(*((1, -1) + (1,) * (inp.lb.ndim - 2)))
    new_lw = torch.clamp(tmp_weight, min=0.) * inp.lw + torch.clamp(tmp_weight, max=0.) * inp.uw
    new_uw = torch.clamp(tmp_weight, min=0.) * inp.uw + torch.clamp(tmp_weight, max=0.) * inp.lw
    new_lb = torch.clamp(tmp_weight, min=0.) * inp.lb + torch.clamp(tmp_weight, max=0.) * inp.ub + tmp_bias
    new_ub = torch.clamp(tmp_weight, min=0.) * inp.ub + torch.clamp(tmp_weight, max=0.) * inp.lb + tmp_bias
    
    return LinearBound(lw=new_lw, lb=new_lb, uw=new_uw, ub=new_ub)
```

**Pattern**: For affine `y = w·x + b`:
- `new_lw = clamp(w, min=0)·inp.lw + clamp(w, max=0)·inp.uw`
- `new_lb = clamp(w, min=0)·inp.lb + clamp(w, max=0)·inp.ub + b`
- (Same for upper bounds with roles swapped)

---

## 6. Addition (Residual) Forward

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/operators/add_sub.py`  
**Method**: `bound_forward()` (lines 49–65)

**Purpose**: Merge dual-track bounds at addition nodes.

**Key excerpt** (lines 49–65):
```python
def bound_forward(self, dim_in, x, y):
    lb, ub = x.lb + y.lb, x.ub + y.ub  # Add bias terms
    
    def add_w(x_w, y_w, x_b, y_b):
        if x_w is None and y_w is None:
            return None
        elif x_w is not None and y_w is not None:
            return x_w + y_w  # Add A matrices
        elif y_w is None:
            return x_w + torch.zeros_like(y_b)
        else:
            return y_w + torch.zeros_like(x_b)
    
    lw = add_w(x.lw, y.lw, x.lb, y.lb)
    uw = add_w(x.uw, y.uw, x.ub, y.ub)
    
    return LinearBound(lw, lb, uw, ub)
```

**Semantics**: Simple element-wise addition of both A matrices and bias vectors.

---

## 7. Final Bound Extraction (Concretization)

**File**: `/home/guanqinzhang/data/guanqin/newACT/alpha-beta-CROWN/auto_LiRPA/auto_LiRPA/concretize_bounds.py`  
**Function**: `forward_concretize()` (lines 290–349)

**Purpose**: Extract concrete bounds from final linear coefficients and input perturbation set.

**Key excerpt** (lines 290–336):
```python
def forward_concretize(self, lower, upper, lw, uw, use_constraints=False, ...):
    """
    Extract concrete bounds from linear coefficients and input box.
    
    :param lower, upper: Intermediate layer bounds (bias terms)
    :param lw, uw: Linear coefficient matrices [batch, in_dim, out_dim]
    """
    res_lower = 0.0
    res_upper = 0.0
    
    # Reshape A matrices for batch processing
    lA = lw.reshape(self.batch_size, self.dim_in, -1).transpose(1, 2)  # [B, out_dim, in_dim]
    uA = uw.reshape(self.batch_size, self.dim_in, -1).transpose(1, 2)
    
    for root in self.roots():  # Iterate over input perturbation nodes
        _lA = lA[:, :, prev_dim_in : (prev_dim_in + root.dim)]
        _uA = uA[:, :, prev_dim_in : (prev_dim_in + root.dim)]
        
        # Concretize using input perturbation (e.g., Lp-norm ball)
        temp_lower = root.perturbation.concretize(
            root.center, _lA, sign=-1, aux=root.aux
        ).view(lower.shape)
        temp_upper = root.perturbation.concretize(
            root.center, _uA, sign=+1, aux=root.aux
        ).view(upper.shape)
        
        res_lower += temp_lower
        res_upper += temp_upper
    
    return res_lower, res_upper
```

**Semantics** (for L∞ box `[x_L, x_U]`):
- Split A matrices by sign: `A_pos = clamp(A, min=0)`, `A_neg = clamp(A, max=0)`
- **Lower bound**: `concrete_lb = A_pos @ x_L + A_neg @ x_U + b_lb`
- **Upper bound**: `concrete_ub = A_pos @ x_U + A_neg @ x_L + b_ub`

---

## Summary: CROWN Forward Flow

1. **Input**: Network + input perturbation set (e.g., L∞ box)
2. **Forward pass**: Recursively call `bound_forward()` on each node
   - ReLU: Compute α (lower slope), upper slope/intercept → dual-track A matrices
   - Conv2D: Apply F.conv2d to A matrices directly (no reset!)
   - Affine (BN/bias): Compose with A matrices via sign-split
   - Add: Merge A matrices element-wise
3. **Output**: Final `(lw, lb, uw, ub)` at network output
4. **Concretization**: Extract concrete bounds using input perturbation set

**Key difference from interval bounds**: A matrices are maintained end-to-end, enabling tighter bounds through CNNs.

