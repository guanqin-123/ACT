# Provable Loss: 凸松弛与对偶方法

## 📚 文献来源

**主要论文**: Wong & Kolter (2018)
> **"Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope"**  
> ICML 2018  
> https://arxiv.org/abs/1711.00851

---

## 🔢 数学原理

### 1. 问题设定

对于分类问题，我们想要证明在 $\ell_\infty$ 扰动下模型的鲁棒性：

$$\min_{x' \in B_\infty(x, \epsilon)} f_y(x') - f_j(x') > 0, \quad \forall j \neq y$$

其中：
- $x$ 是原始输入
- $y$ 是真实标签  
- $f_k(x)$ 是网络对类别 $k$ 的输出（logit）
- $B_\infty(x, \epsilon) = \{x' : \|x' - x\|_\infty \leq \epsilon\}$

**如果这个最小值 > 0，则该样本是"可证明鲁棒"的。**

---

### 2. 凸松弛 (Convex Relaxation)

直接求解上述优化问题是 NP-hard 的（因为 ReLU 网络是非凸的）。

Wong & Kolter 的关键思想是用**凸松弛**来获得一个**下界**：

对于 ReLU 激活函数 $\sigma(z) = \max(0, z)$，当预激活值 $z \in [l, u]$ 时：

$$\sigma(z) \geq 0$$
$$\sigma(z) \geq z$$  
$$\sigma(z) \leq \frac{u}{u-l}(z - l)$$

这形成了一个**凸包络** (convex envelope)：

```
     σ(z)
      |      /
      |     /  <- 上界线: u/(u-l) * (z-l)
      |    /
      |   /____  <- ReLU
      |  /|
      | / |
      |/  |
   ---+---+------ z
      l   0   u
```

**三种神经元状态**：
- **Active** ($l \geq 0$): ReLU 恒等，斜率 $d = 1$
- **Inactive** ($u \leq 0$): ReLU 恒零，斜率 $d = 0$  
- **Crossing** ($l < 0 < u$): 需要松弛，斜率 $d = \frac{u}{u-l}$

---

### 3. 对偶问题 (Dual Problem)

原问题（Primal）：
$$\min_{x', z} c^T z_L \quad \text{s.t. } x' \in B_\infty(x,\epsilon), \text{ 网络约束}$$

其中 $c = e_y - e_j$（目标类别减去攻击类别的 one-hot 向量）。

通过拉格朗日对偶，我们得到**对偶问题**：

$$\max_{\nu \geq 0} g(\nu)$$

对偶函数 $g(\nu)$ 提供了原问题的**下界**（弱对偶性）。

---

### 4. 对偶反向传播 (Dual Backward Pass)

设网络为 $z_{k+1} = W_k \sigma(z_k) + b_k$，目标是计算：

$$\min_{x' \in B_\infty(x,\epsilon)} c^T z_L$$

**对偶变量**: $\nu_k$ 对应第 $k$ 层的拉格朗日乘子

**反向传递规则**:

#### 4.1 线性层 $z = Wx + b$

$$\nu_{k-1} = W^T \nu_k$$
$$\text{obj} \mathrel{+}= -\nu_k^T b$$

#### 4.2 ReLU层

对于 crossing neuron ($l < 0 < u$):

- **斜率**: $d = \frac{u}{u-l}$
- **转置**: $\nu_{k-1} = d \cdot \nu_k$
- **目标贡献**: $\text{obj} \mathrel{+}= [\nu_k]_+ \cdot l$

其中 $[\cdot]_+ = \max(0, \cdot)$

#### 4.3 输入层 ($\ell_\infty$ 球)

$$\text{obj} \mathrel{+}= -[\nu]_-^T \cdot lb - [\nu]_+^T \cdot ub$$

其中：
- $lb = \max(x - \epsilon, 0)$ （下界，clamp 到有效范围）
- $ub = \min(x + \epsilon, 1)$ （上界，clamp 到有效范围）
- $[\nu]_- = \min(0, \nu)$
- $[\nu]_+ = \max(0, \nu)$

---

### 5. 最终结果

$$\text{bound} = \text{obj}$$

这个 bound 是 $c^T z_L$ 的**下界**，即：

$$c^T z_L \geq \text{bound}, \quad \forall x' \in B_\infty(x, \epsilon)$$

**判定规则**：
- **bound > 0** → 样本被证明是鲁棒的 ✓
- **bound ≤ 0** → 无法证明（可能鲁棒也可能不鲁棒）

---

## 🎯 训练中的应用

### 损失函数

在训练时，我们最小化：

$$\mathcal{L} = \text{CrossEntropy}(-\text{worst\_logits}, y)$$

其中 `worst_logits[b,j]` 是 $f_y(x) - f_j(x)$ 的下界。

### 可微分性

**关键点**：整个计算是**可微分的**，梯度可以反向传播来训练网络！

这使得我们可以端到端地训练具有可证明鲁棒性的神经网络。

---

## 📊 直观理解

```
                    Actual robust margin
                    (NP-hard to compute)
                           ↓
    ════════════════════════════════════════
    |←─────────── Gap ───────────→|
    ════════════════════════════════════════
    ↑
    Dual bound (我们计算的下界)
    
    如果 dual bound > 0 → 证明鲁棒
    如果 dual bound < 0 → 无法证明（可能鲁棒也可能不鲁棒）
```

### Gap 的来源

1. **凸松弛**: ReLU 的非凸区域被线性上界替代
2. **弱对偶性**: 对偶问题只提供下界，不一定紧

### 如何减小 Gap

1. **CROWN** (Zhang et al., 2018): 自适应选择更紧的线性边界
2. **分支定界** (Branch and Bound): 分割 crossing neurons 的区间
3. **α-CROWN**: 可学习的松弛参数

---

## 💻 代码实现概览

```python
class ProvableLoss(RobustLoss):
    def __call__(self, model, X, y, epsilon):
        # 1. 计算输入边界
        lb = (X - epsilon).clamp(min=0)
        ub = (X + epsilon).clamp(max=1)
        
        # 2. 前向传播计算各层边界 (CROWN-style)
        layer_bounds = self._forward_bounds(model, lb, ub)
        
        # 3. 对偶反向传播计算 worst-case logits
        worst_logits = self._compute_dual_bounds(model, layer_bounds, lb, ub, y)
        
        # 4. 计算损失
        loss = CrossEntropy(-worst_logits, y)
        
        return loss, metrics
```

---

## 📖 相关文献

1. **Wong & Kolter (2018)** - 原始论文，提出 LP 松弛和对偶网络
   - *Provable Defenses against Adversarial Examples via the Convex Outer Adversarial Polytope*

2. **Zhang et al. (2018) CROWN** - 更紧的线性松弛边界
   - *Efficient Neural Network Robustness Certification with General Activation Functions*

3. **Gowal et al. (2018) IBP** - 区间边界传播（更松但更快）
   - *On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models*

4. **Salman et al. (2019)** - 凸松弛障碍分析
   - *A Convex Relaxation Barrier to Tight Robustness Verification of Neural Networks*

5. **Xu et al. (2020) α-CROWN** - 可学习松弛参数
   - *Automatic Perturbation Analysis for Scalable Certified Robustness and Beyond*

---

## 🔗 ACT 实现

详见: `act/pipeline/finetune/provable.py`

主要类: `ProvableLoss`
- `_forward_bounds()`: CROWN 风格的前向边界传播
- `_compute_dual_bounds()`: 计算所有类别的对偶边界
- `_dual_backward()`: 对偶反向传播核心算法
- `_compute_certified()`: 判断样本是否可证明鲁棒
