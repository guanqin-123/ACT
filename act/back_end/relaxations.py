"""Opt-in ReLU descriptions, dispatched by domain outside the numerical kernels.

All bounds and alpha tensors are live references. Consumers copy what they keep.
Backward dual lower slopes may have a specification axis [B, M, n].
"""

import torch

from act.back_end import observe as obs


def describe_interval(key, pre, post, **kwargs):
    return obs.Relaxation(
        key, obs.relu_status(pre.lb, pre.ub), "box",
        extra={"pre_lb": pre.lb, "pre_ub": pre.ub, "lb": post.lb, "ub": post.ub},
    )


def describe_hybridz(key, pre, post, *, tf):
    pre, extra, exact = tf._observed_relu
    return obs.Relaxation(
        key, obs.relu_status(pre.lb, pre.ub), "exact_split" if exact else "box",
        extra={"pre_lb": pre.lb, "pre_ub": pre.ub, "lb": post.lb, "ub": post.ub, **extra},
    )


def describe_dual(key, pre, post=None, *, alpha=None, **kwargs):
    lb = pre.lb.flatten(1)
    ub = pre.__dict__.get("_repaired_ub", pre.ub.flatten(1))
    status = obs.relu_status(lb, ub)
    unstable = (lb < 0) & (ub > 0)
    chord = ub / (ub - lb).clamp(min=1e-12)
    upper = torch.where(unstable, chord, (lb >= 0).to(lb))
    upper = torch.where(ub <= 0, torch.zeros_like(upper), upper)
    intercept = torch.where(unstable, -upper * lb, torch.zeros_like(lb))
    lower = upper if alpha is None else alpha
    on, off = lb >= 0, ub <= 0
    if lower.dim() == 3:
        on, off = on.unsqueeze(1), off.unsqueeze(1)
    lower = torch.where(off, torch.zeros_like(lower), torch.where(on, torch.ones_like(lower), lower))
    return obs.Relaxation(
        key, status, "linear", lower, torch.zeros_like(lower), upper, intercept,
        {"pre_lb": lb, "pre_ub": ub},
    )


DESCRIBERS = {
    "interval": describe_interval,
    "hybridz": describe_hybridz,
    "dual": describe_dual,
}
