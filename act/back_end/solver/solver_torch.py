
from __future__ import annotations
import math, time
from typing import List, Optional, Tuple
import numpy as np
import torch

from act.back_end.solver.solver_base import Solver, SolveStatus, SolverCaps
from act.util.device_manager import get_default_device, get_default_dtype

class TorchLPSolver(Solver):
    """Continuous LP solver using Torch + Adam with penalty and box projection.

    - Supports GPU via device hint in begin(...).
    - No integrality: add_binary_vars() creates [0,1] continuous vars.
    - add_sos2 is a no-op.
    """
    def __init__(self):
        self._device = get_default_device()
        self._dtype = get_default_dtype()
        self._n = 0
        self._x = None                 # torch.nn.Parameter
        self._lb = None                # torch.Tensor [n]
        self._ub = None                # torch.Tensor [n]
        self._eq = []                  # rows: (vids, coeffs, rhs)
        self._le = []
        self._ge = []
        self._objective = ([], [], 0.0, "min")
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        self._sol = None
        self._timelimit = None

        # parameters
        self.rho_eq = 10.0
        self.rho_ineq = 10.0
        self.max_iter = 5000
        self.tol_feas = 1e-6
        self.lr = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.weight_decay = 0.0

    def capabilities(self) -> SolverCaps:
        return SolverCaps(supports_gpu=True)

    @property
    def n(self) -> int:
        return self._n

    def begin(self, name: str = "verify", device: Optional[str] = None):
        # Use global device manager for default, allow override
        if device is not None:
            self._device = torch.device(device)
        # else keep the device_manager default from __init__
        
        self._n = 0
        self._x = None
        self._lb = None
        self._ub = None
        self._eq.clear(); self._le.clear(); self._ge.clear()
        self._objective = ([], [], 0.0, "min")
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False
        self._sol = None

    def add_vars(self, n: int) -> None:
        if n <= 0:
            return
        if self._n == 0:
            self._n = n
            # Create tensors on the correct device and dtype
            self._lb = torch.full((n,), -np.inf, device=self._device, dtype=self._dtype)
            self._ub = torch.full((n,), +np.inf, device=self._device, dtype=self._dtype)
        else:
            old_n = self._n
            self._n += n
            # Extend tensors on the correct device and dtype
            self._lb = torch.cat([self._lb, torch.full((n,), -np.inf, device=self._device, dtype=self._dtype)])
            self._ub = torch.cat([self._ub, torch.full((n,), +np.inf, device=self._device, dtype=self._dtype)])

    def add_binary_vars(self, n: int) -> List[int]:
        start = self._n
        self.add_vars(n)
        idxs = list(range(start, start + n))
        # relax to [0,1]
        self._lb[idxs] = 0.0
        self._ub[idxs] = 1.0
        return idxs

    def set_bounds(self, idxs: List[int], lb: np.ndarray, ub: np.ndarray) -> None:
        # Convert to tensors with correct device and dtype
        lb_t = torch.as_tensor(lb, device=self._device, dtype=self._dtype)
        ub_t = torch.as_tensor(ub, device=self._device, dtype=self._dtype)
        self._lb[idxs] = lb_t
        self._ub[idxs] = ub_t

    def add_lin_eq(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._eq.append((vids, coeffs, rhs))

    def add_lin_le(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        self._le.append((vids, coeffs, rhs))

    def add_lin_ge(self, vids: List[int], coeffs: List[float], rhs: float) -> None:
        vids2 = vids
        coeffs2 = [-float(a) for a in coeffs]
        rhs2 = -float(rhs)
        self._le.append((vids2, coeffs2, rhs2))

    def add_sum_eq(self, vids: List[int], rhs: float) -> None:
        coeffs = [1.0] * len(vids)
        self.add_lin_eq(vids, coeffs, rhs)

    def add_ge_zero(self, vids: List[int]) -> None:
        for i in vids:
            self.add_lin_le([i], [-1.0], 0.0)

    def add_sos2(self, var_ids: List[int], weights: Optional[List[float]] = None) -> None:
        return  # no-op

    def set_objective_linear(self, vids: List[int], coeffs: List[float], const: float = 0.0, sense: str = "min") -> None:
        self._objective = (vids, coeffs, float(const), "min" if sense != "max" else "max")

    def optimize(self, timelimit: Optional[float] = None) -> None:
        self._timelimit = timelimit

        # initialize x at box center (or zeros where infinite)
        if self._x is None:
            lb = torch.where(torch.isfinite(self._lb), self._lb, torch.zeros_like(self._lb))
            ub = torch.where(torch.isfinite(self._ub), self._ub, torch.zeros_like(self._ub))
            mid = 0.5 * (lb + ub)
            both_inf = (~torch.isfinite(self._lb)) & (~torch.isfinite(self._ub))
            mid = torch.where(both_inf, torch.zeros_like(mid), mid)
            self._x = torch.nn.Parameter(mid.clone().to(device=self._device, dtype=self._dtype), requires_grad=True)

        # CRITICAL: Re-enable gradients after any potential parameter manipulation
        self._x.requires_grad_(True)

        opt = torch.optim.Adam([self._x], lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)

        def rows_to_dense(rows):
            if not rows:
                return torch.zeros((0, self._n), device=self._device, dtype=self._dtype), \
                       torch.zeros((0,), device=self._device, dtype=self._dtype)
            A = torch.zeros((len(rows), self._n), device=self._device, dtype=self._dtype)
            b = torch.zeros((len(rows),), device=self._device, dtype=self._dtype)
            for r, (vids, coeffs, rhs) in enumerate(rows):
                A[r, torch.as_tensor(vids, device=self._device)] = torch.as_tensor(coeffs, device=self._device, dtype=self._dtype)
                b[r] = float(rhs)
            return A, b

        Aeq, beq = rows_to_dense(self._eq)
        Ale, ble = rows_to_dense(self._le)

        vids, coeffs, c0, sense = self._objective
        c = torch.zeros((self._n,), device=self._device, dtype=self._dtype, requires_grad=False)
        if vids:
            c[torch.as_tensor(vids, device=self._device)] = torch.as_tensor(coeffs, device=self._device, dtype=self._dtype)

        t_end = None if timelimit is None else (time.time() + float(timelimit))
        self._status = SolveStatus.UNKNOWN
        self._has_solution = False

        for it in range(self.max_iter):
            opt.zero_grad()

            # SOLUTION: Force gradient computation context
            with torch.enable_grad():
                # Create objective function components
                obj = 0.001 * torch.sum(self._x**2)  # Small regularizer
                
                # Add linear objective
                if vids and len(coeffs) > 0:
                    for vid, coeff in zip(vids, coeffs):
                        obj = obj + float(coeff) * self._x[vid]
                
                # Add constant term
                obj = obj + float(c0)
                
                if sense == "max":
                    obj = -obj

                # Add constraint penalties
                if Aeq.shape[0] > 0:
                    # Ensure constraint matrices are on same device
                    Aeq_device = Aeq.to(device=self._x.device, dtype=self._x.dtype)
                    beq_device = beq.to(device=self._x.device, dtype=self._x.dtype)
                    for i in range(Aeq.shape[0]):
                        violation = torch.sum(Aeq_device[i] * self._x) - beq_device[i]
                        obj = obj + self.rho_eq * violation * violation

                if Ale.shape[0] > 0:
                    # Ensure constraint matrices are on same device
                    Ale_device = Ale.to(device=self._x.device, dtype=self._x.dtype)
                    ble_device = ble.to(device=self._x.device, dtype=self._x.dtype)
                    for i in range(Ale.shape[0]):
                        violation = torch.sum(Ale_device[i] * self._x) - ble_device[i]
                        obj = obj + self.rho_ineq * torch.relu(violation)**2

                # Perform backward pass within gradient context
                obj.backward()
            
            opt.step()

            # Project to box constraints while preserving Parameter status
            with torch.no_grad():
                # Ensure bounds are on the same device as _x
                lb_device = self._lb.to(device=self._x.device, dtype=self._x.dtype)
                ub_device = self._ub.to(device=self._x.device, dtype=self._x.dtype)
                x_clamped = torch.minimum(torch.maximum(self._x, lb_device), ub_device)
                self._x.data.copy_(x_clamped)  # Use .data.copy_ to preserve Parameter wrapper
                # Ensure gradients are still enabled after projection
                if not self._x.requires_grad:
                    self._x.requires_grad_(True)

            max_viol = 0.0
            with torch.no_grad():
                if Aeq.shape[0] > 0:
                    # Ensure matrix operations are on same device
                    Aeq_device = Aeq.to(device=self._x.device, dtype=self._x.dtype)
                    beq_device = beq.to(device=self._x.device, dtype=self._x.dtype)
                    max_viol = max(max_viol, float(torch.max(torch.abs(Aeq_device @ self._x - beq_device)).item()))
                if Ale.shape[0] > 0:
                    # Ensure matrix operations are on same device
                    Ale_device = Ale.to(device=self._x.device, dtype=self._x.dtype)
                    ble_device = ble.to(device=self._x.device, dtype=self._x.dtype)
                    max_viol = max(max_viol, float(torch.max(torch.relu(Ale_device @ self._x - ble_device)).item()))

            if max_viol <= self.tol_feas:
                self._status = SolveStatus.SAT
                self._has_solution = True
                self._sol = self._x.detach().clone()
                break

            if t_end is not None and time.time() >= t_end:
                self._status = SolveStatus.SAT if math.isfinite(max_viol) else SolveStatus.UNKNOWN
                self._has_solution = True
                self._sol = self._x.detach().clone()
                break
        else:
            self._status = SolveStatus.SAT
            self._has_solution = True
            self._sol = self._x.detach().clone()

    def status(self) -> str:
        return self._status

    def has_solution(self) -> bool:
        return bool(self._has_solution)

    def get_values(self, vids: List[int]) -> np.ndarray:
        assert self._sol is not None, "No solution available"
        with torch.no_grad():
            return self._sol[vids].detach().cpu().to(self._dtype).numpy()

    def get_counterexample(self, input_ids: List[int]) -> np.ndarray:
        # Already projected to box during optimize(); can return directly.
        return self.get_values(input_ids)
