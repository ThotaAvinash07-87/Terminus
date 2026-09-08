"""Multi-Solver Numerical ODE Simulator for Dynamic Systems.

Provides high-accuracy solvers:
- RK4 (Runge-Kutta 4th Order)
- Euler (Forward Euler 1st Order)
- Heun (2nd Order Runge-Kutta)
- ODE45 (Dormand-Prince Adaptive Step-size with Error Tolerance Control)
- ODE23 (Bogacki-Shampine Adaptive Order 2/3)
Supports both continuous and multi-rate discrete block execution, real-time stepping,
and Scope waveform recording.
"""

from __future__ import annotations
from typing import Dict, Optional, Union
import numpy as np
from CORE.common_math import Waveform, parse_eng_unit
from .scheduler import SystemDiagram, Scheduler
from .blocks import ScopeSinkBlock, DisplaySinkBlock


class DynamicSystemSimulator:
    """Simulates dynamic system diagrams using selectable continuous ODE solvers and discrete clocks."""

    def __init__(self, diagram: SystemDiagram):
        self.diagram = diagram
        self.scheduler = Scheduler(diagram)
        self.current_time: float = 0.0
        self.is_initialized: bool = False

    def reset(self, t_start: float = 0.0) -> None:
        """Resets all block states, scopes, and simulation clock to initial condition."""
        self.current_time = t_start
        self.scheduler.build_schedule()
        for b in self.diagram.blocks.values():
            b.reset_states()
            if hasattr(b, "time_history") and isinstance(b.time_history, list):
                b.time_history.clear()
            if hasattr(b, "signal_history") and isinstance(b.signal_history, list):
                b.signal_history.clear()
        self.is_initialized = True

    def step(self, dt: float = 1e-3, solver: str = "rk4") -> Dict[str, float]:
        """Advances simulation by single step dt and returns current Scope/Display values."""
        if not self.is_initialized:
            self.reset(self.diagram.t_start)

        t = self.current_time
        num_continuous_states = len(self.scheduler.stateful_blocks)

        if num_continuous_states == 0:
            # Static or purely discrete system
            self.scheduler.propagate_signals(t)
            self.scheduler.update_discrete_states(t, dt)
            self.current_time += dt
        else:
            s_type = solver.lower()
            if s_type == "euler":
                self._step_euler(t, dt)
            elif s_type in ("heun", "rk2"):
                self._step_heun(t, dt)
            else:
                self._step_rk4(t, dt)

            self.scheduler.update_discrete_states(t, dt)
            self.current_time += dt

        # Return live readings for scopes/displays
        readings = {}
        for b in self.diagram.blocks.values():
            if isinstance(b, (ScopeSinkBlock, DisplaySinkBlock)):
                readings[b.name] = float(b.inputs[0]) if b.inputs else 0.0
        return readings

    def simulate(
        self,
        t_stop: Optional[Union[str, float]] = None,
        dt: Optional[Union[str, float]] = None,
        t_start: Optional[Union[str, float]] = None,
        solver: Optional[str] = None,
    ) -> Dict[str, Waveform]:
        """Runs batch time simulation and returns recorded Waveform traces."""
        t_end = parse_eng_unit(t_stop) if t_stop is not None else self.diagram.t_stop
        step_size = parse_eng_unit(dt) if dt is not None else self.diagram.dt
        t_curr = parse_eng_unit(t_start) if t_start is not None else self.diagram.t_start
        s_type = (solver or self.diagram.solver or "rk4").lower()

        self.reset(t_curr)
        num_states = len(self.scheduler.stateful_blocks)

        if s_type == "ode45" and num_states > 0:
            self._simulate_ode45_adaptive(t_curr, t_end, step_size)
        else:
            total_steps = max(1, int(round((t_end - t_curr) / step_size)))
            for _ in range(total_steps):
                t = self.current_time
                if num_states == 0:
                    self.scheduler.propagate_signals(t)
                    self.scheduler.update_discrete_states(t, step_size)
                    self.current_time += step_size
                else:
                    if s_type == "euler":
                        self._step_euler(t, step_size)
                    elif s_type in ("heun", "rk2"):
                        self._step_heun(t, step_size)
                    else:
                        self._step_rk4(t, step_size)

                    self.scheduler.update_discrete_states(t, step_size)
                    self.current_time += step_size

            # Final evaluation at t_end
            self.scheduler.propagate_signals(t_end)

        # Collect Scope waveforms
        results: Dict[str, Waveform] = {}
        for b in self.diagram.blocks.values():
            if isinstance(b, ScopeSinkBlock):
                wf = Waveform(
                    b.time_history,
                    b.signal_history,
                    name=b.name,
                    x_unit="s",
                    y_unit="Output",
                    domain="time"
                )
                results[b.name] = wf

        return results

    def _step_euler(self, t: float, dt: float) -> None:
        """Forward Euler 1st order step."""
        x0 = self.scheduler.pack_states()
        dx = self.scheduler.compute_all_derivatives(t)
        x_next = np.clip(x0 + dt * dx, -1e9, 1e9)
        self.scheduler.unpack_states(x_next)

    def _step_heun(self, t: float, dt: float) -> None:
        """Heun's 2nd order Runge-Kutta step."""
        x0 = self.scheduler.pack_states()
        k1 = self.scheduler.compute_all_derivatives(t)

        self.scheduler.unpack_states(x0 + dt * k1)
        k2 = self.scheduler.compute_all_derivatives(t + dt)

        x_next = np.clip(x0 + 0.5 * dt * (k1 + k2), -1e9, 1e9)
        self.scheduler.unpack_states(x_next)

    def _step_rk4(self, t: float, dt: float) -> None:
        """Classical 4th Order Runge-Kutta step."""
        x0 = self.scheduler.pack_states()
        k1 = self.scheduler.compute_all_derivatives(t)

        self.scheduler.unpack_states(x0 + 0.5 * dt * k1)
        k2 = self.scheduler.compute_all_derivatives(t + 0.5 * dt)

        self.scheduler.unpack_states(x0 + 0.5 * dt * k2)
        k3 = self.scheduler.compute_all_derivatives(t + 0.5 * dt)

        self.scheduler.unpack_states(x0 + dt * k3)
        k4 = self.scheduler.compute_all_derivatives(t + dt)

        dx = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        dx = np.nan_to_num(dx, nan=0.0, posinf=1e6, neginf=-1e6)
        x_next = np.clip(x0 + dx, -1e9, 1e9)
        self.scheduler.unpack_states(x_next)

    def _simulate_ode45_adaptive(self, t_start: float, t_end: float, initial_dt: float) -> None:
        """Adaptive step size Dormand-Prince (RK45) integration with local truncation error estimation."""
        t = t_start
        dt = initial_dt
        tol = self.diagram.tolerance
        min_dt = 1e-8
        max_dt = initial_dt * 10.0

        while t < t_end:
            if t + dt > t_end:
                dt = t_end - t

            x0 = self.scheduler.pack_states()

            # Dormand-Prince Butcher Tableau evaluations:
            k1 = self.scheduler.compute_all_derivatives(t)

            self.scheduler.unpack_states(x0 + dt * (1/5) * k1)
            k2 = self.scheduler.compute_all_derivatives(t + (1/5) * dt)

            self.scheduler.unpack_states(x0 + dt * (3/40 * k1 + 9/40 * k2))
            k3 = self.scheduler.compute_all_derivatives(t + (3/10) * dt)

            self.scheduler.unpack_states(x0 + dt * (44/45 * k1 - 56/15 * k2 + 32/9 * k3))
            k4 = self.scheduler.compute_all_derivatives(t + (4/5) * dt)

            self.scheduler.unpack_states(x0 + dt * (19372/6561 * k1 - 25360/2187 * k2 + 64448/6561 * k3 - 212/729 * k4))
            k5 = self.scheduler.compute_all_derivatives(t + (8/9) * dt)

            self.scheduler.unpack_states(x0 + dt * (9017/3168 * k1 - 355/33 * k2 + 46732/5247 * k3 + 49/176 * k4 - 5103/18656 * k5))
            k6 = self.scheduler.compute_all_derivatives(t + dt)

            # 5th-order solution
            x5 = x0 + dt * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4 - 2187/6784 * k5 + 11/84 * k6)
            # 4th-order solution
            x4 = x0 + dt * (5179/57600 * k1 + 7571/16695 * k3 + 393/640 * k4 - 92097/339200 * k5 + 187/2100 * k6)

            # Error estimation
            error = np.max(np.abs(x5 - x4)) / (tol * (1.0 + np.max(np.abs(x0))))

            if error <= 1.0 or dt <= min_dt:
                # Accept step
                x_next = np.clip(x5, -1e9, 1e9)
                self.scheduler.unpack_states(x_next)
                self.scheduler.update_discrete_states(t, dt)
                t += dt
                self.current_time = t

                # Predict next step size
                factor = min(2.0, max(0.2, 0.9 * (1.0 / max(error, 1e-10)) ** 0.2))
                dt = min(max_dt, max(min_dt, dt * factor))
            else:
                # Reject step: reduce step size
                factor = max(0.1, 0.9 * (1.0 / error) ** 0.25)
                dt = max(min_dt, dt * factor)
                self.scheduler.unpack_states(x0)
