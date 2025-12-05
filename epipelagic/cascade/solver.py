"""
Time integration and steady-state solvers for shell cascade models.

Provides:
- CascadeSolver: High-level solver interface
- Adaptive time stepping (RK4, RK45)
- Steady-state detection
- Performance monitoring
"""

from typing import Optional, Tuple, Callable, Dict
import numpy as np
from dataclasses import dataclass, field
from time import time as wall_time

from epipelagic.cascade.shell_model import ShellCascade


@dataclass
class CascadeSolver:
    """
    Time integration solver for shell cascade models.

    Attributes
    ----------
    cascade : ShellCascade
        Shell cascade model to integrate
    dt : float
        Initial timestep
    adaptive : bool
        Use adaptive timestepping
    tolerance : float
        Error tolerance for adaptive stepping
    max_steps : int
        Maximum number of steps
    """

    cascade: ShellCascade
    dt: float = 1e-3
    adaptive: bool = True
    tolerance: float = 1e-6
    max_steps: int = 1000000
    verbose: bool = False

    # Performance tracking
    stats: Dict[str, float] = field(default_factory=dict)

    def integrate(
        self,
        u0: np.ndarray,
        t_final: float,
        callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate cascade equations from t=0 to t=t_final.

        Parameters
        ----------
        u0 : ndarray, shape (n_shells,), complex
            Initial condition
        t_final : float
            Final time
        callback : callable, optional
            Function called at each timestep: callback(t, u)

        Returns
        -------
        t_history : ndarray
            Time points
        u_history : ndarray, shape (n_steps, n_shells)
            Solution history

        Examples
        --------
        >>> cascade = ShellCascade(n_shells=8)
        >>> solver = CascadeSolver(cascade)
        >>> u0 = np.random.randn(8) + 1j * np.random.randn(8)
        >>> t, u = solver.integrate(u0, t_final=10.0)
        """
        start_time = wall_time()

        # Initialize
        t = 0.0
        u = u0.copy()
        dt = self.dt

        t_history = [t]
        u_history = [u.copy()]

        steps = 0
        while t < t_final and steps < self.max_steps:
            # Take timestep
            if self.adaptive:
                u_new, dt_new, error = self._step_adaptive(u, t, dt)
                dt = dt_new
            else:
                u_new = self._step_rk4(u, t, dt)

            # Update
            t += dt
            u = u_new
            steps += 1

            # Store
            t_history.append(t)
            u_history.append(u.copy())

            # Callback
            if callback is not None:
                callback(t, u)

            # Progress
            if self.verbose and steps % 1000 == 0:
                energy = self.cascade.total_energy(u)
                print(f"Step {steps}: t={t:.3f}, E={energy:.3e}, dt={dt:.3e}")

        # Convert to arrays
        t_history = np.array(t_history)
        u_history = np.array(u_history)

        # Store statistics
        self.stats["total_steps"] = steps
        self.stats["wall_time"] = wall_time() - start_time
        self.stats["steps_per_sec"] = steps / self.stats["wall_time"]
        self.stats["final_time"] = t

        if self.verbose:
            print(f"\nIntegration complete:")
            print(f"  Total steps: {steps}")
            print(f"  Wall time: {self.stats['wall_time']:.2f} s")
            print(f"  Performance: {self.stats['steps_per_sec']:.1e} steps/sec")

        return t_history, u_history

    def _step_rk4(self, u: np.ndarray, t: float, dt: float) -> np.ndarray:
        """
        Classical 4th-order Runge-Kutta step.

        Algorithm:
            k1 = f(t, u)
            k2 = f(t + dt/2, u + dt/2 * k1)
            k3 = f(t + dt/2, u + dt/2 * k2)
            k4 = f(t + dt, u + dt * k3)
            u_new = u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        k1 = self.cascade.rhs(u, t)
        k2 = self.cascade.rhs(u + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.cascade.rhs(u + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.cascade.rhs(u + dt * k3, t + dt)

        u_new = u + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        return u_new

    def _step_adaptive(
        self,
        u: np.ndarray,
        t: float,
        dt: float,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Adaptive timestep using embedded Runge-Kutta (RK4/RK5).

        Returns
        -------
        u_new : ndarray
            Solution at t + dt
        dt_new : float
            Suggested next timestep
        error : float
            Estimated error
        """
        # Take one RK4 step
        u_rk4 = self._step_rk4(u, t, dt)

        # Take two RK4 half-steps
        u_half1 = self._step_rk4(u, t, dt/2)
        u_rk4_refined = self._step_rk4(u_half1, t + dt/2, dt/2)

        # Estimate error
        error = np.max(np.abs(u_rk4 - u_rk4_refined))

        # Adjust timestep
        if error > 0:
            dt_new = dt * min(2.0, max(0.5, 0.9 * (self.tolerance / error)**0.2))
        else:
            dt_new = dt * 2.0

        # Use more accurate solution
        u_new = u_rk4_refined

        return u_new, dt_new, error

    def find_steady_state(
        self,
        u0: np.ndarray,
        max_time: float = 1000.0,
        energy_tolerance: float = 1e-6,
        check_interval: int = 100,
    ) -> Tuple[np.ndarray, bool]:
        """
        Integrate until steady state is reached.

        Parameters
        ----------
        u0 : ndarray
            Initial condition
        max_time : float
            Maximum integration time
        energy_tolerance : float
            Relative energy change threshold for steady state
        check_interval : int
            Check convergence every N steps

        Returns
        -------
        u_steady : ndarray
            Steady-state solution
        converged : bool
            True if steady state reached

        Algorithm:
            Steady state detected when:
                |dE/dt| / E < tolerance
            for sustained period.
        """
        t = 0.0
        u = u0.copy()
        dt = self.dt

        energy_history = []
        converged = False

        steps = 0
        while t < max_time and steps < self.max_steps:
            # Step forward
            u = self._step_rk4(u, t, dt)
            t += dt
            steps += 1

            # Check convergence periodically
            if steps % check_interval == 0:
                energy = self.cascade.total_energy(u)
                energy_history.append(energy)

                if len(energy_history) >= 10:
                    # Check relative energy change
                    recent_energies = energy_history[-10:]
                    dE = np.max(recent_energies) - np.min(recent_energies)
                    E_avg = np.mean(recent_energies)

                    if E_avg > 0 and dE / E_avg < energy_tolerance:
                        converged = True
                        if self.verbose:
                            print(f"Steady state reached at t={t:.2f}")
                        break

        if self.verbose and not converged:
            print(f"Warning: Steady state not reached after t={max_time}")

        return u, converged

    def compute_attracting_state(
        self,
        n_trials: int = 10,
        t_transient: float = 100.0,
    ) -> np.ndarray:
        """
        Compute attracting state by averaging over multiple initial conditions.

        Parameters
        ----------
        n_trials : int
            Number of random initial conditions
        t_transient : float
            Transient time to discard

        Returns
        -------
        u_attract : ndarray
            Averaged attracting state

        Algorithm:
            1. Generate n_trials random initial conditions
            2. Integrate each for time t_transient
            3. Average final states
        """
        u_final_states = []

        for trial in range(n_trials):
            # Random initial condition
            u0 = (
                np.random.randn(self.cascade.n_shells)
                + 1j * np.random.randn(self.cascade.n_shells)
            )
            u0 *= 0.1  # Small amplitude

            # Integrate
            u_steady, _ = self.find_steady_state(u0, max_time=t_transient)
            u_final_states.append(u_steady)

        # Average
        u_attract = np.mean(u_final_states, axis=0)
        return u_attract
