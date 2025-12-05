"""
Taichi-accelerated GPU solver for shell cascade models.

This module provides GPU-accelerated implementations using Taichi,
targeting >10⁶ steps/second performance for Phase 1 requirements.

Taichi enables:
- Just-in-time compilation to GPU (CUDA/Metal/Vulkan)
- Automatic parallelization
- Minimal Python overhead
"""

from typing import Optional, Tuple
import numpy as np

try:
    import taichi as ti
    TAICHI_AVAILABLE = True
except ImportError:
    TAICHI_AVAILABLE = False
    print("Warning: Taichi not available. GPU acceleration disabled.")


if TAICHI_AVAILABLE:
    # Initialize Taichi (will auto-select GPU if available)
    ti.init(arch=ti.gpu, default_fp=ti.f64)


    @ti.data_oriented
    class TaichiCascadeSolver:
        """
        GPU-accelerated shell cascade solver using Taichi.

        Performance target: >10⁶ steps/second on modern GPU

        Attributes
        ----------
        n_shells : int
            Number of cascade shells
        nu : ti.field(float)
            Viscosity
        k0 : float
            Fundamental wavenumber
        lambda_k : float
            Wavenumber ratio

        Fields (GPU memory):
        - u_real, u_imag: Shell velocities (complex as two real fields)
        - k: Wavenumbers
        - forcing: External forcing
        """

        def __init__(
            self,
            n_shells: int = 8,
            nu: float = 1e-3,
            k0: float = 1.0,
            lambda_k: float = 2.0,
            forcing_amplitude: float = 0.1,
            forcing_shell: int = 1,
            epsilon: float = 1.0,
        ):
            """Initialize GPU solver."""
            self.n_shells = n_shells
            self.nu_val = nu
            self.k0_val = k0
            self.lambda_k_val = lambda_k
            self.epsilon_val = epsilon

            # Allocate GPU fields
            self.u_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.u_imag = ti.field(dtype=ti.f64, shape=n_shells)
            self.k = ti.field(dtype=ti.f64, shape=n_shells)
            self.forcing_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.forcing_imag = ti.field(dtype=ti.f64, shape=n_shells)

            # Temporary storage for RK4
            self.k1_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.k1_imag = ti.field(dtype=ti.f64, shape=n_shells)
            self.k2_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.k2_imag = ti.field(dtype=ti.f64, shape=n_shells)
            self.k3_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.k3_imag = ti.field(dtype=ti.f64, shape=n_shells)
            self.k4_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.k4_imag = ti.field(dtype=ti.f64, shape=n_shells)
            self.temp_real = ti.field(dtype=ti.f64, shape=n_shells)
            self.temp_imag = ti.field(dtype=ti.f64, shape=n_shells)

            # Scalars
            self.nu = ti.field(dtype=ti.f64, shape=())
            self.epsilon = ti.field(dtype=ti.f64, shape=())

            # Initialize
            self.nu[None] = nu
            self.epsilon[None] = epsilon
            self._init_wavenumbers()
            self._init_forcing(forcing_amplitude, forcing_shell)

        @ti.kernel
        def _init_wavenumbers(self):
            """Initialize wavenumber array on GPU."""
            for n in range(self.n_shells):
                self.k[n] = self.k0_val * ti.pow(self.lambda_k_val, ti.f64(n))

        @ti.kernel
        def _init_forcing(self, amplitude: ti.f64, shell: ti.i32):
            """Initialize forcing on GPU."""
            for n in range(self.n_shells):
                if n == shell:
                    self.forcing_real[n] = amplitude
                    self.forcing_imag[n] = 0.0
                else:
                    self.forcing_real[n] = 0.0
                    self.forcing_imag[n] = 0.0

        @ti.func
        def interaction_term(
            self,
            u_r: ti.template(),
            u_i: ti.template(),
            n: ti.i32,
        ) -> ti.types.vector(2, ti.f64):
            """
            Compute nonlinear interaction F_n(u) for shell n.

            Returns [real, imag] components.
            """
            F_r = 0.0
            F_i = 0.0

            # Forward interaction: k_{n+1} u_{n+1}* u_{n+2}
            if n + 2 < self.n_shells:
                # Complex conjugate: u* = (a - ib)
                u_conj_r = u_r[n+1]
                u_conj_i = -u_i[n+1]

                # Product: u* w = (a-ib)(c+id) = (ac+bd) + i(ad-bc)
                prod_r = u_conj_r * u_r[n+2] - u_conj_i * u_i[n+2]
                prod_i = u_conj_r * u_i[n+2] + u_conj_i * u_r[n+2]

                F_r += self.k[n+1] * prod_r
                F_i += self.k[n+1] * prod_i

            # Mixed interaction: ε k_{n+1} u_{n-1}* u_{n+1}
            if n-1 >= 0 and n+1 < self.n_shells:
                u_conj_r = u_r[n-1]
                u_conj_i = -u_i[n-1]

                prod_r = u_conj_r * u_r[n+1] - u_conj_i * u_i[n+1]
                prod_i = u_conj_r * u_i[n+1] + u_conj_i * u_r[n+1]

                F_r += self.epsilon[None] * self.k[n+1] * prod_r
                F_i += self.epsilon[None] * self.k[n+1] * prod_i

            # Backward interaction: (1-ε) k_{n-1} u_{n-1}* u_{n-2}
            if n-2 >= 0:
                u_conj_r = u_r[n-1]
                u_conj_i = -u_i[n-1]

                prod_r = u_conj_r * u_r[n-2] - u_conj_i * u_i[n-2]
                prod_i = u_conj_r * u_i[n-2] + u_conj_i * u_r[n-2]

                factor = 1.0 - self.epsilon[None]
                F_r += factor * self.k[n-1] * prod_r
                F_i += factor * self.k[n-1] * prod_i

            return ti.Vector([F_r, F_i])

        @ti.kernel
        def compute_rhs(
            self,
            u_r: ti.template(),
            u_i: ti.template(),
            dudt_r: ti.template(),
            dudt_i: ti.template(),
        ):
            """
            Compute du/dt = i k F(u) - ν k² u + f.

            Since u is complex, stored as (u_real, u_imag).
            """
            for n in range(self.n_shells):
                # Nonlinear term
                F = self.interaction_term(u_r, u_i, n)
                F_r = F[0]
                F_i = F[1]

                # i k F = i k (F_r + i F_i) = -k F_i + i k F_r
                nonlinear_r = -self.k[n] * F_i
                nonlinear_i = self.k[n] * F_r

                # Viscous term: -ν k² u
                viscous_r = -self.nu[None] * self.k[n]**2 * u_r[n]
                viscous_i = -self.nu[None] * self.k[n]**2 * u_i[n]

                # Total RHS
                dudt_r[n] = nonlinear_r + viscous_r + self.forcing_real[n]
                dudt_i[n] = nonlinear_i + viscous_i + self.forcing_imag[n]

        @ti.kernel
        def update_temp(self, dt: ti.f64, stage: ti.i32):
            """Update temporary state for RK4 stage."""
            for n in range(self.n_shells):
                if stage == 2:
                    self.temp_real[n] = self.u_real[n] + 0.5 * dt * self.k1_real[n]
                    self.temp_imag[n] = self.u_imag[n] + 0.5 * dt * self.k1_imag[n]
                elif stage == 3:
                    self.temp_real[n] = self.u_real[n] + 0.5 * dt * self.k2_real[n]
                    self.temp_imag[n] = self.u_imag[n] + 0.5 * dt * self.k2_imag[n]
                elif stage == 4:
                    self.temp_real[n] = self.u_real[n] + dt * self.k3_real[n]
                    self.temp_imag[n] = self.u_imag[n] + dt * self.k3_imag[n]

        @ti.kernel
        def update_state(self, dt: ti.f64):
            """Final update for RK4."""
            for n in range(self.n_shells):
                self.u_real[n] += dt / 6.0 * (
                    self.k1_real[n] + 2*self.k2_real[n] + 2*self.k3_real[n] + self.k4_real[n]
                )
                self.u_imag[n] += dt / 6.0 * (
                    self.k1_imag[n] + 2*self.k2_imag[n] + 2*self.k3_imag[n] + self.k4_imag[n]
                )

        def rk4_step(self, dt: float):
            """
            Single RK4 timestep (Python-orchestrated).

            Calls GPU kernels for each stage.
            """
            # Stage 1: k1 = f(u)
            self.compute_rhs(self.u_real, self.u_imag, self.k1_real, self.k1_imag)

            # Stage 2: k2 = f(u + dt/2 * k1)
            self.update_temp(dt, 2)
            self.compute_rhs(self.temp_real, self.temp_imag, self.k2_real, self.k2_imag)

            # Stage 3: k3 = f(u + dt/2 * k2)
            self.update_temp(dt, 3)
            self.compute_rhs(self.temp_real, self.temp_imag, self.k3_real, self.k3_imag)

            # Stage 4: k4 = f(u + dt * k3)
            self.update_temp(dt, 4)
            self.compute_rhs(self.temp_real, self.temp_imag, self.k4_real, self.k4_imag)

            # Update: u += dt/6 * (k1 + 2k2 + 2k3 + k4)
            self.update_state(dt)

        def set_state(self, u: np.ndarray):
            """Set state from numpy array (complex)."""
            self.u_real.from_numpy(u.real)
            self.u_imag.from_numpy(u.imag)

        def get_state(self) -> np.ndarray:
            """Get state as numpy array (complex)."""
            u_r = self.u_real.to_numpy()
            u_i = self.u_imag.to_numpy()
            return u_r + 1j * u_i

        def integrate(
            self,
            u0: np.ndarray,
            n_steps: int,
            dt: float = 1e-3,
        ) -> np.ndarray:
            """
            Integrate for n_steps with GPU acceleration.

            Parameters
            ----------
            u0 : ndarray, complex
                Initial condition
            n_steps : int
                Number of steps
            dt : float
                Timestep

            Returns
            -------
            u_final : ndarray, complex
                Final state

            Performance:
                Target: >10⁶ steps/sec on GPU
            """
            # Upload initial condition
            self.set_state(u0)

            # Time stepping loop (GPU kernel calls)
            for step in range(n_steps):
                self.rk4_step(dt)

            # Download result
            return self.get_state()

        @ti.kernel
        def compute_energy(self) -> ti.f64:
            """Compute total energy on GPU."""
            energy = 0.0
            for n in range(self.n_shells):
                energy += 0.5 * (self.u_real[n]**2 + self.u_imag[n]**2)
            return energy


def benchmark_taichi_solver(
    n_shells: int = 8,
    n_steps: int = 100000,
    dt: float = 1e-3,
) -> dict:
    """
    Benchmark Taichi solver performance.

    Returns
    -------
    stats : dict
        Performance statistics including steps/sec
    """
    if not TAICHI_AVAILABLE:
        return {"error": "Taichi not available"}

    from time import time

    # Create solver
    solver = TaichiCascadeSolver(n_shells=n_shells)

    # Initial condition
    u0 = np.random.randn(n_shells) + 1j * np.random.randn(n_shells)
    u0 *= 0.1

    # Warm-up
    solver.integrate(u0, n_steps=100, dt=dt)

    # Benchmark
    start = time()
    u_final = solver.integrate(u0, n_steps=n_steps, dt=dt)
    elapsed = time() - start

    steps_per_sec = n_steps / elapsed

    stats = {
        "n_shells": n_shells,
        "n_steps": n_steps,
        "wall_time": elapsed,
        "steps_per_sec": steps_per_sec,
        "target_met": steps_per_sec > 1e6,
        "final_energy": 0.5 * np.sum(np.abs(u_final)**2),
    }

    return stats
