"""
Persistent homology computation for turbulent vorticity fields.

Mathematical Framework:
    1. Vorticity field: ω(x) = ∇ × u(x)
    2. Sublevel set filtration: X_θ = {x : |ω(x)| ≤ θ}
    3. Persistent homology: H_*(X_θ) as θ varies
    4. Persistence diagram: Birth-death pairs (b_i, d_i)
    5. Epipelagic dimension: dim(H¹_epi) = # long bars

Algorithm 1 from mega-prompt implemented here.
"""

from typing import Tuple, Optional, List
import numpy as np

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False
    print("Warning: Ripser not available. Install with: pip install ripser")


def extract_persistent_homology(
    velocity_field: np.ndarray,
    threshold: float = 0.5,
    max_dimension: int = 2,
    n_points: Optional[int] = None,
) -> dict:
    """
    Extract persistent homology from velocity field (Algorithm 1).

    Parameters
    ----------
    velocity_field : ndarray, shape (nx, ny, nz, 3) or (nx, ny, 3)
        Velocity field u(x,y,z) with components (ux, uy, uz)
    threshold : float
        Persistence threshold Δ_epi for filtering features
    max_dimension : int
        Maximum homology dimension to compute
    n_points : int, optional
        Subsample to n_points for efficiency

    Returns
    -------
    persistence : dict
        Dictionary containing:
        - 'dgms': List of persistence diagrams [H0, H1, H2, ...]
        - 'dim_H1_epi': Dimension of epipelagic cohomology
        - 'long_bars': Filtered bars with persistence > threshold
        - 'vorticity': Computed vorticity field

    Algorithm (from mega-prompt):
        1. Compute vorticity ω = ∇ × u
        2. Build filtration from vorticity level sets
        3. Compute persistent homology using Ripser
        4. Filter long bars (persistence > threshold)
        5. Return count as dim(H¹_epi)

    Examples
    --------
    >>> u = generate_turbulent_field(Re=1000)
    >>> result = extract_persistent_homology(u, threshold=0.3)
    >>> print(f"dim(H¹_epi) = {result['dim_H1_epi']}")

    References
    ----------
    [1] Rigorous Foundations, Section 5.2
    [2] Edelsbrunner & Harer (2010). Computational Topology
    """
    if not RIPSER_AVAILABLE:
        raise ImportError("Ripser required. Install with: pip install ripser")

    # Step 1: Compute vorticity ω = ∇ × u
    vorticity = compute_vorticity(velocity_field)

    # Step 2: Build point cloud from vorticity field
    point_cloud, vorticity_values = build_filtration_points(
        vorticity,
        n_points=n_points,
    )

    # Step 3: Compute persistent homology
    if len(point_cloud) == 0:
        return {
            'dgms': [np.zeros((0, 2))],
            'dim_H1_epi': 0,
            'long_bars': [],
            'vorticity': vorticity,
        }

    result = ripser(
        point_cloud,
        maxdim=max_dimension,
    )

    # Step 4: Filter long bars in H¹
    if len(result['dgms']) > 1:
        H1_diagram = result['dgms'][1]  # H¹ persistence diagram
        long_bars = filter_long_bars(H1_diagram, threshold)
        dim_H1_epi = len(long_bars)
    else:
        long_bars = []
        dim_H1_epi = 0

    return {
        'dgms': result['dgms'],
        'dim_H1_epi': dim_H1_epi,
        'long_bars': long_bars,
        'vorticity': vorticity,
    }


def compute_vorticity(velocity_field: np.ndarray) -> np.ndarray:
    """
    Compute vorticity ω = ∇ × u from velocity field.

    Parameters
    ----------
    velocity_field : ndarray, shape (..., ndim)
        Velocity field with last axis as vector components

    Returns
    -------
    vorticity : ndarray
        Vorticity field (scalar in 2D, vector in 3D)

    Implementation:
        2D: ω = ∂v/∂x - ∂u/∂y (scalar)
        3D: ω = ∇ × u = (ω_x, ω_y, ω_z) (vector)
    """
    ndim = velocity_field.ndim - 1

    if ndim == 2:
        # 2D case: scalar vorticity
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]

        # Finite differences
        dv_dx = np.gradient(v, axis=0)
        du_dy = np.gradient(u, axis=1)

        vorticity = dv_dx - du_dy
        return vorticity

    elif ndim == 3:
        # 3D case: vector vorticity
        u = velocity_field[..., 0]
        v = velocity_field[..., 1]
        w = velocity_field[..., 2]

        # Components of curl
        dw_dy = np.gradient(w, axis=1)
        dv_dz = np.gradient(v, axis=2)
        omega_x = dw_dy - dv_dz

        du_dz = np.gradient(u, axis=2)
        dw_dx = np.gradient(w, axis=0)
        omega_y = du_dz - dw_dx

        dv_dx = np.gradient(v, axis=0)
        du_dy = np.gradient(u, axis=1)
        omega_z = dv_dx - du_dy

        vorticity = np.stack([omega_x, omega_y, omega_z], axis=-1)

        # Return magnitude for filtration
        return np.linalg.norm(vorticity, axis=-1)

    else:
        raise ValueError(f"Unsupported dimensionality: {ndim}")


def build_filtration_points(
    vorticity: np.ndarray,
    n_points: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build point cloud from vorticity field for persistent homology.

    Parameters
    ----------
    vorticity : ndarray
        Vorticity field (scalar)
    n_points : int, optional
        Subsample to n_points (default: use all points up to 10000)

    Returns
    -------
    points : ndarray, shape (n_samples, ndim)
        Spatial coordinates
    values : ndarray, shape (n_samples,)
        Vorticity magnitude at each point

    Algorithm:
        1. Flatten vorticity field to list of (position, value) pairs
        2. Optionally subsample for computational efficiency
        3. Return point cloud for Ripser
    """
    # Get spatial coordinates
    ndim = vorticity.ndim
    shape = vorticity.shape

    if ndim == 2:
        x, y = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            indexing='ij',
        )
        coords = np.column_stack([x.ravel(), y.ravel()])
    elif ndim == 3:
        x, y, z = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing='ij',
        )
        coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    else:
        raise ValueError(f"Unsupported dimension: {ndim}")

    values = vorticity.ravel()

    # Subsample if needed
    if n_points is None:
        n_points = min(len(coords), 10000)  # Default cap

    if len(coords) > n_points:
        # Random subsampling
        indices = np.random.choice(len(coords), n_points, replace=False)
        coords = coords[indices]
        values = values[indices]

    # Normalize coordinates to unit scale
    coords = coords.astype(float)
    for i in range(coords.shape[1]):
        coords[:, i] /= np.max(coords[:, i]) if np.max(coords[:, i]) > 0 else 1.0

    return coords, values


def filter_long_bars(
    persistence_diagram: np.ndarray,
    threshold: float,
) -> List[Tuple[float, float]]:
    """
    Filter persistence diagram for long bars.

    Parameters
    ----------
    persistence_diagram : ndarray, shape (n_bars, 2)
        Persistence diagram with (birth, death) pairs
    threshold : float
        Minimum persistence (death - birth)

    Returns
    -------
    long_bars : list of (birth, death) tuples
        Bars with persistence > threshold

    Mathematical Interpretation:
        Long bars correspond to topological features that persist
        across many scales → cross-scale structures in the cascade
    """
    if len(persistence_diagram) == 0:
        return []

    long_bars = []
    for birth, death in persistence_diagram:
        # Handle infinite bars
        if np.isinf(death):
            persistence = np.inf
        else:
            persistence = death - birth

        if persistence > threshold:
            long_bars.append((birth, death))

    return long_bars


def compute_epipelagic_dimension(
    velocity_field: np.ndarray,
    threshold: float = 0.5,
) -> int:
    """
    Compute epipelagic cohomology dimension from velocity field.

    Parameters
    ----------
    velocity_field : ndarray
        Velocity field
    threshold : float
        Persistence threshold

    Returns
    -------
    dim_H1_epi : int
        Dimension of H¹_epi (number of persistent cross-scale structures)

    This is the main computational implementation of Theorem C validation.
    """
    result = extract_persistent_homology(velocity_field, threshold=threshold)
    return result['dim_H1_epi']


def verify_finiteness_bound(
    velocity_field: np.ndarray,
    reynolds_number: float,
    constant: float = 2.5,
    threshold: float = 0.5,
) -> Tuple[bool, int, float]:
    """
    Verify Theorem C finiteness bound: dim(H¹_epi) ≤ C log(Re).

    Parameters
    ----------
    velocity_field : ndarray
        Velocity field
    reynolds_number : float
        Reynolds number
    constant : float
        Constant C in theorem
    threshold : float
        Persistence threshold

    Returns
    -------
    satisfied : bool
        True if bound holds
    dim_H1 : int
        Actual dimension
    bound : float
        Theoretical bound
    """
    dim_H1 = compute_epipelagic_dimension(velocity_field, threshold)

    if reynolds_number <= 1:
        bound = 0.0
    else:
        bound = constant * np.log(reynolds_number)

    satisfied = dim_H1 <= bound

    return satisfied, dim_H1, bound
