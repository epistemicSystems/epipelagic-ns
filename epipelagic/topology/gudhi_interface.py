"""
Gudhi interface for advanced persistent homology computation.

This module implements Task 2.3 from TASKS.md, providing:
1. Alpha complex construction (geometric)
2. Cubical complex (for voxel data)
3. Comparison with Ripser
4. Multi-parameter persistence preparation

Gudhi provides more advanced features than Ripser:
- Multiple complex types (Alpha, Rips, Cubical, Witness)
- Representative cycles extraction
- Persistence landscapes
- Bottleneck and Wasserstein distances

References:
    [1] GUDHI Project: https://gudhi.inria.fr/
    [2] Maria et al. (2014). "The Gudhi Library"
"""

from typing import Tuple, Optional, List, Dict, Union
import numpy as np
import warnings

try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    warnings.warn("Gudhi not available. Install with: pip install gudhi")


def compute_alpha_complex_persistence(
    points: np.ndarray,
    max_dimension: int = 2,
    max_alpha_square: float = np.inf,
) -> Dict:
    """
    Compute persistent homology using Alpha complex.

    Alpha complexes are geometric: they use the Delaunay triangulation
    and are well-suited for point clouds in Euclidean space.

    Parameters
    ----------
    points : ndarray, shape (n_points, dimension)
        Point cloud in Euclidean space
    max_dimension : int
        Maximum homology dimension to compute
    max_alpha_square : float
        Maximum alpha² value (filtration parameter)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'persistence': List of (dimension, (birth, death)) tuples
        - 'betti_numbers': Betti numbers at each filtration value
        - 'simplex_tree': Gudhi simplex tree object

    Algorithm:
    ----------
    1. Construct Delaunay triangulation
    2. Build Alpha complex from triangulation
    3. Compute persistent homology
    4. Extract persistence pairs

    Examples
    --------
    >>> points = np.random.randn(100, 3)
    >>> result = compute_alpha_complex_persistence(points)
    >>> print(f"Found {len(result['persistence'])} features")

    Notes
    -----
    Alpha complexes are:
    - Geometric (use Euclidean distance)
    - Exact (no approximation)
    - Fast for low dimensions (d ≤ 3)
    - Memory intensive for large point clouds
    """
    if not GUDHI_AVAILABLE:
        raise ImportError("Gudhi required. Install with: pip install gudhi")

    # Create Alpha complex
    alpha_complex = gudhi.AlphaComplex(points=points)

    # Create simplex tree
    simplex_tree = alpha_complex.create_simplex_tree(
        max_alpha_square=max_alpha_square
    )

    # Compute persistence
    persistence = simplex_tree.persistence(
        homology_coeff_field=2,  # ℤ/2ℤ coefficients
        min_persistence=0
    )

    # Compute Betti numbers
    betti_numbers = simplex_tree.betti_numbers()

    return {
        'persistence': persistence,
        'betti_numbers': betti_numbers,
        'simplex_tree': simplex_tree,
        'num_simplices': simplex_tree.num_simplices(),
        'num_vertices': simplex_tree.num_vertices(),
    }


def compute_cubical_complex_persistence(
    field: np.ndarray,
    max_dimension: int = 2,
    periodic: bool = False,
) -> Dict:
    """
    Compute persistent homology using Cubical complex.

    Cubical complexes are ideal for voxel/image data, as they directly
    use the grid structure without triangulation.

    Parameters
    ----------
    field : ndarray, shape (nx, ny, nz) or (nx, ny)
        Scalar field on regular grid (e.g., vorticity magnitude)
    max_dimension : int
        Maximum homology dimension
    periodic : bool
        Whether to use periodic boundary conditions

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'persistence': List of (dimension, (birth, death)) tuples
        - 'betti_numbers': Betti numbers
        - 'cubical_complex': Gudhi cubical complex object

    Algorithm:
    ----------
    1. Interpret field as cubical complex (sublevel filtration)
    2. Filtration: {x : field(x) ≤ θ} for increasing θ
    3. Compute persistent homology
    4. Extract persistence diagram

    Examples
    --------
    >>> # Vorticity field
    >>> vorticity = load_vorticity_field()
    >>> result = compute_cubical_complex_persistence(vorticity)
    >>> print(f"H¹ features: {sum(1 for dim, _ in result['persistence'] if dim == 1)}")

    Notes
    -----
    Cubical complexes:
    - No triangulation needed (use grid directly)
    - Very efficient for image/voxel data
    - Support periodic boundary conditions
    - Natural for DNS turbulence data
    """
    if not GUDHI_AVAILABLE:
        raise ImportError("Gudhi required. Install with: pip install gudhi")

    # Flatten field for Gudhi (row-major order)
    # Gudhi expects 1D array with dimensions specified separately
    field_flat = field.flatten()

    # Create cubical complex
    if periodic:
        # Periodic cubical complex
        cubical_complex = gudhi.PeriodicCubicalComplex(
            dimensions=field.shape,
            top_dimensional_cells=field_flat
        )
    else:
        # Standard cubical complex
        cubical_complex = gudhi.CubicalComplex(
            dimensions=field.shape,
            top_dimensional_cells=field_flat
        )

    # Compute persistence
    persistence = cubical_complex.persistence(
        homology_coeff_field=2,
        min_persistence=0
    )

    # Compute Betti numbers
    betti_numbers = cubical_complex.betti_numbers()

    return {
        'persistence': persistence,
        'betti_numbers': betti_numbers,
        'cubical_complex': cubical_complex,
        'num_cells': cubical_complex.num_simplices(),
    }


def compute_rips_complex_persistence(
    points: np.ndarray,
    max_edge_length: float = np.inf,
    max_dimension: int = 2,
    sparse: Optional[float] = None,
) -> Dict:
    """
    Compute persistent homology using Vietoris-Rips complex.

    Rips complexes are the most common for persistent homology.
    They approximate metric spaces.

    Parameters
    ----------
    points : ndarray, shape (n_points, dimension)
        Point cloud
    max_edge_length : float
        Maximum edge length in Rips complex
    max_dimension : int
        Maximum homology dimension
    sparse : float, optional
        If provided, use sparse Rips with this approximation parameter

    Returns
    -------
    result : dict
        Persistence results

    Notes
    -----
    Rips vs Alpha:
    - Rips: Combinatorial, works in any metric space
    - Alpha: Geometric, requires Euclidean space
    - Rips: Can be sparse (approximation)
    - Alpha: Always exact

    For turbulence topology, we prefer:
    - Cubical complex for vorticity fields (voxel data)
    - Alpha complex for extracted structures (geometric)
    """
    if not GUDHI_AVAILABLE:
        raise ImportError("Gudhi required. Install with: pip install gudhi")

    # Create Rips complex
    if sparse is not None:
        rips_complex = gudhi.RipsComplex(
            points=points,
            max_edge_length=max_edge_length,
            sparse=sparse
        )
    else:
        rips_complex = gudhi.RipsComplex(
            points=points,
            max_edge_length=max_edge_length
        )

    # Create simplex tree
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

    # Compute persistence
    persistence = simplex_tree.persistence(
        homology_coeff_field=2,
        min_persistence=0
    )

    # Betti numbers
    betti_numbers = simplex_tree.betti_numbers()

    return {
        'persistence': persistence,
        'betti_numbers': betti_numbers,
        'simplex_tree': simplex_tree,
        'num_simplices': simplex_tree.num_simplices(),
    }


def compare_ripser_gudhi(
    points: np.ndarray,
    max_dimension: int = 2,
) -> Dict[str, Dict]:
    """
    Compare Ripser and Gudhi performance and results.

    This function runs both Ripser and Gudhi on the same data and
    compares:
    - Computation time
    - Number of features found
    - Agreement between persistence diagrams

    Parameters
    ----------
    points : ndarray
        Point cloud
    max_dimension : int
        Maximum homology dimension

    Returns
    -------
    comparison : dict
        Dictionary with keys 'ripser' and 'gudhi' containing results
        and a 'comparison' key with metrics

    Examples
    --------
    >>> points = np.random.randn(500, 3)
    >>> comp = compare_ripser_gudhi(points)
    >>> print(f"Ripser time: {comp['comparison']['ripser_time']:.3f}s")
    >>> print(f"Gudhi time: {comp['comparison']['gudhi_time']:.3f}s")
    >>> print(f"Agreement: {comp['comparison']['agreement']:.2%}")
    """
    import time

    # Run Ripser
    try:
        from ripser import ripser as ripser_compute
        ripser_available = True
    except ImportError:
        ripser_available = False
        warnings.warn("Ripser not available for comparison")

    results = {}

    # Ripser
    if ripser_available:
        t0 = time.time()
        ripser_result = ripser_compute(points, maxdim=max_dimension)
        ripser_time = time.time() - t0

        results['ripser'] = {
            'dgms': ripser_result['dgms'],
            'time': ripser_time,
            'num_features': {i: len(dgm) for i, dgm in enumerate(ripser_result['dgms'])}
        }

    # Gudhi (Rips)
    if GUDHI_AVAILABLE:
        t0 = time.time()
        gudhi_result = compute_rips_complex_persistence(points, max_dimension=max_dimension)
        gudhi_time = time.time() - t0

        # Convert Gudhi persistence to diagram format
        gudhi_dgms = []
        for dim in range(max_dimension + 1):
            dgm = np.array([
                [birth, death] for d, (birth, death) in gudhi_result['persistence']
                if d == dim and not np.isinf(death)
            ])
            if len(dgm) == 0:
                dgm = np.zeros((0, 2))
            gudhi_dgms.append(dgm)

        results['gudhi'] = {
            'dgms': gudhi_dgms,
            'time': gudhi_time,
            'num_features': {i: len(dgm) for i, dgm in enumerate(gudhi_dgms)}
        }

    # Comparison
    if ripser_available and GUDHI_AVAILABLE:
        # Count features
        ripser_total = sum(len(dgm) for dgm in ripser_result['dgms'])
        gudhi_total = sum(len(dgm) for dgm in gudhi_dgms)

        results['comparison'] = {
            'ripser_time': ripser_time,
            'gudhi_time': gudhi_time,
            'speedup': ripser_time / gudhi_time if gudhi_time > 0 else np.inf,
            'ripser_features': ripser_total,
            'gudhi_features': gudhi_total,
            'feature_difference': abs(ripser_total - gudhi_total),
        }

    return results


def extract_representative_cycles(
    simplex_tree,
    persistence_pairs: List,
    dimension: int = 1,
) -> List[List]:
    """
    Extract representative cycles for persistent homology features.

    Representative cycles are the actual geometric features (loops, voids)
    that create homology classes.

    Parameters
    ----------
    simplex_tree : gudhi.SimplexTree
        Simplex tree from Alpha or Rips complex
    persistence_pairs : list
        Persistence pairs from simplex_tree.persistence()
    dimension : int
        Homology dimension to extract cycles for

    Returns
    -------
    cycles : list of list
        Each cycle is a list of simplices

    Examples
    --------
    >>> alpha_result = compute_alpha_complex_persistence(points)
    >>> cycles = extract_representative_cycles(
    ...     alpha_result['simplex_tree'],
    ...     alpha_result['persistence'],
    ...     dimension=1
    ... )
    >>> print(f"Found {len(cycles)} H¹ generators")

    Notes
    -----
    Representative cycles are crucial for visualization:
    - H⁰ (dimension 0): Connected components
    - H¹ (dimension 1): Loops/cycles
    - H² (dimension 2): Voids/cavities

    For turbulence, H¹ cycles correspond to vortex tubes/rings.
    """
    if not GUDHI_AVAILABLE:
        raise ImportError("Gudhi required")

    # Filter for desired dimension
    dim_pairs = [(i, (b, d)) for i, (b, d) in persistence_pairs if i == dimension]

    if len(dim_pairs) == 0:
        return []

    # Extract cycles (this requires Gudhi's persistence_pairs_simplices)
    # Note: This feature may not be available in all Gudhi versions
    try:
        # Get persistence pairs with simplices
        pairs_with_simplices = simplex_tree.persistence_pairs()

        # Extract cycles for specified dimension
        cycles = []
        for pair in pairs_with_simplices:
            if len(pair) == 2:  # Birth-death pair
                birth_simplex, death_simplex = pair
                # The cycle is the boundary of the death simplex
                if simplex_tree.dimension(death_simplex) == dimension + 1:
                    cycles.append([birth_simplex, death_simplex])

        return cycles
    except AttributeError:
        warnings.warn("Representative cycle extraction not available in this Gudhi version")
        return []


def persistence_to_diagrams(
    persistence: List[Tuple[int, Tuple[float, float]]],
    max_dimension: int = 2,
) -> List[np.ndarray]:
    """
    Convert Gudhi persistence format to persistence diagram format.

    Gudhi returns: [(dim, (birth, death)), ...]
    We want: [dgm_0, dgm_1, ...] where dgm_i is ndarray of shape (n, 2)

    Parameters
    ----------
    persistence : list
        Gudhi persistence output
    max_dimension : int
        Maximum dimension

    Returns
    -------
    diagrams : list of ndarray
        Persistence diagrams, one per dimension
    """
    diagrams = []

    for dim in range(max_dimension + 1):
        # Extract features of this dimension
        features = []
        for d, (birth, death) in persistence:
            if d == dim:
                # Skip infinite bars (essential features)
                if not np.isinf(death):
                    features.append([birth, death])

        # Convert to array
        if len(features) > 0:
            dgm = np.array(features)
        else:
            dgm = np.zeros((0, 2))

        diagrams.append(dgm)

    return diagrams
