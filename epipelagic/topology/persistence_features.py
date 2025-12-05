"""
Persistence feature extraction for machine learning.

This module implements Task 2.3 feature extraction from TASKS.md:
1. Birth/death coordinates
2. Representative cycles (homology generators)
3. Persistence images (ML-ready features)
4. Betti numbers vs threshold

These features enable:
- Machine learning on topological signatures
- Classification of turbulent regimes
- Regression for Reynolds number prediction

References:
    [1] Adams et al. (2017). "Persistence Images: A Stable Vector Representation of Persistent Homology"
    [2] Khasawneh & Munch (2016). "Chatter Detection in Turning Using Persistence Diagrams"
"""

from typing import Tuple, Optional, Callable
import numpy as np
from scipy import stats


def extract_birth_death_coordinates(
    persistence_diagram: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract birth and death coordinates from persistence diagram.

    Parameters
    ----------
    persistence_diagram : ndarray, shape (n, 2)
        Persistence diagram

    Returns
    -------
    births : ndarray, shape (n,)
        Birth times
    deaths : ndarray, shape (n,)
        Death times

    Examples
    --------
    >>> dgm = np.array([[0.1, 0.5], [0.2, 0.8]])
    >>> births, deaths = extract_birth_death_coordinates(dgm)
    >>> print(f"Births: {births}, Deaths: {deaths}")
    """
    if len(persistence_diagram) == 0:
        return np.array([]), np.array([])

    births = persistence_diagram[:, 0]
    deaths = persistence_diagram[:, 1]

    return births, deaths


def compute_persistence_statistics(
    persistence_diagram: np.ndarray,
) -> dict:
    """
    Compute statistical summary of persistence diagram.

    Parameters
    ----------
    persistence_diagram : ndarray
        Persistence diagram

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'num_features': Number of features
        - 'mean_persistence': Mean persistence (death - birth)
        - 'std_persistence': Std of persistence
        - 'max_persistence': Maximum persistence
        - 'total_persistence': Sum of all persistence values
        - 'persistence_quantiles': Quantiles [0.25, 0.5, 0.75]

    Examples
    --------
    >>> dgm = persistence_diagram_H1
    >>> stats = compute_persistence_statistics(dgm)
    >>> print(f"Mean persistence: {stats['mean_persistence']:.3f}")

    Applications:
    -------------
    These statistics provide simple scalar features for ML:
    - Total persistence correlates with turbulence intensity
    - Max persistence indicates strongest coherent structures
    - Number of features relates to cascade complexity
    """
    if len(persistence_diagram) == 0:
        return {
            'num_features': 0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'max_persistence': 0.0,
            'total_persistence': 0.0,
            'persistence_quantiles': [0.0, 0.0, 0.0],
        }

    # Compute persistence values
    lifetimes = persistence_diagram[:, 1] - persistence_diagram[:, 0]

    # Filter finite values
    finite_lifetimes = lifetimes[np.isfinite(lifetimes)]

    if len(finite_lifetimes) == 0:
        finite_lifetimes = np.array([0.0])

    # Compute statistics
    result = {
        'num_features': len(persistence_diagram),
        'mean_persistence': float(np.mean(finite_lifetimes)),
        'std_persistence': float(np.std(finite_lifetimes)),
        'max_persistence': float(np.max(finite_lifetimes)),
        'total_persistence': float(np.sum(finite_lifetimes)),
        'persistence_quantiles': np.quantile(finite_lifetimes, [0.25, 0.5, 0.75]).tolist(),
    }

    return result


def compute_persistence_image(
    persistence_diagram: np.ndarray,
    resolution: Tuple[int, int] = (20, 20),
    sigma: Optional[float] = None,
    weight_function: Optional[Callable] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Compute persistence image from persistence diagram.

    Persistence images convert diagrams to fixed-size raster images,
    enabling standard machine learning techniques.

    Parameters
    ----------
    persistence_diagram : ndarray, shape (n, 2)
        Persistence diagram (birth, death)
    resolution : tuple of int
        Image resolution (height, width)
    sigma : float, optional
        Gaussian bandwidth (default: adaptive)
    weight_function : callable, optional
        Function (birth, death) → weight (default: persistence)

    Returns
    -------
    image : ndarray, shape (resolution[0], resolution[1])
        Persistence image
    bounds : tuple of float
        (birth_min, birth_max, death_min, death_max) bounds

    Algorithm:
    ----------
    1. Transform to (birth, persistence) coordinates
    2. Apply weight function (default: weight = persistence)
    3. Place weighted Gaussian at each point
    4. Sum to get continuous surface
    5. Discretize on grid

    Examples
    --------
    >>> dgm = persistence_diagram_H1
    >>> img, bounds = compute_persistence_image(dgm, resolution=(50, 50))
    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(img, origin='lower', extent=bounds)
    >>> plt.colorbar()

    Applications:
    -------------
    - Input to CNNs for regime classification
    - Clustering turbulent states
    - Transfer learning from image models

    References:
    -----------
    Adams et al. (2017). "Persistence Images"
    """
    if len(persistence_diagram) == 0:
        return np.zeros(resolution), (0, 1, 0, 1)

    # Transform to (birth, persistence) coordinates
    births = persistence_diagram[:, 0]
    deaths = persistence_diagram[:, 1]
    persistence = deaths - births

    # Filter finite values
    finite_mask = np.isfinite(persistence)
    births = births[finite_mask]
    persistence = persistence[finite_mask]

    if len(births) == 0:
        return np.zeros(resolution), (0, 1, 0, 1)

    # Determine bounds
    birth_min, birth_max = np.min(births), np.max(births)
    pers_min, pers_max = 0, np.max(persistence)

    # Add padding
    birth_padding = 0.1 * (birth_max - birth_min) if birth_max > birth_min else 0.1
    pers_padding = 0.1 * pers_max if pers_max > 0 else 0.1

    birth_min -= birth_padding
    birth_max += birth_padding
    pers_max += pers_padding

    # Create grid
    birth_grid = np.linspace(birth_min, birth_max, resolution[1])
    pers_grid = np.linspace(pers_min, pers_max, resolution[0])
    B, P = np.meshgrid(birth_grid, pers_grid)

    # Determine sigma (Gaussian bandwidth)
    if sigma is None:
        # Adaptive: based on average nearest neighbor distance
        if len(births) > 1:
            # Estimate from data
            sigma = np.mean(persistence) / 5.0
        else:
            sigma = (pers_max - pers_min) / 20.0

    # Weight function (default: persistence)
    if weight_function is None:
        weights = persistence
    else:
        weights = np.array([weight_function(b, d) for b, d in zip(births, deaths)])

    # Compute persistence image
    image = np.zeros(resolution)

    for i in range(len(births)):
        # Gaussian centered at (births[i], persistence[i]) with weight weights[i]
        gaussian = weights[i] * np.exp(
            -((B - births[i])**2 + (P - persistence[i])**2) / (2 * sigma**2)
        )
        image += gaussian

    # Normalize
    if np.max(image) > 0:
        image /= np.max(image)

    bounds = (birth_min, birth_max, pers_min, pers_max)

    return image, bounds


def compute_betti_numbers_vs_threshold(
    persistence_diagrams: list,
    thresholds: Optional[np.ndarray] = None,
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Betti numbers as function of filtration threshold.

    Parameters
    ----------
    persistence_diagrams : list of ndarray
        List of persistence diagrams [H⁰, H¹, H², ...]
    thresholds : ndarray, optional
        Threshold values (default: auto from diagrams)
    resolution : int
        Number of threshold values if auto

    Returns
    -------
    thresholds : ndarray, shape (resolution,)
        Threshold values
    betti_curves : ndarray, shape (n_dimensions, resolution)
        Betti numbers for each dimension at each threshold

    Examples
    --------
    >>> dgms = [dgm_H0, dgm_H1, dgm_H2]
    >>> t, betti = compute_betti_numbers_vs_threshold(dgms)
    >>> import matplotlib.pyplot as plt
    >>> for i in range(len(dgms)):
    ...     plt.plot(t, betti[i], label=f'β{i}')
    >>> plt.legend()

    Applications:
    -------------
    For epipelagic turbulence:
    - β⁰(θ): Connected components (energy-containing eddies)
    - β¹(θ): Loops (vortex rings, circulation)
    - β²(θ): Voids (bubble-like structures)

    Expect: β¹ grows then plateaus in epipelagic regime
    """
    n_dims = len(persistence_diagrams)

    # Determine threshold range
    if thresholds is None:
        all_births = []
        all_deaths = []

        for dgm in persistence_diagrams:
            if len(dgm) > 0:
                all_births.extend(dgm[:, 0].tolist())
                deaths = dgm[:, 1]
                finite_deaths = deaths[np.isfinite(deaths)]
                all_deaths.extend(finite_deaths.tolist())

        if len(all_births) == 0:
            thresholds = np.linspace(0, 1, resolution)
        else:
            t_min = min(all_births)
            t_max = max(all_deaths) if len(all_deaths) > 0 else max(all_births) + 1
            thresholds = np.linspace(t_min, t_max, resolution)

    # Compute Betti curves
    betti_curves = np.zeros((n_dims, len(thresholds)))

    for dim_idx, dgm in enumerate(persistence_diagrams):
        if len(dgm) == 0:
            continue

        births = dgm[:, 0]
        deaths = dgm[:, 1]

        for t_idx, theta in enumerate(thresholds):
            # Count features alive at theta
            alive = (births <= theta) & (theta < deaths)
            betti_curves[dim_idx, t_idx] = np.sum(alive)

    return thresholds, betti_curves


def extract_epipelagic_features(
    persistence_diagrams: list,
    threshold: float = 0.5,
) -> dict:
    """
    Extract features specific to epipelagic cohomology analysis.

    This function extracts features relevant for validating Theorem C:
    dim(H¹_epi) ≤ C log(Re)

    Parameters
    ----------
    persistence_diagrams : list of ndarray
        Persistence diagrams [H⁰, H¹, H²]
    threshold : float
        Persistence threshold for "long" features

    Returns
    -------
    features : dict
        Dictionary containing:
        - 'dim_H0_epi': dim(H⁰_epi) (connected components)
        - 'dim_H1_epi': dim(H¹_epi) (THE KEY QUANTITY)
        - 'dim_H2_epi': dim(H²_epi) (voids)
        - 'H1_persistence_stats': Statistics of H¹ features
        - 'cascade_indicator': Indicator of cascade activity

    Examples
    --------
    >>> dgms = extract_persistence_diagrams(vorticity)
    >>> features = extract_epipelagic_features(dgms, threshold=0.5)
    >>> print(f"dim(H¹_epi) = {features['dim_H1_epi']}")

    Theoretical Connection:
    -----------------------
    dim(H¹_epi) = # long-lived H¹ features
                = dimension of epipelagic cohomology
                ≤ C log(Re)  (Theorem C)

    This is the computational realization of the mathematical theory.
    """
    features = {}

    # Extract dim(H^d_epi) for each dimension
    for dim_idx, dgm in enumerate(persistence_diagrams):
        if len(dgm) == 0:
            features[f'dim_H{dim_idx}_epi'] = 0
            continue

        # Compute persistence values
        persistence = dgm[:, 1] - dgm[:, 0]

        # Filter long-lived features
        long_lived = persistence > threshold

        # Count long-lived features
        dim_epi = np.sum(long_lived)
        features[f'dim_H{dim_idx}_epi'] = int(dim_epi)

    # Special focus on H¹ (the key quantity for epipelagic theory)
    if len(persistence_diagrams) > 1:
        dgm_H1 = persistence_diagrams[1]

        if len(dgm_H1) > 0:
            # Statistics of H¹ features
            stats = compute_persistence_statistics(dgm_H1)
            features['H1_persistence_stats'] = stats

            # Cascade activity indicator
            # High total H¹ persistence → active energy cascade
            features['cascade_indicator'] = stats['total_persistence']
        else:
            features['H1_persistence_stats'] = {'num_features': 0}
            features['cascade_indicator'] = 0.0

    return features


def create_feature_vector(
    persistence_diagrams: list,
    include_images: bool = False,
    image_resolution: Tuple[int, int] = (20, 20),
) -> np.ndarray:
    """
    Create fixed-length feature vector from persistence diagrams.

    This converts variable-length diagrams to fixed-length vectors
    suitable for ML algorithms.

    Parameters
    ----------
    persistence_diagrams : list of ndarray
        Persistence diagrams [H⁰, H¹, H²]
    include_images : bool
        If True, include flattened persistence images
    image_resolution : tuple
        Resolution for persistence images

    Returns
    -------
    feature_vector : ndarray
        Fixed-length feature vector

    Feature Components:
    -------------------
    1. Epipelagic dimensions (dim H⁰, H¹, H²)
    2. Persistence statistics (mean, std, max, total)
    3. Persistence entropy
    4. Betti number integrals
    5. [Optional] Persistence images (flattened)

    Examples
    --------
    >>> dgms = extract_persistence_diagrams(vorticity)
    >>> features = create_feature_vector(dgms, include_images=True)
    >>> print(f"Feature vector length: {len(features)}")

    Use Case:
    ---------
    >>> # Collect features from multiple Reynolds numbers
    >>> X = []  # Feature matrix
    >>> y = []  # Reynolds numbers
    >>> for Re in reynolds_numbers:
    ...     velocity = simulate_cascade(Re)
    ...     dgms = extract_topology(velocity)
    ...     features = create_feature_vector(dgms)
    ...     X.append(features)
    ...     y.append(Re)
    >>> # Train regression model
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> model = RandomForestRegressor()
    >>> model.fit(X, y)
    """
    from .barcode_analysis import persistent_entropy

    features_list = []

    # 1. Epipelagic dimensions
    epi_features = extract_epipelagic_features(persistence_diagrams)
    features_list.extend([
        epi_features.get('dim_H0_epi', 0),
        epi_features.get('dim_H1_epi', 0),
        epi_features.get('dim_H2_epi', 0),
    ])

    # 2. Statistics for each dimension
    for dgm in persistence_diagrams:
        stats = compute_persistence_statistics(dgm)
        features_list.extend([
            stats['num_features'],
            stats['mean_persistence'],
            stats['std_persistence'],
            stats['max_persistence'],
            stats['total_persistence'],
        ])

    # 3. Persistence entropy for each dimension
    for dgm in persistence_diagrams:
        entropy = persistent_entropy(dgm)
        features_list.append(entropy)

    # 4. Betti number features
    if len(persistence_diagrams) > 0:
        thresholds, betti_curves = compute_betti_numbers_vs_threshold(persistence_diagrams)

        # Integral of Betti curves (area under curve)
        for i in range(len(persistence_diagrams)):
            betti_integral = np.trapz(betti_curves[i], thresholds)
            features_list.append(betti_integral)

    # 5. [Optional] Persistence images
    if include_images:
        for dgm in persistence_diagrams:
            if len(dgm) > 0:
                img, _ = compute_persistence_image(dgm, resolution=image_resolution)
                features_list.extend(img.flatten())
            else:
                # Empty diagram → zero image
                features_list.extend(np.zeros(image_resolution).flatten())

    return np.array(features_list)
