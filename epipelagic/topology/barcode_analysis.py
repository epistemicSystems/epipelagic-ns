"""
Persistence barcode and diagram analysis.

This module implements Task 2.3 barcode analysis from TASKS.md:
1. Persistence landscapes
2. Bottleneck and Wasserstein distances
3. Statistical significance testing
4. Persistent entropy

These tools allow us to:
- Compare topological signatures across different Reynolds numbers
- Quantify topological changes
- Test significance of observed features

References:
    [1] Bubenik (2015). "Statistical Topological Data Analysis using Persistence Landscapes"
    [2] Kerber et al. (2017). "Geometry Helps to Compare Persistence Diagrams"
"""

from typing import Tuple, List, Optional, Callable
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import warnings


def compute_persistence_landscape(
    persistence_diagram: np.ndarray,
    k: int = 5,
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute persistence landscape from persistence diagram.

    Persistence landscapes are functional summaries of persistence diagrams
    that can be averaged, compared using L^p norms, and used for statistics.

    Parameters
    ----------
    persistence_diagram : ndarray, shape (n, 2)
        Persistence diagram with (birth, death) pairs
    k : int
        Number of landscape functions to compute
    resolution : int
        Number of points for discretization

    Returns
    -------
    t_values : ndarray, shape (resolution,)
        Grid points for landscape functions
    landscapes : ndarray, shape (k, resolution)
        k landscape functions λ₁, λ₂, ..., λₖ

    Algorithm:
    ----------
    For each point (b, d) in the diagram, define tent function:
        Λ(b,d)(t) = max(0, min(t - b, d - t))

    The k-th landscape function is:
        λₖ(t) = k-th largest value of {Λ(b,d)(t)} over all (b, d)

    Examples
    --------
    >>> dgm = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.6]])
    >>> t, landscapes = compute_persistence_landscape(dgm, k=2)
    >>> import matplotlib.pyplot as plt
    >>> for i in range(2):
    ...     plt.plot(t, landscapes[i], label=f'λ_{i+1}')
    >>> plt.legend()

    Properties:
    -----------
    - Stable: small changes in diagram → small changes in landscape
    - Linear: can average landscapes from multiple diagrams
    - Complete: landscapes determine diagram (for finite diagrams)
    """
    if len(persistence_diagram) == 0:
        t_values = np.linspace(0, 1, resolution)
        return t_values, np.zeros((k, resolution))

    # Determine grid
    births = persistence_diagram[:, 0]
    deaths = persistence_diagram[:, 1]
    t_min = np.min(births)
    t_max = np.max(deaths)

    # Add padding
    padding = 0.1 * (t_max - t_min)
    t_values = np.linspace(t_min - padding, t_max + padding, resolution)

    # Compute tent functions for each bar
    n_bars = len(persistence_diagram)
    tent_values = np.zeros((n_bars, resolution))

    for i, (b, d) in enumerate(persistence_diagram):
        # Tent function: Λ(t) = max(0, min(t - b, d - t))
        tent_values[i] = np.maximum(0, np.minimum(t_values - b, d - t_values))

    # Compute k landscape functions
    landscapes = np.zeros((k, resolution))

    for j in range(resolution):
        # At each t, sort tent values and take k largest
        sorted_values = np.sort(tent_values[:, j])[::-1]  # Descending order

        # Fill in landscape functions
        for i in range(min(k, len(sorted_values))):
            landscapes[i, j] = sorted_values[i]

    return t_values, landscapes


def landscape_distance(
    landscape1: np.ndarray,
    landscape2: np.ndarray,
    p: float = 2.0,
) -> float:
    """
    Compute L^p distance between persistence landscapes.

    Parameters
    ----------
    landscape1, landscape2 : ndarray, shape (k, resolution)
        Persistence landscapes
    p : float
        Order of L^p norm (default: 2 for L² norm)

    Returns
    -------
    distance : float
        L^p distance between landscapes

    Formula:
    --------
    d_p(λ, μ) = (∫ |λ(t) - μ(t)|^p dt)^(1/p)

    Examples
    --------
    >>> t1, L1 = compute_persistence_landscape(dgm1)
    >>> t2, L2 = compute_persistence_landscape(dgm2)
    >>> dist = landscape_distance(L1, L2, p=2)
    >>> print(f"L² distance: {dist:.3f}")
    """
    if landscape1.shape != landscape2.shape:
        raise ValueError("Landscapes must have same shape")

    # Compute pointwise differences
    diff = np.abs(landscape1 - landscape2)

    # Sum over all landscape functions
    total_diff = np.sum(diff ** p)

    # Normalize by number of points (trapezoid rule)
    dt = 1.0 / landscape1.shape[1]

    # L^p norm
    distance = (total_diff * dt) ** (1.0 / p)

    return distance


def bottleneck_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    internal_p: float = np.inf,
) -> float:
    """
    Compute bottleneck distance between persistence diagrams.

    The bottleneck distance is the infimum over all bijections of the
    maximum distance between matched points.

    Parameters
    ----------
    dgm1, dgm2 : ndarray, shape (n, 2)
        Persistence diagrams
    internal_p : float
        Internal distance metric (default: ∞ for L^∞)

    Returns
    -------
    distance : float
        Bottleneck distance

    Algorithm:
    ----------
    d_B(X, Y) = inf_γ sup_{x∈X} d(x, γ(x))

    where γ ranges over all bijections X → Y (including diagonal points).

    We solve this using the Hungarian algorithm on an extended cost matrix.

    Examples
    --------
    >>> dgm1 = np.array([[0.1, 0.5], [0.2, 0.8]])
    >>> dgm2 = np.array([[0.15, 0.52], [0.25, 0.75]])
    >>> dist = bottleneck_distance(dgm1, dgm2)
    >>> print(f"Bottleneck distance: {dist:.4f}")

    Properties:
    -----------
    - Stability: d_B(PH(f), PH(g)) ≤ ||f - g||_∞
    - Metric: satisfies triangle inequality
    - Robust: less sensitive to outliers than Wasserstein
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    # Add diagonal points (projections to diagonal)
    # For each point (b, d), add its projection ((b+d)/2, (b+d)/2)
    n1 = len(dgm1)
    n2 = len(dgm2)

    # Augment diagrams with diagonal projections
    dgm1_aug = np.vstack([dgm1, dgm2]) if len(dgm1) > 0 else dgm2
    dgm2_aug = np.vstack([dgm2, dgm1]) if len(dgm2) > 0 else dgm1

    # Project to diagonal
    for i in range(len(dgm1_aug)):
        if i >= n1:  # Diagonal points from dgm2
            b, d = dgm1_aug[i]
            mid = (b + d) / 2
            dgm1_aug[i] = [mid, mid]

    for i in range(len(dgm2_aug)):
        if i >= n2:  # Diagonal points from dgm1
            b, d = dgm2_aug[i]
            mid = (b + d) / 2
            dgm2_aug[i] = [mid, mid]

    # Compute cost matrix
    if internal_p == np.inf:
        # L^∞ metric
        cost_matrix = np.max(np.abs(dgm1_aug[:, None] - dgm2_aug[None, :]), axis=2)
    else:
        # L^p metric
        cost_matrix = np.linalg.norm(dgm1_aug[:, None] - dgm2_aug[None, :], ord=internal_p, axis=2)

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Bottleneck distance is the maximum cost in optimal matching
    distance = np.max(cost_matrix[row_ind, col_ind])

    return distance


def wasserstein_distance(
    dgm1: np.ndarray,
    dgm2: np.ndarray,
    p: float = 2.0,
    internal_p: float = 2.0,
) -> float:
    """
    Compute Wasserstein distance between persistence diagrams.

    The Wasserstein distance sums all matched distances, making it more
    sensitive to global structure than bottleneck distance.

    Parameters
    ----------
    dgm1, dgm2 : ndarray
        Persistence diagrams
    p : float
        Order of Wasserstein distance (default: 2)
    internal_p : float
        Internal metric order (default: 2 for L²)

    Returns
    -------
    distance : float
        p-Wasserstein distance

    Formula:
    --------
    W_p(X, Y) = (inf_γ Σ_{x∈X} d(x, γ(x))^p)^(1/p)

    Examples
    --------
    >>> dist_w = wasserstein_distance(dgm1, dgm2, p=2)
    >>> print(f"2-Wasserstein distance: {dist_w:.4f}")

    Properties:
    -----------
    - More sensitive than bottleneck to all features
    - Stable under perturbations
    - W_∞ = bottleneck distance
    """
    if len(dgm1) == 0 and len(dgm2) == 0:
        return 0.0

    # Similar augmentation as bottleneck distance
    n1 = len(dgm1)
    n2 = len(dgm2)

    # Ensure same number of points by adding diagonal points
    max_n = max(n1, n2)

    dgm1_aug = np.copy(dgm1)
    dgm2_aug = np.copy(dgm2)

    # Add diagonal points to smaller diagram
    if n1 < max_n:
        # Add projections of dgm2 points to diagonal
        for i in range(n1, max_n):
            if i < n2:
                b, d = dgm2[i]
                mid = (b + d) / 2
                dgm1_aug = np.vstack([dgm1_aug, [mid, mid]])

    if n2 < max_n:
        # Add projections of dgm1 points to diagonal
        for i in range(n2, max_n):
            if i < n1:
                b, d = dgm1[i]
                mid = (b + d) / 2
                dgm2_aug = np.vstack([dgm2_aug, [mid, mid]])

    # Compute cost matrix
    cost_matrix = np.linalg.norm(
        dgm1_aug[:, None] - dgm2_aug[None, :],
        ord=internal_p,
        axis=2
    ) ** p

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Wasserstein distance
    total_cost = np.sum(cost_matrix[row_ind, col_ind])
    distance = total_cost ** (1.0 / p)

    return distance


def persistent_entropy(
    persistence_diagram: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute persistent entropy of a persistence diagram.

    Persistent entropy quantifies the complexity/information content
    of a persistence diagram.

    Parameters
    ----------
    persistence_diagram : ndarray, shape (n, 2)
        Persistence diagram
    normalize : bool
        If True, normalize by total persistence

    Returns
    -------
    entropy : float
        Persistent entropy

    Formula:
    --------
    H = -Σᵢ pᵢ log(pᵢ)

    where pᵢ = (dᵢ - bᵢ) / Σⱼ(dⱼ - bⱼ) is the normalized persistence.

    Interpretation:
    ---------------
    - High entropy: many features with similar persistence
    - Low entropy: dominated by few long-lived features
    - For epipelagic regime: expect moderate entropy
      (not too many features, not too few)

    Examples
    --------
    >>> dgm = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.6]])
    >>> H = persistent_entropy(dgm)
    >>> print(f"Entropy: {H:.3f}")
    """
    if len(persistence_diagram) == 0:
        return 0.0

    # Compute persistence values
    lifetimes = persistence_diagram[:, 1] - persistence_diagram[:, 0]

    # Filter out zero or negative lifetimes
    lifetimes = lifetimes[lifetimes > 0]

    if len(lifetimes) == 0:
        return 0.0

    # Normalize to probabilities
    if normalize:
        total = np.sum(lifetimes)
        if total == 0:
            return 0.0
        probabilities = lifetimes / total
    else:
        probabilities = lifetimes

    # Compute entropy
    # Handle numerical issues: p log p → 0 as p → 0
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log(p)

    return entropy


def betti_curve(
    persistence_diagram: np.ndarray,
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Betti curve (Betti number as function of threshold).

    Parameters
    ----------
    persistence_diagram : ndarray
        Persistence diagram
    resolution : int
        Number of threshold values

    Returns
    -------
    thresholds : ndarray, shape (resolution,)
        Threshold values
    betti_numbers : ndarray, shape (resolution,)
        Betti number at each threshold

    Algorithm:
    ----------
    At threshold θ:
    - Count features born before θ
    - Subtract features that died before θ
    - Result is β(θ) = # features alive at θ

    Examples
    --------
    >>> dgm = persistence_diagram_H1
    >>> t, beta = betti_curve(dgm)
    >>> plt.plot(t, beta)
    >>> plt.xlabel('Threshold θ')
    >>> plt.ylabel('β¹(θ)')
    """
    if len(persistence_diagram) == 0:
        return np.array([0, 1]), np.array([0, 0])

    # Determine range
    births = persistence_diagram[:, 0]
    deaths = persistence_diagram[:, 1]

    # Filter out infinite deaths
    finite_deaths = deaths[~np.isinf(deaths)]

    t_min = np.min(births)
    t_max = np.max(finite_deaths) if len(finite_deaths) > 0 else np.max(births) + 1

    # Create threshold grid
    thresholds = np.linspace(t_min, t_max, resolution)

    # Compute Betti numbers
    betti_numbers = np.zeros(resolution)

    for i, theta in enumerate(thresholds):
        # Count features alive at theta
        alive = (births <= theta) & (theta < deaths)
        betti_numbers[i] = np.sum(alive)

    return thresholds, betti_numbers


def statistical_significance_test(
    observed_diagram: np.ndarray,
    null_diagrams: List[np.ndarray],
    metric: Callable = bottleneck_distance,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Test statistical significance of observed persistence diagram.

    Null hypothesis: observed diagram comes from null distribution
    Alternative: observed diagram is significantly different

    Parameters
    ----------
    observed_diagram : ndarray
        Observed persistence diagram
    null_diagrams : list of ndarray
        Null distribution (e.g., from random noise)
    metric : callable
        Distance function (default: bottleneck_distance)
    alpha : float
        Significance level (default: 0.05)

    Returns
    -------
    significant : bool
        True if null hypothesis rejected
    p_value : float
        Empirical p-value

    Algorithm:
    ----------
    1. Compute distances from observed to all null diagrams
    2. Compute distances between pairs of null diagrams
    3. p-value = fraction of null-null distances ≥ obs-null distance
    4. Reject if p-value < alpha

    Examples
    --------
    >>> # Generate null distribution from noise
    >>> null_dgms = [compute_persistence(noise) for _ in range(100)]
    >>> significant, p = statistical_significance_test(observed_dgm, null_dgms)
    >>> print(f"Significant: {significant}, p = {p:.4f}")
    """
    if len(null_diagrams) == 0:
        raise ValueError("Need at least one null diagram")

    # Compute distances from observed to null
    obs_null_distances = []
    for null_dgm in null_diagrams:
        dist = metric(observed_diagram, null_dgm)
        obs_null_distances.append(dist)

    # Compute distances between null diagrams (for calibration)
    null_null_distances = []
    n_null = len(null_diagrams)
    for i in range(n_null):
        for j in range(i + 1, n_null):
            dist = metric(null_diagrams[i], null_diagrams[j])
            null_null_distances.append(dist)

    # Empirical p-value
    # What fraction of null-null distances are ≥ median obs-null distance?
    if len(null_null_distances) == 0:
        # Single null diagram: use 0 as reference
        threshold = np.median(obs_null_distances)
        p_value = 0.0
    else:
        threshold = np.median(obs_null_distances)
        p_value = np.mean(np.array(null_null_distances) >= threshold)

    # Test significance
    significant = p_value < alpha

    return significant, p_value