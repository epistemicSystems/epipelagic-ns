"""
Spectral Sequence: {Eᵣᵖ'ᑫ, dᵣ} converging to cascade cohomology.

The spectral sequence arises from filtering the cascade complex by "depth"
(distance from energy injection scale). The epipelagic principle states that
this sequence degenerates at E₂ (i.e., E₂ = E∞).
"""

from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class SpectralSequence:
    """
    Spectral sequence for cascade complex.

    A spectral sequence is a sequence of pages {Eᵣ}ᵣ≥₀ connected by
    differentials dᵣ: Eᵣᵖ'ᑫ → Eᵣᵖ⁺ʳ'ᑫ⁻ʳ⁺¹.

    Attributes
    ----------
    max_page : int
        Maximum page number to compute
    pages : Dict[int, Dict[Tuple[int, int], np.ndarray]]
        Dictionary mapping page number r to bidegree (p,q) to vector space
    differentials : Dict[int, Dict[Tuple[int, int], np.ndarray]]
        Differentials dᵣ for each page
    """

    max_page: int = 10
    pages: Dict[int, Dict[Tuple[int, int], np.ndarray]] = field(default_factory=dict)
    differentials: Dict[int, Dict[Tuple[int, int], np.ndarray]] = field(default_factory=dict)

    @classmethod
    def from_filtered_complex(
        cls,
        complex_matrices: List[np.ndarray],
        filtration: List[int],
        max_page: int = 10,
    ) -> "SpectralSequence":
        """
        Construct spectral sequence from filtered cochain complex.

        Parameters
        ----------
        complex_matrices : List[np.ndarray]
            List of differential matrices [d⁰, d¹, d², ...]
        filtration : List[int]
            Filtration degree for each generator
        max_page : int
            Maximum page to compute

        Returns
        -------
        spectral_sequence : SpectralSequence
            Computed spectral sequence

        Algorithm (Classic Spectral Sequence Construction):
        1. E₀ᵖ'ᑫ = Cᵖ⁺ᑫ_p / Cᵖ⁺ᑫ_{p-1} (associated graded)
        2. d₀ induced from original differential
        3. Eᵣ₊₁ = H(Eᵣ, dᵣ) (homology of previous page)
        """
        seq = cls(max_page=max_page)

        # Build E₀ page (associated graded)
        seq._build_E0_page(complex_matrices, filtration)

        # Compute successive pages via taking homology
        for r in range(max_page):
            seq._compute_next_page(r)

            # Check for degeneration
            if seq._check_degeneration(r):
                print(f"Spectral sequence degenerates at E_{r+1}")
                break

        return seq

    def _build_E0_page(
        self,
        complex_matrices: List[np.ndarray],
        filtration: List[int],
    ) -> None:
        """Build E₀ page as associated graded of filtered complex."""
        # For simplicity, initialize with identity
        # Full implementation requires careful filtration handling
        self.pages[0] = {
            (0, 0): np.eye(complex_matrices[0].shape[1]),
            (1, 0): np.eye(complex_matrices[0].shape[0]),
        }
        self.differentials[0] = {
            (0, 0): complex_matrices[0],
        }

    def _compute_next_page(self, r: int) -> None:
        """
        Compute Eᵣ₊₁ page from Eᵣ via homology.

        Eᵣ₊₁ᵖ'ᑫ = H(Eᵣᵖ'ᑫ, dᵣ) = ker(dᵣ) / im(dᵣ)
        """
        if r not in self.pages:
            return

        next_page = {}
        next_differentials = {}

        for (p, q), space in self.pages[r].items():
            # Get differential from (p,q)
            if (p, q) in self.differentials[r]:
                dr = self.differentials[r][(p, q)]

                # Compute kernel and image
                ker_dr = self._compute_kernel(dr)
                im_dr = self._compute_image(dr)

                # Homology = ker / im
                homology = self._quotient_space(ker_dr, im_dr)

                if homology.shape[1] > 0:
                    next_page[(p, q)] = homology

        self.pages[r + 1] = next_page

    def _compute_kernel(self, matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Compute kernel of matrix via SVD."""
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return np.eye(matrix.shape[1])

        _, S, Vh = np.linalg.svd(matrix, full_matrices=True)
        rank = np.sum(S > tol)
        return Vh[rank:].T

    def _compute_image(self, matrix: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Compute image of matrix via SVD."""
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return np.zeros((matrix.shape[0], 0))

        U, S, _ = np.linalg.svd(matrix, full_matrices=False)
        rank = np.sum(S > tol)
        return U[:, :rank]

    def _quotient_space(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        tol: float = 1e-10,
    ) -> np.ndarray:
        """
        Compute quotient space: numerator / denominator.

        Returns basis for elements in numerator orthogonal to denominator.
        """
        if denominator.shape[1] == 0:
            return numerator

        # Project numerator onto orthogonal complement of denominator
        Q, _ = np.linalg.qr(denominator)
        projection = np.eye(numerator.shape[0]) - Q @ Q.T
        quotient_basis = projection @ numerator

        # Extract non-zero columns
        norms = np.linalg.norm(quotient_basis, axis=0)
        return quotient_basis[:, norms > tol]

    def _check_degeneration(self, r: int, tol: float = 1e-8) -> bool:
        """
        Check if spectral sequence has degenerated at page r.

        Degeneration occurs when all differentials dᵣ vanish.
        """
        if r not in self.differentials:
            return True

        for dr in self.differentials[r].values():
            if np.max(np.abs(dr)) > tol:
                return False

        return True

    def get_E_infinity(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Get E∞ page (limiting page of spectral sequence).

        Returns
        -------
        E_inf : Dict[Tuple[int, int], np.ndarray]
            E∞ᵖ'ᑫ for all bidegrees
        """
        # Return last computed page
        if not self.pages:
            return {}

        max_r = max(self.pages.keys())
        return self.pages[max_r]

    def degenerates_at_E2(self, tol: float = 1e-8) -> bool:
        """
        Check if spectral sequence degenerates at E₂ (epipelagic criterion).

        Returns
        -------
        degenerates : bool
            True if E₂ = E∞ (all dᵣ vanish for r ≥ 2)
        """
        if 2 not in self.pages:
            return False

        # Check if pages stabilize at r=2
        for r in range(2, len(self.pages)):
            if not self._pages_equal(self.pages[r], self.pages[2], tol):
                return False

        return True

    def _pages_equal(
        self,
        page1: Dict[Tuple[int, int], np.ndarray],
        page2: Dict[Tuple[int, int], np.ndarray],
        tol: float = 1e-8,
    ) -> bool:
        """Check if two pages are equal (same dimensions)."""
        if set(page1.keys()) != set(page2.keys()):
            return False

        for key in page1.keys():
            dim1 = page1[key].shape[1]
            dim2 = page2[key].shape[1]
            if abs(dim1 - dim2) > tol:
                return False

        return True

    def __repr__(self) -> str:
        """String representation."""
        n_pages = len(self.pages)
        degenerates = self.degenerates_at_E2()

        return (
            f"SpectralSequence(pages={n_pages}, "
            f"E₂_degeneration={degenerates})"
        )
