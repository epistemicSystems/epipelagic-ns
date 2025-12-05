# Contributing to Epipelagic-NS

Thank you for your interest in contributing to the Epipelagic Turbulence Research Framework!

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/epistemicSystems/epipelagic-ns.git
cd epipelagic-ns
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

### 4. Verify Installation

```bash
pytest tests/
```

## Development Workflow

### Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/unit/test_cascade_complex.py

# With coverage
pytest --cov=epipelagic --cov-report=html

# Skip slow tests
pytest -m "not slow"
```

### Code Quality

```bash
# Format code
black epipelagic/ tests/

# Lint
flake8 epipelagic/

# Type checking
mypy epipelagic/ --ignore-missing-imports
```

### Benchmarking

```bash
pytest tests/benchmarks/ --benchmark-only
```

## Code Standards

### Style Guide

- Follow PEP 8
- Line length: 100 characters
- Use type hints for function signatures
- Docstrings: NumPy format

### Example Function

```python
def compute_cohomology(
    complex: CascadeComplex,
    degree: int = 1,
) -> Tuple[np.ndarray, int]:
    """
    Compute cohomology group Hⁿ(C•).

    Parameters
    ----------
    complex : CascadeComplex
        Cascade complex to analyze
    degree : int
        Cohomology degree (0 or 1)

    Returns
    -------
    basis : ndarray
        Basis vectors for Hⁿ
    dimension : int
        Dimension of cohomology group

    Examples
    --------
    >>> complex = CascadeComplex(...)
    >>> basis, dim = compute_cohomology(complex, degree=1)
    """
    # Implementation
    pass
```

### Testing Requirements

- All new features require tests
- Unit tests for components
- Integration tests for workflows
- Maintain >90% coverage

## Multi-Agent Collaboration

This project follows a multi-agent research framework:

- **Sonnet 4.5**: Mathematical theory and proofs
- **Haiku 4.5**: Rapid prototyping and experiments
- **Code (You!)**: Production implementation

### Communication Protocol

When requesting help from other agents:

```markdown
FROM: Code
TO: Haiku
TYPE: Request
PRIORITY: Medium
CONTENT: "Need quick validation of Algorithm X with parameters Y"
```

## Research Phases

### Phase 1: Foundation (Current)

**Focus**: Core infrastructure, cascade solvers, basic cohomology

**Contributions Needed**:
- [ ] Optimize Taichi solver performance
- [ ] Additional cascade models (Leith, Lorenz)
- [ ] Visualization tools

### Phase 2: Topology

**Focus**: Persistent homology, dim(H¹ₑₚᵢ) validation

**Contributions Needed**:
- [ ] Gudhi integration
- [ ] Parallel topology computation
- [ ] DNS data processing pipeline

### Phase 3: Quantum

**Focus**: Quasi-particle formalism, Fock space

**Future Work**

### Phase 4: Langlands

**Focus**: Geometric correspondence

**Future Work**

## Pull Request Process

1. **Create Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code
   - Add tests
   - Update documentation

3. **Run Quality Checks**
   ```bash
   black .
   flake8 epipelagic/
   pytest
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

5. **Push and Create PR**
   ```bash
   git push -u origin feature/your-feature-name
   ```

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] No merge conflicts

## Documentation

### Adding Documentation

- Mathematical theory: `docs/theory/`
- API reference: Auto-generated from docstrings
- Tutorials: `docs/tutorials/`
- Examples: `examples/`

### Building Docs

```bash
cd docs/
make html
```

## Questions?

- Open an issue for bugs or feature requests
- Email: research@epistemic.systems
- Discussion forum: [GitHub Discussions]

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
