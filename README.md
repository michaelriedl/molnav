# MolNav
![Testing](https://github.com/michaelriedl/molnav/actions/workflows/pytest.yml/badge.svg)
![Formatting](https://github.com/michaelriedl/molnav/actions/workflows/ruff.yml/badge.svg)
[![Documentation](https://img.shields.io/badge/Documentation-blue?style=flat&logo=readthedocs)](https://michaelriedl.com/molnav/)

MolNav (Molecular Navigator): Navigating the future of molecular modeling.

This repository implements a number of state-of-the-art deep learning models for molecular modeling using PyTorch.

## Getting Started
To get started with MolNav, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/michaelriedl/molnav.git
```

2. Install the package:
```bash
uv sync --all-extras
```

Note: MolNav is not yet available on PyPI, so you need to install it from source.

## Documentation
To build the documentation locally, follow these steps:

```bash
# One-time setup
uv sync --all-extras

# Build documentation
cd docs
make html

# Or using sphinx-build directly
uv run sphinx-build -b html source build/html
```

then you can view the built documentation by opening:

```bash
# Navigate to:
docs/build/html/index.html
```

## Citations
If you use MolNav in your work, please cite it as follows:
```bibtex
@software{molnav,
  title={MolNav: Navigating the future of molecular modeling},
  author={Michael Riedl},
  year={2025},
  version={0.1.0},
  url={https://github.com/michaelriedl/molnav}
}
```

In the subsections below, you can find more information about the different models implemented in MolNav.

### Transformer Models
