# MolNav Documentation Quick Reference

## Build Documentation Locally

```bash
# One-time setup
uv sync --all-extras

# Build documentation
cd docs
make html

# Or using sphinx-build directly
uv run sphinx-build -b html source build/html
```

## View Built Documentation

```bash
# Navigate to:
docs/build/html/index.html
```

## Clean Build Files

```bash
cd docs
make clean
```

## Automatic Deployment

### GitHub Pages
- **URL**: `https://michaelriedl.github.io/molnav/`
- **Setup**: Go to repo Settings → Pages → Set source to "GitHub Actions"
- **Trigger**: Automatically deploys on push to main/master branch
- **Workflow**: `.github/workflows/docs.yml`

## Directory Structure

```
docs/
├── source/              # Source files (.rst, .md)
│   ├── conf.py         # Sphinx configuration
│   ├── index.rst       # Main page
│   └── ...             # Other documentation pages
├── build/              # Generated HTML (gitignored)
└── Makefile            # Build commands
```

## Common Commands

| Command | Description |
|---------|-------------|
| `make html` | Build HTML documentation |
| `make clean` | Clean build directory |
| `make latexpdf` | Build PDF (requires LaTeX) |
| `make epub` | Build EPUB format |
| `make linkcheck` | Check for broken links |
| `make help` | Show all available commands |

## Adding Documentation

1. Create new `.rst` file in `docs/source/`
2. Add to table of contents in `index.rst`:
   ```rst
   .. toctree::
      :maxdepth: 2
      
      your-new-page
   ```
3. Rebuild: `make html`

## Troubleshooting

**Build errors?**
```bash
cd docs
make clean
uv sync --all-extras
make html
```
