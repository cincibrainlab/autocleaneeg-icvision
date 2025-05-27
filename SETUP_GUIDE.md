# Python Package Template Setup Guide

This guide will help you customize this professional Python package template for your specific project.

## Quick Start Checklist

### 1. Project Naming and Metadata

Replace all instances of the following placeholders:

- [ ] `your-package-name` → Your actual package name (e.g., `awesome-data-processor`)
- [ ] `your_package` → Your Python module name (e.g., `awesome_data_processor`)
- [ ] `yourusername` → Your GitHub username
- [ ] `Your Name` → Your actual name
- [ ] `your.email@example.com` → Your email address

**Files to update:**
- `pyproject.toml` (project name, description, URLs, author info)
- `README.md` (all references to package name and URLs)
- `src/your_package/` → Rename directory to your module name
- All Python files (import statements, references)
- `.github/workflows/ci.yml` (coverage paths, test commands)
- `Makefile` (package references)
- `tox.ini` (package references)

### 2. Customize Package Functionality

#### Core Module (`src/your_package/core.py`)
- Replace `YourMainClass` with your actual main class
- Replace `YourSecondaryClass` with your secondary class (or remove if not needed)
- Implement your actual business logic
- Update method signatures and functionality

#### Utils Module (`src/your_package/utils.py`)
- Replace `utility_function` with your actual utility functions
- Keep or modify configuration management functions as needed
- Add your own utility functions

#### CLI Module (`src/your_package/cli.py`)
- Customize CLI commands for your use case
- Update argument parsers for your specific needs
- Modify command functions to match your functionality

### 3. Update Documentation

#### README.md
- [ ] Update project description and features
- [ ] Replace example code with your actual usage examples
- [ ] Update API documentation
- [ ] Add your specific installation instructions
- [ ] Update badge URLs

#### CHANGELOG.md
- [ ] Update version history
- [ ] Add your initial features

#### CONTRIBUTING.md
- [ ] Review and customize contribution guidelines
- [ ] Update contact information

### 4. Configure Development Tools

#### pyproject.toml
- [ ] Update dependencies for your project needs
- [ ] Adjust Python version requirements
- [ ] Customize tool configurations (Black, MyPy, etc.)
- [ ] Update classifiers and keywords

#### GitHub Actions (`.github/workflows/ci.yml`)
- [ ] Verify Python versions to test against
- [ ] Update package name in coverage commands
- [ ] Set up PyPI secrets (see Publishing section below)

### 5. Testing

#### Update Tests
- [ ] Replace test files in `tests/` with your actual tests
- [ ] Update test imports and references
- [ ] Ensure tests cover your functionality
- [ ] Maintain 100% coverage

#### Run Tests
```bash
# Install development dependencies
make install-dev

# Run tests
make test

# Run full test suite across Python versions
make test-all
```

### 6. Set Up Version Control

```bash
# Initialize git (if not already done)
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Python package template"

# Add remote repository
git remote add origin https://github.com/yourusername/your-package-name.git

# Push to GitHub
git push -u origin main
```

## Publishing to PyPI

### 1. Create PyPI Account
- Create accounts on [PyPI](https://pypi.org/) and [Test PyPI](https://test.pypi.org/)
- Generate API tokens for both

### 2. Set Up GitHub Secrets
In your GitHub repository settings, add these secrets:
- `PYPI_API_TOKEN` - Your PyPI API token
- `TEST_PYPI_API_TOKEN` - Your Test PyPI API token

### 3. Manual Publishing

#### Test PyPI (Recommended First)
```bash
# Build package
make build

# Upload to Test PyPI
make publish-test

# Test installation from Test PyPI
pip install -i https://test.pypi.org/simple/ your-package-name
```

#### Production PyPI
```bash
# Build package
make build

# Upload to PyPI
make publish
```

### 4. Automated Publishing
The GitHub Actions workflow will automatically:
- Publish to Test PyPI on pushes to `develop` branch
- Publish to PyPI when you create a release

## Development Workflow

### Daily Development
```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test

# Full quality check
make check
```

### Pre-commit Hooks
The template includes pre-commit hooks that run automatically:
```bash
# Install hooks (done automatically with make install-dev)
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### Release Process
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Commit changes
4. Create git tag: `git tag v0.1.0`
5. Push tag: `git push origin v0.1.0`
6. Create GitHub release
7. Automated deployment will handle PyPI publishing

## Customization Examples

### Adding New Dependencies
```toml
# In pyproject.toml
dependencies = [
    "requests>=2.28.0",
    "pydantic>=1.10.0",
    "click>=8.0.0",
]
```

### Adding New CLI Commands
```python
# In src/your_package/cli.py
def new_command(args):
    """Your new command implementation."""
    pass

# Add to main() function:
new_parser = subparsers.add_parser("new", help="New command")
new_parser.add_argument("--option", help="Command option")
```

### Custom Configuration
```python
# In src/your_package/config.py
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for your package."""
    api_key: str
    timeout: int = 30
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """Create config from dictionary."""
        return cls(**data)
```

## Directory Structure

```
your-package-name/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── src/
│   └── your_package/
│       ├── __init__.py         # Package initialization
│       ├── core.py             # Main functionality
│       ├── utils.py            # Utility functions
│       └── cli.py              # Command-line interface
├── tests/
│   ├── __init__.py
│   ├── test_core.py            # Core module tests
│   └── test_utils.py           # Utils module tests
├── docs/                       # Documentation (optional)
├── .gitignore                  # Git ignore patterns
├── .pre-commit-config.yaml     # Pre-commit hooks
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── Makefile                    # Development commands
├── pyproject.toml              # Modern Python packaging
├── README.md                   # Project documentation
├── SETUP_GUIDE.md              # This file
└── tox.ini                     # Multi-version testing
```

## Best Practices

### Code Quality
- Maintain 100% test coverage
- Use type hints throughout
- Follow Google docstring convention
- Keep functions small and focused
- Use meaningful variable names

### Documentation
- Update docstrings for all public APIs
- Include examples in docstrings
- Keep README updated with current functionality
- Document breaking changes in CHANGELOG

### Version Management
- Follow [Semantic Versioning](https://semver.org/)
- Update version in `pyproject.toml` only
- Tag releases in git
- Maintain detailed changelog

### Security
- Regular dependency updates
- Use Bandit for security linting
- Never commit secrets or API keys
- Use GitHub secret scanning

## Troubleshooting

### Common Issues

**Import Errors**
- Ensure package name consistency across all files
- Check `PYTHONPATH` in development environment
- Verify `__init__.py` exports

**Test Failures**
- Update test imports after renaming
- Ensure test data matches new functionality
- Check coverage requirements

**CI/CD Issues**
- Verify GitHub secrets are set
- Check Python version compatibility
- Ensure all dependencies are specified

**Publishing Issues**
- Verify PyPI credentials
- Check package name availability
- Ensure version number is incremented

### Getting Help
- Check existing GitHub issues
- Review Python packaging documentation
- Consult PyPI publishing guide
- Ask questions in GitHub Discussions

## Next Steps

1. **Customize the template** for your specific use case
2. **Write comprehensive tests** for your functionality
3. **Update documentation** to reflect your project
4. **Set up CI/CD** with your GitHub repository
5. **Publish to PyPI** when ready for distribution

This template provides a solid foundation for professional Python packages. Customize it to fit your needs while maintaining the quality standards and best practices included.
