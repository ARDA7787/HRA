# Contributing to Physiological Anomaly Detection

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/ARDA7787/HRA.git
   cd HRA
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or .venv\Scripts\activate  # Windows
   ```
4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## 🛠️ Development Setup

### Code Quality Tools
```bash
# Install development dependencies
pip install black flake8 mypy pytest pytest-cov

# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/ --ignore-missing-imports

# Run tests
pytest tests/ -v --cov=src
```

### Pre-commit Hooks (Recommended)
```bash
pip install pre-commit
pre-commit install
```

## 📝 Contribution Guidelines

### Code Style
- Follow [PEP 8](https://pep8.org/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Add type hints where possible
- Write clear, descriptive docstrings

### Testing
- Write tests for new features
- Maintain test coverage above 80%
- Use descriptive test names
- Include both unit and integration tests

### Documentation
- Update README.md for new features
- Add docstrings to all functions and classes
- Include usage examples

## 🔄 Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/
   
   # Check formatting
   black --check src/
   
   # Check linting
   flake8 src/
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new anomaly detection model"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request** on GitHub

## 🎯 Areas for Contribution

### 🔬 Models & Algorithms
- New anomaly detection models
- Improved ensemble methods
- Online/streaming algorithms
- Transfer learning approaches

### 📊 Features & Preprocessing
- Additional signal processing techniques
- New feature engineering methods
- Data augmentation strategies
- Robust preprocessing pipelines

### 🌐 API & Infrastructure
- New API endpoints
- Performance optimizations
- Monitoring and logging improvements
- Deployment automation

### 📈 Visualization & Analysis
- Interactive dashboard enhancements
- New visualization types
- Explanation and interpretability tools
- Performance analysis utilities

### 🧪 Testing & Quality
- Additional test cases
- Performance benchmarks
- Integration tests
- Security testing

## 📋 Pull Request Guidelines

### Before Submitting
- [ ] Tests pass locally
- [ ] Code is formatted with Black
- [ ] No linting errors
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)

### PR Description Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Bug Reports

Use the GitHub issue tracker to report bugs. Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## 💡 Feature Requests

We welcome feature requests! Please:
- Check if the feature already exists
- Provide clear use case description
- Explain why the feature would be valuable
- Consider contributing the implementation

## 🏷️ Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add VAE-based anomaly detection model
fix: resolve memory leak in feature extraction
docs: update API documentation
test: add integration tests for ensemble methods
```

## 🔄 Release Process

1. **Version Bumping**: Update version in `setup.py`
2. **Changelog**: Update `CHANGELOG.md` with new features/fixes
3. **Testing**: Ensure all tests pass
4. **Documentation**: Update docs if needed
5. **Tag**: Create git tag for the release
6. **Deploy**: Automated deployment via GitHub Actions

## 📞 Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: Contact maintainers directly for sensitive issues

## 🙏 Recognition

Contributors will be:
- Listed in the README.md
- Mentioned in release notes
- Added to the AUTHORS file

Thank you for contributing to making physiological anomaly detection better! 🎉
