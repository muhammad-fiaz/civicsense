# Contributing to CivicSense

Thank you for your interest in contributing to CivicSense! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.10+
- [UV package manager](https://docs.astral.sh/uv/)
- Git

### Setup

```bash
git clone https://github.com/muhammad-fiaz/civicsense.git
cd civicsense
uv sync
```

### Running Tests

```bash
uv run pytest tests/ -v
```

### Linting

```bash
uv run ruff check civicsense/
uv run ruff format civicsense/
```

## Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Write Google-style docstrings
- Keep line length under 100 characters
- Use `ruff` for linting and formatting

## Reporting Issues

- Use the GitHub issue tracker
- Include steps to reproduce the issue
- Include your environment details (OS, Python version, UV version)

## License

By contributing, you agree that your contributions will be licensed under the GNU General Public License v3.0.
