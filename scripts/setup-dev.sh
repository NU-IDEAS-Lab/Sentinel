#!/bin/bash

# Setup development environment with commit hooks

echo "Setting up development environment..."

# Install commitizen
pip install commitizen pre-commit

# Install pre-commit hooks
pre-commit install --hook-type commit-msg
pre-commit install

echo "Development environment setup complete!"
echo ""
echo "Commit message format examples:"
echo "  feat(gen_safety): add new safety rule validation"
echo "  fix(models): resolve LLM timeout issue"
echo "  docs: update README with new setup instructions"
echo "  test(safety_eval): add unit tests for CTL evaluation"
echo ""
echo "Available commit types:"
echo "  feat, fix, docs, style, refactor, test, chore, ci, perf, build"
