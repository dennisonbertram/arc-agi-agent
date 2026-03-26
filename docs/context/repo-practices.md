# Repository Practices

## Python Version

Python 3.12+. Use modern syntax throughout:
- `X | Y` union types instead of `Optional[X]` / `Union[X, Y]`
- `list[X]` / `dict[K, V]` / `tuple[X, ...]` instead of `List` / `Dict` / `Tuple` from `typing`
- `from __future__ import annotations` at the top of every module (deferred evaluation)
- `match` / `case` where it improves clarity

## Package Manager

Use `uv` exclusively. Never use `pip install` directly.

```bash
uv add <package>          # add a runtime dependency
uv add --dev <package>    # add a dev dependency
uv sync                   # install all deps from lockfile
uv run pytest             # run tests inside the venv
```

## Code Style

- Line length: 100 characters (enforced by `ruff`).
- Formatter: `ruff format`.
- Linter: `ruff check`.
- Run both before every commit:
  ```bash
  ruff check src/ tests/ scripts/
  ruff format src/ tests/ scripts/
  ```

## Imports

- Standard library first, then third-party, then local (`src.*`).
- All `src.*` imports use the package path (not relative imports).
- Guard any optional/external imports that may not be installed:
  ```python
  try:
      from arcengine import Frame
  except ImportError:
      Frame = Any
  ```

## Docstrings

NumPy-style docstrings on all public classes and functions:

```python
def foo(x: int, y: float) -> str:
    """Short one-line summary.

    Longer description if needed.

    Parameters
    ----------
    x:
        Description of x.
    y:
        Description of y.

    Returns
    -------
    str
        Description of return value.

    Raises
    ------
    ValueError
        When x is negative.
    """
```

## Stub Convention

Unimplemented methods raise `NotImplementedError` with a helpful message:

```python
def my_method(self) -> None:
    raise NotImplementedError("MyClass.my_method is not yet implemented.")
```

This convention allows tests to verify the interface contract before implementation.

## Testing

- Framework: `pytest`.
- Test files: `tests/test_<module_name>.py`.
- Run with: `pytest` or `uv run pytest`.
- Test coverage for all public methods.
- Use `pytest.raises(NotImplementedError)` to assert stub behavior.
- Keep tests fast — mock external dependencies (arc-agi SDK, GPU).

### Test file example

```python
# tests/test_config.py
from src.config import Config


def test_config_defaults():
    cfg = Config()
    assert cfg.num_colors == 16
    assert cfg.operation_mode == "OFFLINE"
```

## File Organization

```
src/
  config.py          # One global config object imported everywhere
  agent/             # Agent classes
  models/            # Neural network modules
  training/          # Trainers, buffers, reward shaping
  environment/       # ARC SDK wrapper and state processing
  evaluation/        # Evaluator and metrics
  utils/             # Shared utilities (logging, viz, serialization)
scripts/             # CLI entry points (not importable as a package)
tests/               # Mirror of src/ structure
docs/
  context/           # Intent and practices (this file)
  investigations/    # Research notes
  implementation/    # Implementation notes
```

## Configuration

All hyperparameters and paths live in `src/config.py`. Import `config` from there:

```python
from src.config import config

batch_size = config.batch_size
```

Override any setting via `.env` or environment variable (no prefix required):

```bash
LEARNING_RATE=1e-3 python scripts/train.py
```

## Checkpoints

- Save to `checkpoints/` (gitignored).
- Filename convention: `<run_name>_step<N>.pt` or `<run_name>_best.pt`.
- Always save both model weights and optimizer state.

## Logging

- Use `src.utils.logger.get_logger(__name__)` for console logs.
- Use `src.utils.logger.TBLogger` for TensorBoard metrics.
- Log scalar metrics at every update step; heavier logging (histograms, images) every N steps.

## Git Conventions

- Branch names: `feat/<description>`, `fix/<description>`, `refactor/<description>`.
- Commit messages: conventional commits style (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
- Keep commits atomic — one logical change per commit.
- Do not commit `.env`, checkpoint files, or logs (all gitignored).
