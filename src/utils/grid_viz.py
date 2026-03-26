"""Grid visualization utilities for ARC-AGI-3."""
from __future__ import annotations

from pathlib import Path
from typing import Any


# ARC color palette (index → RGB tuple).
ARC_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # black
    1: (0, 116, 217),    # blue
    2: (255, 65, 54),    # red
    3: (46, 204, 64),    # green
    4: (255, 220, 0),    # yellow
    5: (170, 170, 170),  # grey
    6: (240, 18, 190),   # fuchsia
    7: (255, 133, 27),   # orange
    8: (127, 219, 255),  # azure
    9: (135, 12, 37),    # maroon
}


def render_grid(
    grid: list[list[int]],
    cell_size: int = 40,
    title: str | None = None,
    show: bool = False,
    save_path: Path | None = None,
) -> Any:
    """Render an ARC grid as a colour image using matplotlib.

    Parameters
    ----------
    grid:
        2-D list of integer color values.
    cell_size:
        Pixel size of each cell in the rendered image.
    title:
        Optional title displayed above the grid.
    show:
        If ``True``, display the figure interactively.
    save_path:
        If provided, save the figure to this path.

    Returns
    -------
    Any
        The matplotlib ``Figure`` object.

    Raises
    ------
    NotImplementedError
        Until implemented.
    """
    raise NotImplementedError("render_grid is not yet implemented.")


def render_pair(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    predicted_grid: list[list[int]] | None = None,
    title: str | None = None,
    show: bool = False,
    save_path: Path | None = None,
) -> Any:
    """Render an input/output pair side-by-side, with optional prediction.

    Parameters
    ----------
    input_grid:
        The task input grid.
    output_grid:
        The ground-truth output grid.
    predicted_grid:
        Optional agent prediction to show as a third panel.
    title:
        Overall figure title.
    show:
        If ``True``, display the figure interactively.
    save_path:
        If provided, save the figure to this path.

    Returns
    -------
    Any
        The matplotlib ``Figure`` object.

    Raises
    ------
    NotImplementedError
        Until implemented.
    """
    raise NotImplementedError("render_pair is not yet implemented.")


def grid_to_image(grid: list[list[int]], cell_size: int = 40) -> Any:
    """Convert an ARC grid to a PIL Image.

    Parameters
    ----------
    grid:
        2-D list of integer color values.
    cell_size:
        Pixel size of each cell.

    Returns
    -------
    PIL.Image.Image

    Raises
    ------
    NotImplementedError
        Until implemented.
    """
    raise NotImplementedError("grid_to_image is not yet implemented.")
