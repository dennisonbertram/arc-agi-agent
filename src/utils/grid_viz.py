"""Grid visualization utilities."""
import numpy as np

COLORS = [
    "\u2b1c", "\u1f532", "\U0001f7eb", "\U0001f7e7", "\u2b1b", "\u1f533",
    "\U0001f7ea", "\U0001f7e3", "\U0001f7e5", "\U0001f7e6", "\U0001f499",
    "\U0001f7e8", "\U0001f7e0", "\U0001f534", "\U0001f7e9", "\U0001f49c",
]

# Fallback ASCII color codes for terminals without emoji support
ASCII_COLORS = list("0123456789ABCDEF")


def grid_to_ascii(grid, max_size: int = 20) -> str:
    """Convert grid to colored ASCII representation."""
    if isinstance(grid, list):
        grid = np.array(grid)
    h, w = grid.shape[:2]
    h, w = min(h, max_size), min(w, max_size)
    lines = []
    for y in range(h):
        line = ""
        for x in range(w):
            val = int(grid[y, x]) % 16
            line += ASCII_COLORS[val]
        lines.append(line)
    return "\n".join(lines)


def print_grid(grid, max_size: int = 20):
    print(grid_to_ascii(grid, max_size))
