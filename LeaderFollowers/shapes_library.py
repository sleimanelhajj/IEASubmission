# shape_definitions.py
# Description: Contains predefined shape coordinate lists (square, circle, triangle, etc.)
# and a helper function to shift (translate) those shapes so that they fit neatly within the grid.

# ---------------------------------------------------------
# Shape coordinate definitions
# ---------------------------------------------------------
# Each shape is defined as a list of (row, column) tuples. 
# They generally assume a certain bounding box (e.g., 5x5, 9x9, etc.) 
# but still need to be shifted or scaled to fit a particular grid.
# ---------------------------------------------------------

# 🟥 Square Shape (16 positions)
SQUARE = [
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 7), (2, 7), (3, 7), (4, 7),
    (4, 6), (4, 5), (4, 4), (4, 3),
    (3, 3), (2, 3), (1, 3)
]

# 🟥 Filled Square Shape (25 positions)
FILLED_SQUARE = [
    (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
    (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
    (2, 3), (2, 4), (2, 5), (2, 6), (2, 7),
    (3, 3), (3, 4), (3, 5), (3, 6), (3, 7),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7)
]

# 🔵 Circle Shape (16 positions)
CIRCLE = [
    (1, 4), (1, 5),
    (2, 3), (2, 6),
    (3, 2), (3, 7),
    (4, 1), (4, 8),
    (5, 1), (5, 8),
    (6, 2), (6, 7),
    (7, 3), (7, 6),
    (8, 4), (8, 5)
]

# 🔺 Triangle Shape (15 positions)
TRIANGLE = [
    (2, 5),
    (3, 4), (3, 5), (3, 6),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)
]


# ---------------------------------------------------------
# A dictionary of shape names mapped to their coordinate lists
# ---------------------------------------------------------
SHAPES = {
    "Square": SQUARE,
    "Square (Filled)": FILLED_SQUARE,
    "Circle": CIRCLE,
    "Triangle": TRIANGLE,
}

# ---------------------------------------------------------
# shift_shape_coords function
# ---------------------------------------------------------
# Purpose: Takes a list of shape coordinates and shifts (translates)
# them so that the shape is centered in the specified grid size,
# while ensuring no coordinates go out of grid bounds.
# ---------------------------------------------------------
def shift_shape_coords(shape_coords, grid_size):
    """
    Move shape to be centered in the grid and ensure it stays within bounds.
    
    shape_coords: List of (row, col) tuples for the shape.
    grid_size: A tuple (rows, cols) that defines the grid dimension.

    Returns:
        A shifted list of (row, col) tuples, with the shape 
        centered in the grid and clipped to remain within bounds.
    """
    # rows and cols extracted from the grid_size tuple
    rows, cols = grid_size

    # Determine the min and max row and column in the shape.
    min_r = min(r for r, _ in shape_coords)
    max_r = max(r for r, _ in shape_coords)
    min_c = min(c for _, c in shape_coords)
    max_c = max(c for _, c in shape_coords)

    # Calculate the shape's height and width based on those min/max values
    shape_height = max_r - min_r + 1
    shape_width = max_c - min_c + 1

    # row_offset and col_offset:
    # how far we move the shape from its default coordinates
    # so that it's roughly centered in the grid.
    # The max(0, ...) ensures we don't offset negatively (which would be out of bounds).
    row_offset = max(0, (rows - shape_height) // 2 - min_r)
    col_offset = max(0, (cols - shape_width) // 2 - min_c)

    # Build a new list of coordinates after applying the offset
    shifted_coords = []
    for r, c in shape_coords:
        new_r = r + row_offset
        new_c = c + col_offset

        # Only include the coordinate if it's within the grid bounds
        if 0 <= new_r < rows and 0 <= new_c < cols:
            shifted_coords.append((new_r, new_c))

    # Return the newly shifted set of shape coordinates
    return shifted_coords
