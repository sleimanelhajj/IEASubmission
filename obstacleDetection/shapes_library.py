# 🟥 Square Shape (16 positions)
SQUARE = [
    (0, 3), (1, 3), (2, 3), (3, 3), 
    (0, 4), (1, 4), (2, 4), (3, 4),
    (0, 5), (1, 5), (2, 5), (3, 5),
    (0, 6), (1, 6), (2, 6), (3, 6)
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

# 🔺 Triangle Shape
TRIANGLE = [
    (2, 5),
    (3, 4), (3, 5), (3, 6),
    (4, 3), (4, 4), (4, 5), (4, 6), (4, 7),
    (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)
]



# 🎭 Store all shapes in a dictionary
SHAPES = {
    "Square": SQUARE,
    "Circle": CIRCLE,
    "Triangle": TRIANGLE
    
}

def shift_shape_coords(shape_coords, grid_size):
    """Move shape to be centered in the grid and ensure it stays within bounds."""
    rows, cols = grid_size
    
    min_r, max_r = min(r for r, _ in shape_coords), max(r for r, _ in shape_coords)
    min_c, max_c = min(c for _, c in shape_coords), max(c for _, c in shape_coords)
    
    shape_height = max_r - min_r + 1
    shape_width = max_c - min_c + 1
    
    row_offset = max(0, (rows - shape_height) // 2 - min_r)
    col_offset = max(0, (cols - shape_width) // 2 - min_c)
    
    shifted_coords = []
    for r, c in shape_coords:
        new_r, new_c = r + row_offset, c + col_offset
        if 0 <= new_r < rows and 0 <= new_c < cols:
            shifted_coords.append((new_r, new_c))
    
    return shifted_coords
