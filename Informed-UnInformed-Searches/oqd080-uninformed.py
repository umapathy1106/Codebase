"""
Uninformed Search: Breadth-First Search (BFS) to solve the maze.
BFS is complete (guaranteed to find a solution if one exists) and
finds the shortest path when all step costs are equal (cost = 1).
"""
from collections import deque

def read_maze(filename):
    """Read the maze from a text file and return it as a 2D list of characters."""
    with open(filename, 'r') as f:
        # Read each line and convert to list of characters
        # Preserve spaces by not stripping trailing whitespace
        maze = [list(line.rstrip('\n')) for line in f.readlines()]
    # Ensure all rows are the same length by padding shorter rows with spaces
    max_len = max(len(row) for row in maze)
    for row in maze:
        while len(row) < max_len:
            row.append(' ')
    return maze

def find_start_and_exit(maze):
    """
    Find start (bottom-right opening) and exit (top-left opening).
    The maze starts from bottom right and exits from top left.
    """
    rows = len(maze)
    cols = len(maze[0])

    # Find exit: top-left area — look for a space on the top row
    exit_pos = None
    for c in range(cols):
        if maze[0][c] == ' ':
            exit_pos = (0, c)
            break

    # Find start: bottom-right area — look for a space on the bottom row
    start_pos = None
    for c in range(cols - 1, -1, -1):
        if maze[rows - 1][c] == ' ':
            start_pos = (rows - 1, c)
            break

    return start_pos, exit_pos

def bfs(maze, start, exit_pos):
    """
    Breadth-First Search (BFS):
    - Explores all nodes at the current depth before moving to the next depth level.
    - Uses a FIFO queue (deque) to manage the frontier.
    - Guarantees the shortest path when all moves cost the same (cost = 1).
    - Time complexity: O(V + E) where V = cells, E = edges between adjacent cells.
    """
    rows = len(maze)
    cols = len(maze[0])

    # FIFO queue stores (row, col, path_so_far)
    queue = deque()
    queue.append((start[0], start[1], [start]))

    # Track visited cells to avoid revisiting
    visited = set()
    visited.add(start)

    # Four possible directions: up, down, left, right (no diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        row, col, path = queue.popleft()

        # Check if we reached the exit
        if (row, col) == exit_pos:
            return path

        # Explore all four neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check bounds, check if it's a passage (not a wall), check if not visited
            if (0 <= new_row < rows and 0 <= new_col < cols
                    and maze[new_row][new_col] != 'O'
                    and (new_row, new_col) not in visited):
                visited.add((new_row, new_col))
                queue.append((new_row, new_col, path + [(new_row, new_col)]))

    # BFS exhausted all reachable cells without finding the exit
    return None

def print_maze_with_path(maze, path):
    """Print the maze with the solution path marked by '*'."""
    # Create a copy of the maze so we don't modify the original
    display = [row[:] for row in maze]

    # Mark the path with '*'
    for row, col in path:
        display[row][col] = '*'

    # Print the maze
    for row in display:
        print(''.join(row))

def main():
    maze = read_maze('maze.txt')
    start, exit_pos = find_start_and_exit(maze)

    print(f"Start: {start}")
    print(f"Exit:  {exit_pos}")
    print()

    # Run BFS to find a path
    path = bfs(maze, start, exit_pos)

    if path is None:
        print("No solution found")
    else:
        print_maze_with_path(maze, path)

if __name__ == '__main__':
    main()
