"""
Informed Search: A* algorithm to solve the maze.
A* uses a heuristic function to guide the search toward the goal efficiently.
It is both complete and optimal when using an admissible heuristic (one that
never overestimates the true cost). We use Manhattan distance as the heuristic
since movement is restricted to 4 directions (no diagonals).
"""
import heapq

def read_maze(filename):
    """Read the maze from a text file and return it as a 2D list of characters."""
    with open(filename, 'r') as f:
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

def manhattan_distance(pos, goal):
    """
    Manhattan distance heuristic: |row1 - row2| + |col1 - col2|
    This is admissible because in a grid with 4-directional movement,
    you need at least this many steps to reach the goal (ignoring walls).
    Since it never overestimates, A* with this heuristic is guaranteed to
    find the optimal (shortest) path.
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def astar(maze, start, exit_pos):
    """
    A* Search Algorithm:
    - Maintains a priority queue ordered by f(n) = g(n) + h(n)
      where g(n) = actual cost from start to n
            h(n) = estimated cost from n to goal (Manhattan distance)
    - Expands the node with the lowest f(n) first
    - Guarantees the optimal path when the heuristic is admissible
    - More efficient than BFS because it focuses on promising directions
    """
    rows = len(maze)
    cols = len(maze[0])

    # Priority queue: (f_cost, g_cost, row, col, path)
    # f_cost = g_cost + heuristic — used for ordering in the priority queue
    # g_cost = actual cost from start — each move costs 1
    open_list = []
    h = manhattan_distance(start, exit_pos)
    heapq.heappush(open_list, (h, 0, start[0], start[1], [start]))

    # Dictionary to track the best known g_cost for each cell
    # If we reach a cell with a higher g_cost than already recorded, skip it
    best_g = {start: 0}

    # Four possible directions: up, down, left, right (no diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_list:
        f_cost, g_cost, row, col, path = heapq.heappop(open_list)

        # Check if we reached the exit
        if (row, col) == exit_pos:
            return path, g_cost

        # Skip if we've already found a better path to this cell
        if g_cost > best_g.get((row, col), float('inf')):
            continue

        # Explore all four neighbors
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # Check bounds and ensure it's a passage (not a wall 'O')
            if (0 <= new_row < rows and 0 <= new_col < cols
                    and maze[new_row][new_col] != 'O'):

                new_g = g_cost + 1  # Each move costs 1

                # Only explore this neighbor if we found a better path to it
                if new_g < best_g.get((new_row, new_col), float('inf')):
                    best_g[(new_row, new_col)] = new_g
                    new_h = manhattan_distance((new_row, new_col), exit_pos)
                    new_f = new_g + new_h
                    heapq.heappush(open_list, (new_f, new_g, new_row, new_col, path + [(new_row, new_col)]))

    # A* exhausted all reachable cells without finding the exit
    return None, -1

def print_maze_with_path(maze, path):
    """Print the maze with the solution path marked by '*'."""
    display = [row[:] for row in maze]
    for row, col in path:
        display[row][col] = '*'
    for row in display:
        print(''.join(row))

def main():
    print("=" * 50)
    print("Informed Search: A* with Manhattan Distance")
    print("=" * 50)

    maze = read_maze('maze.txt')
    start, exit_pos = find_start_and_exit(maze)

    print(f"Maze size: {len(maze)} rows x {len(maze[0])} cols")
    print(f"Start position (bottom-right): {start}")
    print(f"Exit position  (top-left):     {exit_pos}")
    print(f"Initial heuristic (Manhattan distance): {manhattan_distance(start, exit_pos)}")
    print("\nSearching for optimal path using A*...")

    # Run A* to find the optimal path
    result = astar(maze, start, exit_pos)
    path, cost = result

    if path is None:
        print("No solution found")
    else:
        print(f"Optimal path found! Length: {len(path)} cells\n")
        print("Maze with solution path marked by '*':")
        print("-" * 50)
        print_maze_with_path(maze, path)
        print("-" * 50)
        print(f"\nCost of path: {cost}")

if __name__ == '__main__':
    main()
