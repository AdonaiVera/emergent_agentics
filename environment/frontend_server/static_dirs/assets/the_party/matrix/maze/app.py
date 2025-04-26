import csv
import numpy as np

def convert_maze_numbers(input_file, output_file, allowed_numbers):
    # Read the CSV file
    with open(input_file, 'r') as file:
        reader = csv.reader((line.replace(' ', '') for line in file))
        maze = np.array([[int(cell) for cell in row if cell] for row in reader])
    
    # Create a mask for numbers that are not in allowed_numbers
    # This will convert any number not in allowed_numbers to 0
    for i in range(maze.shape[0]):
        for j in range(maze.shape[1]):
            if maze[i,j] not in allowed_numbers:
                maze[i,j] = 0
    
    # Save the modified maze
    np.savetxt(output_file, maze, delimiter=',', fmt='%d')
    
    # Print summary of changes
    original_numbers = np.unique(maze)
    print(f"Numbers found in original maze: {sorted(list(original_numbers))}")
    print(f"Numbers kept (allowed): {sorted(allowed_numbers)}")
    print(f"Modified maze saved to: {output_file}")



'''
# Arena maze
allowed_numbers = [0, 32172, 32182, 32192, 32202, 32143, 32153, 32163, 32173, 32183]  
input_file = "arena_maze.csv"  # Your input file
output_file = "arena_maze_modified.csv"  # Where to save the result

# Game Object Maze
allowed_numbers = [0, 32227, 32237, 32247, 32257, 32267, 32277, 32208, 32218, 32228, 32238, 32248, 32258, 32268, 32278, 32209, 32219, 32229, 32239, 32249, 32259, 32269, 32279, 32210, 32220, 32230, 32240, 32250, 32260, 32270, 32280, 32211, 32221, 32231, 32241, 32251, 32261, 32271, 32281, 32212, 32222, 32232, 32242, 32252, 32262, 32272, 32282]
input_file = "game_object_maze.csv"  # Your input file
output_file = "game_object_maze_modified.csv"  # Where to save the result


# Sector Maze
allowed_numbers = [0, 32177]
input_file = "sector_maze.csv"  # Your input file
output_file = "sector_maze_modified.csv"  # Where to save the result


# Spawning Location Maze
allowed_numbers = [0, 32313, 32323, 32294, 32304, 32314, 32324]
input_file = "spawning_location_maze.csv"  # Your input file
output_file = "spawning_location_maze_modified.csv"  # Where to save the result
'''

# Spawning Location Maze
allowed_numbers = [0, 32313, 32323, 32294, 32304, 32314, 32324]
input_file = "collision_maze.csv"  # Your input file
output_file = "collision_maze_modified.csv"  # Where to save the result


convert_maze_numbers(input_file, output_file, allowed_numbers)