# Dijkstra's Algorithm for a point robot

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy
- if using Google Colab for displaying images use cv2_imshow instead of cv2.imshow

## Usage

1. Install the required dependencies using:

    ```bash
    pip install opencv-python numpy
    ```

2. Run the script in a Python environment, such as Jupyter Notebook or any Python IDE.
3. Follow the on-screen instructions to input the start and goal coordinates.

## Script Components

- **Obstacles Definition:** The maze is defined with obstacles, including a hexagon, rectangles, and an inverted "C" shape with 5 mm of clearance on all the obstacles ad all the walls.
- **Dijkstra's Algorithm:** The script implements Dijkstra's algorithm for finding the shortest path on the maze using the cost system and considering obstacles.
- **Visualization:** The script visualizes the exploration process and the shortest path using OpenCV. It generates a video (dijkstra.mp4) that illustrates the algorithm's progress.

## User Input

- The script prompts the user to input the start and goal coordinates. Ensure the coordinates are valid and not inside obstacles.
- The script checks the validity of the start and goal positions and proceeds with the algorithm if they are valid.

## Output

- The script prints the shortest path, total cost, and visited nodes during the algorithm execution.
- A video file (dijkstra.mp4) is generated, showing the exploration process, visited nodes, and the final shortest path on the maze.
