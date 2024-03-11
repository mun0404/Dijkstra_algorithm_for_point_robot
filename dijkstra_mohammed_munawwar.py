import cv2
import numpy as np
import heapq
# from google.colab.patches import cv2_imshow #(use this when you are using google colab)

''' Step 1 : First we define all the possible action from a point i.e. UP, DOWN, RIGHT, LEFT, UP-LEFT, UP-RIGHT, DOWN-LEFT and DOWN-RIGHT.
'''

# Function for moving UP
def U(row, col):
    new_row = row
    new_col = col - 1
    cost = 1
    return new_row, new_col, cost

# Function for moving UP-LEFT
def UL(row, col):
    new_row = row - 1
    new_col = col - 1
    cost = 1.4  # Diagonal move cost
    return new_row, new_col, cost

# Function for moving UP-RIGHT
def UR(row, col):
    new_row = row + 1
    new_col = col - 1
    cost = 1.4  # Diagonal move cost
    return new_row, new_col, cost

# Function for moving LEFT
def L(row, col):
    new_row = row - 1
    new_col = col
    cost = 1
    return new_row, new_col, cost

# Function for moving RIGHT
def R(row, col):
    new_row = row + 1
    new_col = col
    cost = 1
    return new_row, new_col, cost

# Function for moving DOWN
def D(row, col):
    new_row = row
    new_col = col + 1
    cost = 1
    return new_row, new_col, cost

# Function for moving DOWN-LEFT
def DL(row, col):
    new_row = row - 1
    new_col = col + 1
    cost = 1.4  # Diagonal move cost
    return new_row, new_col, cost

# Function for moving DOWN-RIGHT
def DR(row, col):
    new_row = row + 1
    new_col = col + 1
    cost = 1.4  # Diagonal move cost
    return new_row, new_col, cost


''' Step 2: we generat all the possible moves from the current state ifd 
 The canvas is created with various shapes, including a hexagon, rectangles, and a boundary.
Obstacle points are identified based on color, and the obstacle information is stored in 'obs_point'.
'''
# Defining canvas dimensions
canvas_width = 1200
canvas_height = 500

# Defining hexagon parameters
hexagon_size = 150
hexagon_center = (650, 250)

# Defining rectangle parameters
rect_width = 75
rect_height = 400

rect_top_left_1 = (100, 0)
rect_top_left_2 = (275, 100)

# Defining inverted C parameters
top_bottom_rect_width = 200
top_bottom_rect_height = 75

side_rect_width = 80
side_rect_height = 400

C_top_rect = (900, 50)
C_bottom_rect = (900, 375)
C_side_rect = (1020, 50)

# Creating a black canvas with a light grey boundary
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)  # Light grey background

# Calculating hexagon vertices
hexagon_vertices = []
for i in range(6):
    angle_rad = i * (2 * np.pi / 6)
    x = int(hexagon_center[0] + hexagon_size * np.sin(angle_rad))
    y = int(hexagon_center[1] + hexagon_size * np.cos(angle_rad))
    hexagon_vertices.append((x, y))

# Drawing hexagon on the canvas and filling it.
cv2.fillPoly(canvas, [np.array(hexagon_vertices)], color=(64, 64, 64))

# Drawing rectangle on the canvas
cv2.rectangle(canvas, rect_top_left_1, (rect_top_left_1[0] + rect_width, rect_top_left_1[1] + rect_height),
              color=(64, 64, 64), thickness=-1)
cv2.rectangle(canvas, rect_top_left_2, (rect_top_left_2[0] + rect_width, rect_top_left_2[1] + rect_height),
              color=(64, 64, 64), thickness=-1)

# Drawing inverted C shape
cv2.rectangle(canvas, C_top_rect, (C_top_rect[0] + top_bottom_rect_width, C_top_rect[1] + top_bottom_rect_height),
              color=(64, 64, 64), thickness=-1)
cv2.rectangle(canvas, C_bottom_rect, (C_bottom_rect[0] + top_bottom_rect_width, C_bottom_rect[1] + top_bottom_rect_height),
              color=(64, 64, 64), thickness=-1)
cv2.rectangle(canvas, C_side_rect, (C_side_rect[0] + side_rect_width, C_side_rect[1] + side_rect_height),
              color=(64, 64, 64), thickness=-1)

# Adding a 5mm thick clearance around the obstacles in light grey color
clearance = 5
boundary_color = (200, 200, 200)  # Light grey color

cv2.rectangle(canvas, (clearance, clearance),
              (canvas_width - clearance, canvas_height - clearance),
              color=boundary_color, thickness=clearance)

# Adding clearance around hexagon
cv2.polylines(canvas, [np.array(hexagon_vertices)], isClosed=True, color=boundary_color, thickness=clearance)

# Adding clearance around rectangles
cv2.rectangle(canvas, (rect_top_left_1[0] - clearance, rect_top_left_1[1] - clearance),
              (rect_top_left_1[0] + rect_width + clearance, rect_top_left_1[1] + rect_height + clearance),
              color=boundary_color, thickness=clearance)
cv2.rectangle(canvas, (rect_top_left_2[0] - clearance, rect_top_left_2[1] - clearance),
              (rect_top_left_2[0] + rect_width + clearance, rect_top_left_2[1] + rect_height + clearance),
              color=boundary_color, thickness=clearance)

# Adding clearance around inverted C shape
cv2.rectangle(canvas, (C_top_rect[0] - clearance, C_top_rect[1] - clearance),
              (C_top_rect[0] + top_bottom_rect_width + clearance, C_top_rect[1] + top_bottom_rect_height + clearance),
              color=boundary_color, thickness=clearance)
cv2.rectangle(canvas, (C_bottom_rect[0] - clearance, C_bottom_rect[1] - clearance),
              (C_bottom_rect[0] + top_bottom_rect_width + clearance, C_bottom_rect[1] + top_bottom_rect_height + clearance),
              color=boundary_color, thickness=clearance)
cv2.rectangle(canvas, (C_side_rect[0] - clearance, C_side_rect[1] - clearance),
              (C_side_rect[0] + side_rect_width + clearance, C_side_rect[1] + side_rect_height + clearance),
              color=boundary_color, thickness=clearance)

# Display the canvas with the hexagon, rectangle, and boundary
cv2.imshow("Initial canvas",canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Function to check for obstacles
obs_point = {}
for row in range(canvas_width):
    for col in range(canvas_height):
        if np.array_equal(canvas[col, row], [128, 128, 128]) or np.array_equal(canvas[col, row], [200, 200, 200]):
            obs_point[(row, col)] = True
        else:
            obs_point[(row, col)] = False


''' Step 3 : This function implements Dijkstra's algorithm to find the shortest path from the start to the goal.
          It uses a priority queue to explore nodes with the lowest cost first, updating costs and parents accordingly.
          The final path and total cost are reconstructed after reaching the goal.'''

# Function to get coordinates of points with a specific color
def check_goal(row, col):
    return (row, col) == goal

# Comment: This function checks if a given point's coordinates match the goal coordinates.

# Function to check if a point is valid
def is_valid(row, col):
    return (
            row is not None
            and col is not None
            and row >= 0
            and row < canvas_width
            and col >= 0
            and col < canvas_height
            and not obs_point[row, col]
    )

def possible_moves(row, col):
    actions = [U, UL, UR, L, R, D, DL, DR]
    valid_moves = []
    for action in actions:
        new_row, new_col, cost = action(row, col)
        if is_valid(new_row, new_col):
            valid_moves.append((new_row, new_col, cost))
    return valid_moves

''' Step 4 : Here we write a dijkstra algorithm to find the shortest path between the start point and goal. '''

# Dijkstra's algorithm implementation
def dijkstra(start, goal):
    visited = set()
    visited_nodes = []  # List to store visited nodes
    pq = [(0, start, None)]  # Tuple (cost, node, parent-node)
    parent_dict = {}  # Dictionary to store parents nodes
    cost_dict = {start: 0}  # Dictionary to store total cost

    while pq:
        cost, current, parent = heapq.heappop(pq)
        if current == goal:
            print("Goal reached!")
            parent_dict[current] = parent
            break
        if current not in visited and is_valid(*current):
            visited.add(current)
            visited_nodes.append(current)
            parent_dict[current] = parent
            for next_row, next_col, move_cost in possible_moves(*current):
                if (next_row, next_col) not in visited:
                    new_cost = cost + move_cost
                    heapq.heappush(pq, (new_cost, (next_row, next_col), current))
                    cost_dict[(next_row, next_col)] = new_cost

    # Reconstructing the path and calculating the total cost
    path = []
    total_cost = 0
    current = goal
    while current is not None:
        path.insert(0, current)
        total_cost += cost_dict.get(current, 0)  
        current = parent_dict.get(current, None) 

    return path, total_cost, visited_nodes

# New animate function
def animate(visited_nodes, shortest_path, start, goal, output_filename='path_animation.mp4', skip_frames=10):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec used to compress the frames
    fps = 20  # Frames per second
    size = (canvas_width, canvas_height)
    out = cv2.VideoWriter(output_filename, fourcc, fps, size)

    vis_canvas = canvas.copy()

    # Animate visited nodes with frame skipping
    for i, node in enumerate(visited_nodes):
        if i % skip_frames == 0:  # Write only every Nth frame
            cv2.circle(vis_canvas, (node[0], node[1]), 2, (0, 255, 0), -1)
            out.write(vis_canvas)  # Write frame to video

    # Animate shortest path without skipping
    for node in shortest_path:
        cv2.circle(vis_canvas, (node[0], node[1]), 2, (0, 0, 255), -1)
        out.write(vis_canvas)  # Write frame to video

    # Mark the start and goal after all animations
    cv2.circle(vis_canvas, start, 5, (255, 255, 0), -1) 
    cv2.circle(vis_canvas, goal, 5, (0, 255, 255), -1)  
    for _ in range(fps):  
        out.write(vis_canvas)

    # Release everything if job is finished
    out.release()

k = 0
s = True
while s == True and k <= 10:

  X_1 = int(input('Enter the x coordinate of start position: '))
  Y_1 = int(input('Enter the y coordinate of start position: '))
  
  X_2 = int(input('Enter the x coordinate of goal position: '))
  Y_2 = int(input('Enter the y coordinate of goal position: '))
  
  start = (X_1, Y_1)
  goal = (X_2, Y_2)

  if not is_valid(*start) or not is_valid(*goal):
    print('Start or goal position is invalid or inside an obstacle.')
    k += 1
    continue
  else:
    s = False
    print("The points dont go out of bounds or overlap with obstacles!")
    shortest_path, total_cost, visited_nodes = dijkstra(start, goal)
    # print(f"Total cost of the shortest path: {total_cost}")  # Optionally print the total cost
    animate(visited_nodes, shortest_path,start,goal)
