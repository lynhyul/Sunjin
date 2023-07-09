import cv2
import numpy as np

def approximate_polygon(polygons):
    # Convert,polygons to numpy array
    points = np.array(polygons)

    # Calculate convex hull
    hull = cv2.convexHull(points)

    # Find the longest edge in the convex hull
    max_edge = find_longest_edge(hull)

    # Divide the polygon into two triangles
    triangles = divide_polygon(hull, max_edge)

    # Remove the smaller triangle
    triangles = remove_smaller_triangle(triangles)

    # Get the vertices of the remaining triangles
    vertices = get_triangle_vertices(triangles)

    # Generate rectangles from triangle vertices
    rectangles = generate_rectangles(vertices)

    return rectangles

def find_longest_edge(points):
    max_distance = 0
    max_index = 0

    for i in range(len(points)):
        p1 = points[i][0]
        p2 = points[(i + 1) % len(points)][0]

        distance = np.linalg.norm(p2 - p1)

        if distance > max_distance:
            max_distance = distance
            max_index = i

    return max_index

def divide_polygon(points, index):
    triangles = []

    for i in range(len(points)):
        p1 = points[i][0]
        p2 = points[(i + 1) % len(points)][0]

        if i == index:
            triangles.append([points[:index+1]])
            triangles.append([points[index:]])
            break

        triangles.append([[p1, p2]])

    return triangles

def remove_smaller_triangle(triangles):
    areas = []

    for triangle in triangles:
        points = np.array(triangle[0])
        area = cv2.contourArea(points)
        areas.append(area)

    min_index = np.argmin(areas)
    triangles.pop(min_index)

    return triangles

def get_triangle_vertices(triangles):
    vertices = []

    for triangle in triangles:
        points = np.array(triangle[0])
        vertices.extend(points)

    return vertices

def generate_rectangles(vertices):
    rectangles = []

    for i in range(0, len(vertices), 3):
        rect = cv2.boundingRect(np.array(vertices[i:i+3]))
        rectangles.append(rect)

    return rectangles

print(approximate_polygon())