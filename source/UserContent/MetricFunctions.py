import random
import numpy as np
from scipy.stats import linregress
import math
from collections import deque

from ..Helpers.HelperFunctions import TupleDistance

"""
The following three functions are generic examples of what you can do to calculate metrics for skeletonized images
Each generates a random number for different features in the image: the whole image, each cluster of connected polylines in an image, and each lines
They each get the same information: the skeletonized numpy array, the array immediately before skeletonization, then a vectorized version of the skeleton

The vectorized version of the skeleton consists of point XY coordinates, a list of lines which are each a list of point indices, and a list of clusters
which are each a list of line indices

When a function is called per image, it should return a single number
When a function is called per cluster, it should return a list of numbers the same length as the input list of clusters
When a function is called per line, it should return a list of numbers the same length as the input list of lines

Once a function is complete, it should be included in METRIC_FUNCTION_MAP in source/UserContent/FunctionMaps.py
It needs a key string in camel case (the key is formatted and used as the display name), an indicator of the type (per image, per cluster, or per line)
and a boolean as to whether it's calculated in image space
If you indicate the function is calculated in image space, SkelPy will scale it based on the image dimensions
"""

def RandomNumberPerImage(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> float:
    """
    This is an example function for calculating metrics for a skeleton. This should calculate something that will tell the user some information about the skeleton.

    All metric calculation functions should have the same structure as this.

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        float: This example function returns a float but others can return anything that wouldn't cause an error when input into the string constructor str().
        As mentioned in the documentation at the top of this file, metric functions that are run per image should return a single value, ones that are called
        for every line should return a list of values the same length as the lines list, and ones called per cluster should return a list of values the same
        length as the clusters list.
    """
    
    return random.uniform(0, 1)

def RandomNumberPerCluster(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> list[float]:
    result = []
    
    for _ in range(len(clusters)):
        result.append(random.uniform(0, 1))

    return result

def RandomNumberPerLine(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> list[float]:
    result = []
    
    for _ in range(len(lines)):
        result.append(random.uniform(0, 1))

    return result

def Sinuosity(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> list[float]:
    """
    Calculates the sinuosity of each line in the skeleton. The sinuosity is calculated as (total polyline length) / (distance from start to end point).

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        list[float]: The sinuosity of each line in the skeleton.
    """
    
    result = []

    for line in lines:
        straightLineDistance = TupleDistance(points[line[0]], points[line[-1]])

        totalLength = 0
        for i in range(len(line) - 1):
            totalLength += TupleDistance(points[line[i]], points[line[i + 1]])

        if abs(straightLineDistance) < 1e-5:
            result.append(0)
        else:
            result.append(totalLength / straightLineDistance)

    return result

#fractal dimension
def FractalDimension(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> float:
    """
    Calculates the fractal dimension of the skeleton using the box-counting method (https://en.wikipedia.org/wiki/Box_counting).

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        float: The fractal dimension of the skeleton.
    """
    
    # Ensure the array is binary
    array = np.array(skeleton, dtype=bool)

    # Get the array dimensions
    min_dim = min(array.shape)

    # Box sizes (powers of 2)
    box_sizes = 2 ** np.arange(int(np.log2(min_dim)))

    box_counts = []
    for box_size in box_sizes:
        # Count the number of boxes that contain at least one "1"
        box_count = 0
        for i in range(0, array.shape[0] - box_size + 1, box_size):
            for j in range(0, array.shape[1] - box_size + 1, box_size):
                if np.any(array[i:i+box_size, j:j+box_size]):
                    box_count += 1
        box_counts.append(box_count)
        
        if box_count == 0:
            return 0.0

    # Convert to numpy arrays
    box_sizes = np.array(box_sizes)
    box_counts = np.array(box_counts)

    # Use linear regression to fit a line to log(box_counts) vs log(1/box_size)
    slope, _, _, _, _ = linregress(np.log(1/box_sizes), np.log(box_counts))

    # The slope of the line is the fractal dimension
    return slope

#number of lines in image
def LineCountInImage(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> int:
    """
    Calculates the number of lines in the skeleton.

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        int: The number of lines in the skeleton.
    """
    
    return len(lines)

#number of clusters in image
def ClusterCountInImage(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> int:
    """
    Calculates the number of clusters in the skeleton.

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        int: The number of clusters in the skeleton.
    """
    
    return len(clusters)

#number of lines in each cluster
def LineCountInCluster(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> list[int]:
    """
    Calculates the number of lines in each cluster.

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        list[int]: The number of lines in each cluster.
    """
    
    result = [len(cluster) for cluster in clusters]
    return result

#average length of lines in cluster
def AverageLineLengthInCluster(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> list[float]:
    """
    Calculates the average length of the lines in each cluster.

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        list[int]: The average length of the lines in each cluster.
    """
    
    result = []

    for currentLines in clusters:
        totalLength = 0.0

        for lineIndex in currentLines:
            currentLine = lines[lineIndex]

            for i in range(len(currentLine) - 1):
                point1 = points[currentLine[i]]
                point2 = points[currentLine[i + 1]]

                segmentLength = math.sqrt(pow(point2[0] - point1[0], 2) + pow(point2[1] - point1[1], 2))

                totalLength += segmentLength

        averageLength = totalLength / len(currentLines)

        result.append(averageLength)

    return result

def CountConnectedPixels(grid: np.ndarray, start_row: int, start_col: int) -> int:
    """
    Uses graph-searching to count the number of white pixels connected to the pixel at index (start_row, start_col). If the input pixel is black, this returns 0.

    Args:
        grid (np.ndarray): The binary 2D array that is used when searching for neighbors. The shape of this array is (H, W).
        start_row (int): The first index for the input pixel.
        start_col (int): The second index for the input pixel.
        
    Returns:
        int: The number of white pixels connected to the input pixel.
    """
    
    if grid[start_row, start_col] != 1.0:
        return 0

    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    queue = deque()
    queue.append((start_row, start_col))
    visited[start_row, start_col] = True
    count = 1

    # Define 8 directions (including diagonals)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                not visited[nr, nc] and grid[nr, nc] == 1.0):
                visited[nr, nc] = True
                queue.append((nr, nc))
                count += 1

    return count

def DrawLine(startPosition:tuple[float, float], direction:tuple[float, float], dimension:tuple[int, int]) -> np.ndarray:
    """
    This draws a line of ones in a new array of zeros from one end of the image to the other.

    Args:
        startPosition (tuple[float, float]): The starting position in the range 0-1 for drawing the line. This is not an endpoint or a midpoint, just a guaranteed point on the line.
        direction (tuple[float, float]): The direction vector for the line. After drawing in one direction, this is reversed and used to draw in the other direction.
        dimension (tuple[int, int]): The dimension of the array to be generated.
        
    Returns:
        np.ndarray: The new binary array with a line of ones drawn.
    """
    
    result = np.zeros(dimension)

    rows, cols = dimension
    x0 = int(cols * startPosition[0])
    y0 = int(rows * startPosition[1])
    dx, dy = direction

    norm = (dx**2 + dy**2)**0.5
    if norm == 0:
        return result  # No direction
    dx /= norm
    dy /= norm

    def draw_one_direction(x, y, dx, dy):
        while 0 <= int(round(y)) < rows and 0 <= int(round(x)) < cols:
            result[int(round(y)), int(round(x))] = 1
            x += dx
            y += dy

    # Draw in both directions
    draw_one_direction(x0, y0, dx, dy)       # Forward
    draw_one_direction(x0, y0, -dx, -dy)     # Backward

    return result

def CalculateLineWidth(thresholdedImage:np.ndarray, direction:tuple[float, float], lineMidpoint:tuple[float, float]) -> float:
    """
    Calculates the width of a line at its midpoint given the midpoint and the image before skeletonization.
    A line is drawn orthogonal to the line being considered and then ANDed with the image before skeletonization.
    The width is calculated using the number of points in this ANDed image connected to the midpoint.

    Args:
        thresholdedImage (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        direction (tuple[float, float]): The direction perpendicular to the line being considered.
        lineMidpoint (tuple[float, float]): The midpoint of the line being considered, represented as XY coordinates in the range 0-1.
        
    Returns:
        float: The width of the line relative to the size of the image.
    """

    #draw line along direction
    lineDrawn = DrawLine(lineMidpoint, direction, thresholdedImage.shape)

    #AND thresholded image and the line
    andImage = np.logical_and(thresholdedImage > 0.5, lineDrawn > 0.5).astype(np.float64)

    #BFS to get the number of white pixels connected to starting point
    width = CountConnectedPixels(andImage, int(lineMidpoint[1] * thresholdedImage.shape[0]), int(lineMidpoint[0] * thresholdedImage.shape[1]))
    
    width /= max(thresholdedImage.shape[0], thresholdedImage.shape[1])

    return width

def CalculateWidthAtLineCenter(skeleton:np.ndarray, imgBeforeSkeleton:np.ndarray, lines:list[list[int]], points:list[tuple[float, float]], clusters:list[list[int]]) -> list[float]:
    """
    Calculates the width of each line, relative to the size of the image, at its midpoint.

    Args:
        skeleton (np.ndarry): The skeleton image, represented as a 2D binary array in the shape (H, W).
        imgBeforeSkeleton (np.ndarry): The image immediately before skeletonization in the pipeline. This should be a 2D binary array in the shape (H, W).
        lines (list[list[int]]): The list of lines in this skeleton, each represented as a list of indices into the points list.
        points (list[tuple[float, float]]): The list of points in this skeleton, each represented as a tuple of XY coordinates in the range 0-1.
        clusters (list[list[int]]): The list of connected polyline clusters in this skeleton, each represented as a list of indices into the lines list.
        
    Returns:
        list[int]: The widths of all lines at their midpoints.
    """
    
    result = []
    
    #loop through each line
    for line in lines:
        if len(line) < 2:
            result.append(0.0)

        #get center point of line
        numPointsInLine = len(line)
        centerPointIndex = line[numPointsInLine // 2]
        centerPoint = points[centerPointIndex]
        centerPoint = (centerPoint[0], 1 - centerPoint[1])

        lastPointIndex = line[(numPointsInLine // 2) - 1]
        lastPoint = points[lastPointIndex]
        lastPoint = (lastPoint[0], 1 - lastPoint[1])

        #get direction of line
        direction = (centerPoint[0] - lastPoint[0], centerPoint[1] - lastPoint[1])
        magnitude = math.sqrt(math.pow(direction[0], 2) + math.pow(direction[1], 2))
        direction = (direction[0] / magnitude, direction[1] / magnitude)

        #get orthogonal direction to line
        orthogonalDirection = (direction[1], -direction[0])

        #width function
        currentWidth = CalculateLineWidth(imgBeforeSkeleton, orthogonalDirection, centerPoint)

        #add to result
        result.append(currentWidth)
    
    return result