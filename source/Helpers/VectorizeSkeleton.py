import numpy as np
import math

from collections import defaultdict, Counter, deque

from .HelperFunctions import DistanceToLine, TupleDistance

def GetInitialLines(skeleton:np.ndarray) -> tuple[list, list]:
    """
    Uses breadth-first-search to detect connections between points and lines on a skeletonized binary image array. 
    The resulting list of points will contain every white pixel in the input array. This data is simplified in other functions.

    Args:
        skeleton (np.ndarray): The skeletonized image array. The shape of this array is (H, W) and all values are 0 or 1.

    Returns:
        tuple[list, list]: A list of points (tuples of XY coordinates) and a list of lines (each a list of indices into the list of points).
    """

    #create stack
    stack = []

    #create line assignment array, assign each point to a line or junction
    lineAssignments = np.zeros_like(skeleton)

    #create point assignment array
    pointAssignments = np.full_like(skeleton, -1)

    #create line list - line num: [point1, point2, ...]
    lines:list[list[int]] = []

    #create point list - point num: (x, y)
    points = []

    offsets = [-1, 0, 1]

    #create x and y indices
    #loop through array with x and y indices

    for y in range(skeleton.shape[0]):
        for x in range(skeleton.shape[1]):
            #if array = 1 and hasn't been assigned yet, add to stack
            if skeleton[y][x] == 1 and lineAssignments[y][x] == 0:
                stack.append((x, y, -1))

            #if stack empty, continue
            if len(stack) == 0:
                continue

            #stack element: (x index, y index, line index)
            #while stack not empty
            while len(stack) > 0:
                #pop first element
                pointX, pointY, lineInd = stack[0]
                del stack[0]

                if lineAssignments[pointY][pointX] == 1:
                    continue

                pointNum = len(points)

                #mark point on line
                if lineInd >= 0:
                    lines[lineInd].append(pointNum)
                else:
                    lineInd = len(lines)
                    newLine = [pointNum]
                    lines.append(newLine)

                #add point to point list
                points.append((pointX, pointY))

                #mark on point assignment array
                pointAssignments[pointY][pointX] = pointNum

                #mark on line assignment array
                lineAssignments[pointY][pointX] = 1

                #loop through neighbor offsets
                neighbors = []

                for yOffset in offsets:
                    for xOffset in offsets:
                        if pointY + yOffset < 0 or pointY + yOffset >= skeleton.shape[0]:
                            continue

                        if pointX + xOffset < 0 or pointX + xOffset >= skeleton.shape[1]:
                            continue

                        if xOffset == 0 and yOffset == 0:
                            continue

                        #count number of unassigned white pixels nearby
                        if skeleton[pointY + yOffset][pointX + xOffset] == 1 and lineAssignments[pointY + yOffset][pointX + xOffset] == 0:
                            neighbors.append((pointX + xOffset, pointY + yOffset))

                #determine if junction, unassigned white pixels nearby > 1
                isJunction = len(neighbors) > 1

                #loop through neighbors
                for neighbor in neighbors:
                    #if not junction, add to stack as an extension of the lines
                    if not isJunction:
                        stack.append((neighbor[0], neighbor[1], lineInd))
                    else:
                    #if junction, create new line, add current point as first point on the line, 
                    #add neighbor to stack as extension of that line
                        newLine = [pointNum]
                        stack.append((neighbor[0], neighbor[1], len(lines)))
                        lines.append(newLine)

    return lines, points

def RemoveShortLines(lines:list[list[int]], minLength:int) -> list[list[int]]:
    """
    Deletes any lines with fewer than minLength pixels. 
    At this point in vectorization, all pixels are represented as individual points, meaning the length of each line can be determined by the number of points it contains.

    Args:
        lines (list[list[int]]): A list of lines, each represented as a list of point indices.
        minLength (int): The minimum number of points a line is required to have to be kept.

    Returns:
        list[list[int]]: The reduced list of lines in the same format as the input parameter.
    """

    index = 0
    while index < len(lines):
        if len(lines[index]) < minLength:
            del lines[index]
            index -= 1

        index += 1

    return lines

def RDP(points:list[tuple[int, int]], polyline:list[int], epsilon:float) -> list[int]:
    """
    Simplifies a polyline using the Ramer-Douglas-Peucker algorithm (https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm).

    Args:
        points (list[tuple[int, int]]): A list of points, each represented as a tuple of XY coordinates.
        polyline (list[int]): An ordered list of indices into points.
        epsilon: Distance threshold for simplification.

    Returns:
        list[int]: A simplified version of the polyline, represented as a list of indices into points.
    """

    def rdp_recursive(start_idx, end_idx):
        max_dist = 0.0
        index = None

        start_point = points[polyline[start_idx]]
        end_point = points[polyline[end_idx]]

        for i in range(start_idx + 1, end_idx):
            dist = DistanceToLine(points[polyline[i]], start_point, end_point)
            if dist > max_dist:
                max_dist = dist
                index = i

        if max_dist > epsilon:
            # Recursive call
            left = rdp_recursive(start_idx, index)
            right = rdp_recursive(index, end_idx)
            return left[:-1] + right  # avoid duplicating index at the junction
        else:
            return [polyline[start_idx], polyline[end_idx]]

    if len(polyline) < 2:
        return polyline  # Not enough points to simplify

    return rdp_recursive(0, len(polyline) - 1)

def SimplifyLines(lines:list[list[int]], points:list[tuple[int, int]], epsilon:float) -> list[list[int]]:
    """
    A wrapper function that calls the RDP algorithm on all polylines it is given.

    Args:
        lines (list[list[int]]): A list of lines, each represented as a list of indices into points.
        points (list[tuple[int, int]]): A list of points, each represented as a tuple of XY coordinates.
        epsilon: Distance threshold for simplification.

    Returns:
        list[list[int]]: The modified version of lines.
    """

    for i in range(len(lines)):
        lines[i] = RDP(points, lines[i], epsilon)

    return lines

def RemoveUnusedPoints(points:list[tuple[int, int]], lines:list[list[int]]) -> tuple[list, list]:
    """
    Detects any points that are not included in a polyline and deletes them from the list of points.
    All polylines are updated to include the new indices of their points.

    Args:
        points (list[tuple[int, int]]): A list of points, each represented as a tuple of XY coordinates.
        lines (list[list[int]]): A list of lines, each represented as a list of indices into points.

    Returns:
        tuple[list, list]: The modified versions of lines and points respectively.
    """

    usedLines = set(index for line in lines for index in line)

    indexMapping = {}
    newPoints = []
    for new_idx, old_idx in enumerate(sorted(usedLines)):
        indexMapping[old_idx] = new_idx
        newPoints.append(points[old_idx])

    newLines = [[indexMapping[idx] for idx in line] for line in lines]

    return newLines, newPoints

def NormalizePoints(points:list[tuple[int, int]], width:int, height:int) -> list[tuple[float, float]]:
    """
    Normalizes all point coordinates to floats in the range 0-1. All point coordinates are divided by the maximum of the width and height.

    Args:
        points (list[tuple[int, int]]): A list of points, each represented as a tuple of XY coordinates.
        width (int): The width of the image in pixels.
        height (int): The height of the image in pixels.

    Returns:
        list[tuple[float, float]]: The normalized list of point coordinates.
    """
    
    divisor = max(width, height)
    
    newPoints = []
    for i in range(len(points)):
        point = (points[i][0] / divisor, 1 - (points[i][1] / divisor))
        newPoints.append(point)

    return newPoints

def MergeNearbyPoints(points: list[tuple[float, float]], polylines: list[list[int]], maxDistance: float) -> tuple[list, list]:
    """
    Merges all points that close enough to each other. When points are merged, the lines are edited to reflect the new indices.

    Args:
        points (list[tuple[int, int]]): A list of points, each represented as a tuple of XY coordinates.
        polylines (list[list[int]]): A list of lines, each represented as a list of indices into points.
        maxDistance (float): The maximum distance where points can be merged into each other.

    Returns:
        tuple[list, list]: The modified versions of lines and points respectively.
    """

    # Union-Find structure to group nearby points
    parent = list(range(len(points)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # Path compression
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Merge nearby points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if TupleDistance(points[i], points[j]) <= maxDistance:
                union(i, j)

    # Group indices by root
    clusters = {}
    for idx in range(len(points)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    # Compute new merged points
    newPoints = []
    index_mapping = {}  # old_index -> new_index
    for new_idx, cluster in enumerate(clusters.values()):
        x_avg = sum(points[i][0] for i in cluster) / len(cluster)
        y_avg = sum(points[i][1] for i in cluster) / len(cluster)
        newPoints.append((x_avg, y_avg))
        for old_idx in cluster:
            index_mapping[old_idx] = new_idx

    # Update polylines with new point indices
    newPolylines = [
        [index_mapping[idx] for idx in polyline]
        for polyline in polylines
    ]

    return newPolylines, newPoints

def MergePolylinesAtEndpoints(polylines: list[list[int]]) -> list[list[int]]:
    """
    Merges all polylines that share endpoints when the endpoint is only contained in two polylines.

    Args:
        polylines (list[list[int]]): A list of lines, each represented as a list of indices into points.

    Returns:
        list[list[int]]: The modified version of polylines.
    """

    # Count total occurrences of each point across all polylines
    point_usage = Counter(pt for poly in polylines for pt in poly)
    
    # Map endpoints to the polylines that use them (start or end)
    endpoint_map = defaultdict(list)
    for idx, poly in enumerate(polylines):
        if poly:
            endpoint_map[poly[0]].append(idx)
            endpoint_map[poly[-1]].append(idx)

    merged = [False] * len(polylines)
    result = []

    for i, poly_i in enumerate(polylines):
        if merged[i] or not poly_i:
            continue

        current_polyline = poly_i[:]
        changed = True

        while changed:
            changed = False
            for endpoint in [current_polyline[0], current_polyline[-1]]:
                # Endpoint must occur exactly twice total across all polylines
                if point_usage[endpoint] != 2:
                    continue

                # Must be used in exactly two polylines (ours and one other)
                connections = [j for j in endpoint_map[endpoint] if not merged[j]]
                if len(connections) != 2:
                    continue

                # Identify the other polyline
                other_idx = [j for j in connections if j != i]
                if not other_idx:
                    continue
                j = other_idx[0]
                poly_j = polylines[j]

                # Merge directionally based on the shared endpoint
                if endpoint == current_polyline[0]:
                    if endpoint == poly_j[0]:
                        poly_j = poly_j[::-1]
                    current_polyline = poly_j[:-1] + current_polyline
                elif endpoint == current_polyline[-1]:
                    if endpoint == poly_j[-1]:
                        poly_j = poly_j[::-1]
                    current_polyline = current_polyline + poly_j[1:]
                else:
                    continue  # Should never happen

                merged[j] = True
                changed = True

                # Update i to refer to the "new" polyline in current context
                i = i if i < j else j  # Keep lowest index as reference
                break

        merged[i] = True
        result.append(current_polyline)

    return result

def GetClusters(lines:list[list[int]]) -> list[list[int]]:
    """
    Detects all lines that are connected to each other in cluster groups.

    Args:
        lines (list[list[int]]): A list of lines, each represented as a list of indices into points.

    Returns:
        list[list[int]]: The list of clusters, each represented as a list of indices into the lines list.
    """

    # Step 1: Build point-to-polyline index
    point_to_polylines = {}
    for i, polyline in enumerate(lines):
        for point in polyline:
            if point not in point_to_polylines:
                point_to_polylines[point] = set()

            point_to_polylines[point].add(i)

    # Build connectivity graph between polylines
    graph = {}
    for i, polyline in enumerate(lines):
        if i not in graph:
            graph[i] = set()

        for point in polyline:
            for neighbor in point_to_polylines[point]:
                if neighbor != i:
                    graph[i].add(neighbor)

    # Step 3: Find connected components using BFS or DFS
    visited = set()
    clusters = []

    for i in range(len(lines)):
        if i not in visited:
            queue = deque([i])
            cluster_indices = []
            while queue:
                idx = queue.popleft()
                if idx not in visited:
                    visited.add(idx)
                    cluster_indices.append(idx)
                    queue.extend(graph[idx] - visited)
            # Create flat list of polylines for the cluster
            clusters.append(cluster_indices)

    return clusters

#lines, points, clusters
def VectorizeSkeleton(skeleton:np.ndarray) -> tuple[list, list, list]:
    """
    A function that takes a skeletonized binary image and vectorizes it so it can be displayed on pixmaps in the rest of the program.
    Clusters of lines are detected, as well as points and polylines.

    Args:
        skeleton (np.ndarray): A binary integer array that represents a skeletonized image.

    Returns:
        tuple[list, list, list]: The lists of lines (each represented as a list of point indices), points (each represented as a tuple of XY coordinates in the range 0-1),
        and conncted line clusters (each represented as a list of indices into the list of lines).
    """

    skeleton = np.asarray(skeleton, dtype=np.int64)
    
    #find initial lines
    lines, points = GetInitialLines(skeleton)

    lines = RemoveShortLines(lines, 5)

    points = NormalizePoints(points, skeleton.shape[1], skeleton.shape[0])

    maxErrorDist = 0.001
    #simplify lines
    lines = SimplifyLines(lines, points, maxErrorDist)

    lines, points = RemoveUnusedPoints(points, lines)

    lines, points = MergeNearbyPoints(points, lines, 0.004)

    lines = MergePolylinesAtEndpoints(lines)

    lines = RemoveZeroLengthLines(points, lines)

    clusters = GetClusters(lines)

    return lines, points, clusters

def RemoveZeroLengthLines(points: list[tuple[float, float]], lines: list[list[int]], minLength:float=0.01) -> list[list[int]]:
    """
    Removes any lines below a certain length.

    Args:
        points (list[tuple[int, int]]): A list of points, each represented as a tuple of XY coordinates.
        polylines (list[list[int]]): A list of lines, each represented as a list of indices into points.
        minLength (float): The minimum length required to keep any line.

    Returns:
        list[list[int]]: The reduced list of lines.
    """

    lineIndex = 0

    while lineIndex < len(lines):
        totalLength = 0.0

        lines[lineIndex] = RemoveDuplicatedPoints(lines[lineIndex])

        currentLine = lines[lineIndex]

        for i in range(len(currentLine) - 1):
            point1 = points[currentLine[i]]
            point2 = points[currentLine[i + 1]]

            segmentLength = TupleDistance(point1, point2)

            totalLength += segmentLength

        if totalLength < minLength:
            del lines[lineIndex]
        else:
            lineIndex += 1

    return lines

def RemoveDuplicatedPoints(polyline:list[int]) -> list[int]:
    """
    Removes any points that are duplicated in a line.

    Args:
        polyline (list[int]): The input polyline, represented as a list of point indices.

    Returns:
        list[int]: The reduced version of the polyline.
    """

    if not polyline:
        return []
    
    result = [polyline[0]]
    for item in polyline[1:]:
        if item != result[-1]:
            result.append(item)
    return result