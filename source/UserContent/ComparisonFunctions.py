from Helpers.HelperFunctions import TupleDistance

"""
The following is an example of a basic comparison function between two skeletons
This is used in the comparison window to display how similar a generated skeleton is when compared to a manually-uploaded skeleton

Each comparison function is given a vectorized version of the two skeletons, each represented as a tuple
The first element of each tuple is a list of lines, each line being an ordered list of indices into the list of points
The second element of each tuple is a list of points, where each is a tuple of XY coordinates

Once a comparison function is completed, it must be included in COMPARISON_FUNCTION_MAP in source/UserContent/FunctionMaps.py
Each function is associated with a key (in camel case) that will be formatted and used as the display name for that function
"""

def ExampleComparisonFunction(skeleton1:tuple[list[list[int]], list[tuple[float, float]]], skeleton2:tuple[list[list[int]], list[tuple[float, float]]]) -> float:
    #calculate the similarity or difference between the two skeletons
    
    return 0

def AvgDistanceToClosestPoint(skeleton1:tuple[list[list[int]], list[tuple[float, float]]], skeleton2:tuple[list[list[int]], list[tuple[float, float]]]) -> float:
    totalDistance = 0.0
    
    for point1 in skeleton1[1]:
        minDistance = float("inf")

        for point2 in skeleton2[1]:
            minDistance = min(minDistance, TupleDistance(point1, point2))

        totalDistance += minDistance

    return totalDistance / len(skeleton1[1])

def MaxDistanceToClosestPoint(skeleton1:tuple[list[list[int]], list[tuple[float, float]]], skeleton2:tuple[list[list[int]], list[tuple[float, float]]]) -> float:
    maxDistance = 0.0
    
    for point1 in skeleton1[1]:
        minDistance = float("inf")

        for point2 in skeleton2[1]:
            minDistance = min(minDistance, TupleDistance(point1, point2))

        maxDistance = max(maxDistance, minDistance)

    return maxDistance