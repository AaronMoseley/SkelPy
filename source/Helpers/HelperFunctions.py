from PySide6.QtGui import QPixmap, QPen, QPainter, QColor, QImage
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QMessageBox

import re
import numpy as np
import cv2
import math

skeletonKey = "skeleton"
originalImageKey = "originalImage"
vectorKey = "vectorized"
linesKey = "lines"
pointsKey = "points"
clusterKey = "clusters"
functionKey = "function"

functionTypeKey = "type"
imageTypeKey = "image"
clusterTypeKey = "cluster"
lineTypeKey = "line"

timestampKey = "timestamp"
sampleKey = "sample"

def CamelCaseToCapitalized(text:str) -> str:
    """
    Converts a camel case string to a capitalized string with spaces.

    Args:
        text (str): The camel case string to convert.

    Returns:
        str: The capitalized string with spaces.
    """
    return re.sub(r"([A-Z])", r" \1", text).title()

def IsPositiveNumeric(inputStr:str) -> bool:
    """
    Detects whether a string is both numeric and positive.

    Args:
        inputStr (str): The string to check.

    Returns:
        bool: Whether the string is both positive and numeric.
    """
    if len(inputStr) == 0:
        return False
    
    for character in inputStr:
        if not character.isdigit():
            return False
        
    return True

def DrawLinesOnPixmap(points:list[tuple[float, float]], lines:list[list[int]], 
                         width:int=249, height:int=249, colorMap:dict={}, line_color=QColor("white"), 
                         line_width:int=2, pixmap:QPixmap=None) -> QPixmap:
    """
    Draw lines of any color on a provided or all-black pixmap.

    Args:
        points (list[tuple[float, float]]): A list of points represented as XY coordinates normalized to 0-1.
        lines (list[list[int]]): A list of lines, each represented as a list of indices into the points list.
        width (int): The width of the new pixmap in pixels.
        height (int): The height of the new pixmap in pixels.
        colorMap (dict[int, QColor]): A map from line index to color of the line for when certain lines need to be colored differently than others.
        line_color (QColor): The line color for all lines not contained in colorMap.
        line_width (int): The width of each line in pixels.
        pixmap (QPixmap): An optional input pixmap that will be used as a background for drawing on.

    Returns:
        QPixmap: The pixmap with lines drawn on it.
    """

    if pixmap is None:
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("black"))

    painter = QPainter(pixmap)
    pen = QPen(line_color)
    pen.setWidth(line_width)
    painter.setPen(pen)

    # Helper to scale normalized points to pixel coordinates
    def scale_point(p):
        x = int(p[0] * max(width, height))
        y = int((1 - p[1]) * max(width, height))
        return QPoint(x, y)

    for lineIndex, line in enumerate(lines):
        if lineIndex in colorMap:
            pen.setColor(colorMap[lineIndex])
            painter.setPen(pen)
        else:
            pen.setColor(line_color)
            painter.setPen(pen)

        if len(line) < 2:
            continue
        for i in range(len(line) - 1):
            p1 = scale_point(points[line[i]])
            p2 = scale_point(points[line[i + 1]])
            painter.drawLine(p1, p2)

    painter.end()
    return pixmap

def ArrayToPixmap(array:np.ndarray, dimension:int=249, correctRange:bool=False, maxPoolDownSample:bool=False) -> QPixmap:
    """
    Converts an input numpy array to a QPixmap.

    Args:
        array (np.ndarray): The input array. The array shape is assumed to be (H, W, C) for RGB/RGBA images or (H, W) for grayscale images. If there are 4 channels, the 4th is assumed to be an alpha channel and removed.
        dimension (int): The maximum dimension for the new pixmap. For non-square images, this can become either the height or width but neither dimension will exceed this value.
        correctRange (bool): Whether the array is given in the correct range (0-255). If false, the array is assumed to be 0-1 and scaled up.
        maxPoolDownSample (bool): Whether to use max pool downsampling on grayscale images when scaling the pixmap image. If false or if the given image is RGB instead of grayscale, cubic resampling is used.

    Returns:
        QPixmap: The pixmap created from the input array.
    """

    arrayCopy = np.copy(array)

    if arrayCopy.ndim > 2 and arrayCopy.shape[-1] == 4:
        arrayCopy = arrayCopy[:, :, :3]

    if not correctRange:
        arrayCopy *= 255.0

    arrayCopy = np.asarray(arrayCopy, dtype=np.uint8)

    scaledHeight = dimension
    scaledWidth = dimension

    if arrayCopy.shape[0] > arrayCopy.shape[1]:
        #scale down height
        scaledWidth = int(dimension * (arrayCopy.shape[1] / arrayCopy.shape[0]))
    elif arrayCopy.shape[1] > arrayCopy.shape[0]:
        #scale down width
        scaledHeight = int(dimension * (arrayCopy.shape[0] / arrayCopy.shape[1]))

    if maxPoolDownSample and arrayCopy.ndim == 2:
        resizedImage = MaxPoolingDownsample(arrayCopy, (scaledHeight, scaledWidth))
    else:
        resizedImage = cv2.resize(arrayCopy, (scaledWidth, scaledHeight), interpolation=cv2.INTER_CUBIC)

    if arrayCopy.ndim == 2:
        rgbArray = cv2.cvtColor(resizedImage, cv2.COLOR_GRAY2RGB)
    else:
        rgbArray = resizedImage

    height, width, channels = rgbArray.shape
    bytesPerLine = width * channels
    qImage = QImage(rgbArray.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    qImage = qImage.copy()
    newPixmap = QPixmap.fromImage(qImage)
    return newPixmap

def MaxPoolingDownsample(image: np.ndarray, outputShape: tuple[int, int]) -> np.ndarray:
    """
    Downscales a grayscale image to a given shape.

    Args:
        image (np.ndarray): The input image to be downscaled. The shape of the array is assumed as (H, W).
        outputShape (tuple[int, int]): The new height and width for the resized image.

    Returns:
        np.ndarray: The resized image array.
    """

    input_h, input_w = image.shape
    output_h, output_w = outputShape

    pooled = np.zeros((output_h, output_w), dtype=np.uint8)

    for i in range(output_h):
        # Compute start and end row indices for pooling window
        start_i = int(i * input_h / output_h)
        end_i = int((i + 1) * input_h / output_h)

        for j in range(output_w):
            # Compute start and end column indices for pooling window
            start_j = int(j * input_w / output_w)
            end_j = int((j + 1) * input_w / output_w)

            # Extract pooling window and apply max
            window = image[start_i:end_i, start_j:end_j]
            pooled[i, j] = np.max(window)

    return pooled

def NormalizeImageArray(array:np.ndarray) -> np.ndarray:
    """
    Normalizes an image array to the 0-1 range. The function subtracts the minimum value then divides by the maximum value in the modified array.

    Args:
        array (np.ndarray): The input image array.

    Returns:
        np.ndarray: The normalized image array.
    """

    arrayCopy = np.copy(array)
    
    maxValue = np.max(arrayCopy)
    minValue = np.min(arrayCopy)
    arrayCopy -= minValue
    maxValue -= minValue
    arrayCopy /= maxValue

    return arrayCopy

def ConvertToGrayScale(array:np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to a grayscale image. If this is a grayscale image, the original array is returned. The input array is assumed to be of the shape (H, W, 3) or (H, W).

    Args:
        array (np.ndarray): The input image array.

    Returns:
        np.ndarray: The grayscale image array.
    """

    arrayCopy = np.copy(array)
    
    if arrayCopy.ndim > 2:
        return arrayCopy.mean(axis=-1)
    
    return arrayCopy

def to_camel_case(text:str) -> str:
    """
    Converts a string of text separated by spaces into a camel case string.

    Args:
        text (str): The input string.

    Returns:
        str: The camel case string.
    """

    words = text.split()
    if not words:
        return ""
    # Lowercase the first word, capitalize the rest
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def DistanceToLine(inputPoint:tuple[float, float], lineStartPoint:tuple[float, float], lineEndPoint:tuple[float, float]) -> float:
    """
    Finds the distance from an input point to a line given the start and end points of the line.

    Args:
        inputPoint (tuple[float, float]): The input point represented as XY coordinates.
        lineStartPoint (tuple[float, float]): The line's start point represented as XY coordinates.
        lineEndPoint (tuple[float, float]): The line's end point represented as XY coordinates.

    Returns:
        float: The distance from the input point to the line.
    """

    inputPoint = np.array(inputPoint, dtype=float)
    lineStartPoint = np.array(lineStartPoint, dtype=float)
    lineEndPoint = np.array(lineEndPoint, dtype=float)
    
    AB = lineEndPoint - lineStartPoint
    AP = inputPoint - lineStartPoint
    AB_len_squared = np.dot(AB, AB)

    if AB_len_squared == 0:
        # A and B are the same point
        return np.linalg.norm(inputPoint - lineStartPoint)

    # Project point P onto the line AB, computing parameter t of the projection
    t = np.dot(AP, AB) / AB_len_squared

    if t < 0.0:
        # Closest to point A
        closest_point = lineStartPoint
    elif t > 1.0:
        # Closest to point B
        closest_point = lineEndPoint
    else:
        # Projection falls on the segment
        closest_point = lineStartPoint + t * AB

    # Return distance from P to the closest point
    return np.linalg.norm(inputPoint - closest_point)

def ShowNotification(message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Notification")
        msg.setText(message)
        msg.exec()

def TupleDistance(point1:tuple[float, float], point2:tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two 2D points.

    Args:
        point1 (tuple[float, float]): The first point represented as XY coordinates.
        point2 (tuple[float, float]): The second point represented as XY coordinates.

    Returns:
        float: The distance from the first point to the second point.
    """

    return math.sqrt(math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2))