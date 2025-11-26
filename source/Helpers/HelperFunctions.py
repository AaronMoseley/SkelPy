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

def camel_case_to_capitalized(text):
    """
    Converts a camel case string to a capitalized string with spaces.

    Args:
        text: The camel case string to convert.

    Returns:
        The capitalized string with spaces.
    """
    return re.sub(r"([A-Z])", r" \1", text).title()

def IsPositiveNumeric(inputStr:str) -> bool:
    if len(inputStr) == 0:
        return False
    
    for character in inputStr:
        if not character.isdigit():
            return False
        
    return True

def draw_lines_on_pixmap(points:list[tuple[float, float]], lines:list[list[int]], 
                         width:int=249, height:int=249, colorMap:dict={}, line_color=QColor("white"), line_width=2, pixmap:QPixmap=None):
    if pixmap is None:
        pixmap = QPixmap(width, height)
        pixmap.fill(QColor("black"))

    painter = QPainter(pixmap)
    pen = QPen(line_color)
    pen.setWidth(line_width)
    painter.setPen(pen)

    # Helper to scale normalized points to pixel coordinates
    def scale_point(p):
        x = int(p[0] * width)
        y = int((1 - p[1]) * height)
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
    arrayCopy = np.copy(array)

    if arrayCopy.ndim > 2:
        if arrayCopy.shape[-1] == 4:
            arrayCopy = arrayCopy[:, :, :3]

        arrayCopy = np.mean(arrayCopy, axis=-1)

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

    # Resize using OpenCV
    if not maxPoolDownSample:
        resized_gray = cv2.resize(arrayCopy, (scaledWidth, scaledHeight), interpolation=cv2.INTER_CUBIC)
    else:
        resized_gray = max_pooling_downsample(arrayCopy, (scaledHeight, scaledWidth))

    # Convert to RGB by stacking channels
    rgb_array = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2RGB)

    height, width, channels = rgb_array.shape
    bytesPerLine = width * channels
    qImage = QImage(rgb_array.data, width, height, bytesPerLine, QImage.Format.Format_RGB888)
    qImage = qImage.copy()
    newPixmap = QPixmap.fromImage(qImage)
    return newPixmap

def max_pooling_downsample(image: np.ndarray, output_shape: tuple) -> np.ndarray:
    """
    Downsamples a 2D grayscale image using max pooling, even when input
    dimensions are not divisible by the output dimensions.

    Parameters:
    - image (np.ndarray): 2D array of dtype np.uint8, shape (H, W)
    - output_shape (tuple): Target shape (new_H, new_W)

    Returns:
    - np.ndarray: Downsampled 2D array of shape output_shape, dtype np.uint8
    """
    input_h, input_w = image.shape
    output_h, output_w = output_shape

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
    arrayCopy = np.copy(array)
    
    maxValue = np.max(arrayCopy)
    minValue = np.min(arrayCopy)
    arrayCopy -= minValue
    maxValue -= minValue
    arrayCopy /= maxValue

    if arrayCopy.ndim > 2:
        return arrayCopy.mean(axis=-1)

    return arrayCopy

def to_camel_case(text:str):
    words = text.split()
    if not words:
        return ""
    # Lowercase the first word, capitalize the rest
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])

def DistanceToLine(P, A, B):
    P = np.array(P, dtype=float)
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    
    AB = B - A
    AP = P - A
    AB_len_squared = np.dot(AB, AB)

    if AB_len_squared == 0:
        # A and B are the same point
        return np.linalg.norm(P - A)

    # Project point P onto the line AB, computing parameter t of the projection
    t = np.dot(AP, AB) / AB_len_squared

    if t < 0.0:
        # Closest to point A
        closest_point = A
    elif t > 1.0:
        # Closest to point B
        closest_point = B
    else:
        # Projection falls on the segment
        closest_point = A + t * AB

    # Return distance from P to the closest point
    return np.linalg.norm(P - closest_point)

def ShowNotification(message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Notification")
        msg.setText(message)
        msg.exec()

def TupleDistance(point1:tuple[float, float], point2:tuple[float, float]) -> float:
    return math.sqrt(math.pow(point2[0] - point1[0], 2) + math.pow(point2[1] - point1[1], 2))