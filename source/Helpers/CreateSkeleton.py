import numpy as np
import os
from PIL import Image

from .VectorizeSkeleton import VectorizeSkeleton
from .HelperFunctions import skeletonKey, vectorKey, pointsKey, linesKey, clusterKey, functionKey
from ..UserContent.FunctionMaps import PIPELINE_STEP_FUNCTION_MAP, METRIC_FUNCTION_MAP

def GenerateSkeleton(directory:str, fileName:str, parameters:list[dict], steps:list, pipelineSteps:dict) -> dict:
    filePath = os.path.join(directory, fileName)
    img = Image.open(filePath)

    originalImageArray = np.asarray(img, dtype=np.float64)

    if originalImageArray.ndim > 2:
        if originalImageArray.shape[-1] == 4:
            originalImageArray = originalImageArray[:, :, :3]

        originalImageArray = np.mean(originalImageArray, axis=-1)

    imgArray = np.copy(originalImageArray)

    maxValue = np.max(imgArray)
    minValue = np.min(imgArray)
    imgArray -= minValue
    maxValue -= minValue
    imgArray /= maxValue

    originalImageArray -= minValue
    originalImageArray /= maxValue

    #call all the functions
    for i, step in enumerate(steps):
        stepFunctionKey = pipelineSteps[step]["function"]

        imgArray = PIPELINE_STEP_FUNCTION_MAP[stepFunctionKey](imgArray, parameters[i])

    skeletonImg = PIPELINE_STEP_FUNCTION_MAP["skeletonize"](imgArray, {})

    result = {}
    result[skeletonKey] = np.asarray(skeletonImg, dtype=np.float64)

    lines, points, clusters = VectorizeSkeleton(skeletonImg)

    vectors = {
        linesKey: lines,
        pointsKey: points,
        clusterKey: clusters
    }

    result[vectorKey] = vectors

    for key in METRIC_FUNCTION_MAP:
        result[key] = METRIC_FUNCTION_MAP[key][functionKey](skeletonImg, imgArray, lines, points, clusters)
        
    print(f"Created skeleton for {fileName}")

    return result