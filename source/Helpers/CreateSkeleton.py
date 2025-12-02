import numpy as np
import os
from PIL import Image

from .VectorizeSkeleton import VectorizeSkeleton
from .HelperFunctions import skeletonKey, vectorKey, pointsKey, linesKey, clusterKey, functionKey
from ..UserContent.FunctionMaps import PIPELINE_STEP_FUNCTION_MAP, METRIC_FUNCTION_MAP

def GenerateSkeleton(directory:str, fileName:str, parameters:list[dict], steps:list[str], pipelineSteps:dict) -> dict:
    """
    A function that generates a skeleton given an input file, parameters, and pipeline defintion.

    Args:
        directory (str): The directory where the input file is located.
        fileName (str): The name of the input file.
        parameters (list[dict]): A list of dictionaries where each entry in the list corresponds to the step in steps at the same index. Each key in the dictionary is the parameter name while the value is the parameter value.
        steps (list[str]): The list of step names in the current pipeline.
        pipelineSteps (dict): The definitions of each step that could be in the pipeline. The keys are the step names (given in the steps parameter) and the values are dictionaries that give the list of parameters used in the step and a key for the function that implements the step. These are read from configs/PipelineSteps.json.

    Returns:
        dict: A dictionary containing the skeletonized binary image, a vectorized version of the skeleton, and any metrics that were calculated. This data is stored in a JSON file inside {output directory}/Calculations.
        Each JSON file contains the results from all skeletonization pipelines that were run on each image. The skeletonized image is also saved separately.
        The vectorized version of the skeleton consists of points (list of floating point XY coordinates normalized to 0-1), polylines (list of indices into the list of points), and connected clusters (list of indices into the list of polylines).
    """

    filePath = os.path.join(directory, fileName)
    img = Image.open(filePath)

    imgArray = np.asarray(img, dtype=np.float64)

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