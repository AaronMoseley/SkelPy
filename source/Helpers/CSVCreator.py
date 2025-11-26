import csv
import os
from .HelperFunctions import skeletonKey, pointsKey, linesKey, clusterKey, vectorKey, functionTypeKey, imageTypeKey, lineTypeKey, clusterTypeKey
from ..UserContent.FunctionMaps import METRIC_FUNCTION_MAP

def GenerateCSVs(jsonObject:dict, baseFileName:str, outputDirectory:str) -> None:
    #create new directory
    currentCSVDirectory = os.path.join(outputDirectory, "Calculations", baseFileName + "_skeleton_csvs")
    os.makedirs(currentCSVDirectory, exist_ok=True)

    #create csv for base file info
        #original file name
        #sample
        #timestamp
        #different skeleton file names
    baseCSVPath = os.path.join(currentCSVDirectory, "fileInfo.csv")

    skeletonTypes = []

    baseCSVHeaders = []
    baseCSVData = []

    for key in jsonObject:
        if not isinstance(jsonObject[key], dict):
            baseCSVHeaders.append(key)
            baseCSVData.append(jsonObject[key])
        elif skeletonKey in jsonObject[key]:
            skeletonTypes.append(key)
            baseCSVHeaders.append(key)
            baseCSVData.append(jsonObject[key][skeletonKey])

    WriteCSV([baseCSVHeaders, baseCSVData], baseCSVPath)

    #loop through each skeleton
    for skeletonType in skeletonTypes:
        #create csv for points
        pointCSVPath = os.path.join(currentCSVDirectory, f"{skeletonType}_points.csv")
        pointData = [
            ["pointIndex", "x", "y"]
        ]

        for i, point in enumerate(jsonObject[skeletonType][vectorKey][pointsKey]):
            pointData.append([i, point[0], point[1]])

        WriteCSV(pointData, pointCSVPath)

        #create csv for line segments
        linesCSVPath = os.path.join(currentCSVDirectory, f"{skeletonType}_lines.csv")
        maxPointCount = 1

        for line in jsonObject[skeletonType][vectorKey][linesKey]:
            maxPointCount = max(maxPointCount, len(line))
            
        lineData = [
            ["lineIndex"] + [f"pointIndex{i}" for i in range(maxPointCount)]
        ]

        for i, lineSegment in enumerate(jsonObject[skeletonType][vectorKey][linesKey]):
            lineData.append([i] + lineSegment)

        WriteCSV(lineData, linesCSVPath)

        #create csv for clusters
        clusterCSVPath = os.path.join(currentCSVDirectory, f"{skeletonType}_clusters.csv")

        maxLineCount = 1

        for cluster in jsonObject[skeletonType][vectorKey][clusterKey]:
            maxLineCount = max(maxLineCount, len(cluster))

        clusterData = [
            ["clusterIndex"] + [f"lineIndex{i}" for i in range(maxLineCount)]
        ]

        for i, cluster in enumerate(jsonObject[skeletonType][vectorKey][clusterKey]):
            clusterData.append([i] + cluster)

        WriteCSV(clusterData, clusterCSVPath)

        #create csv for metadata
        metadataCSVPath = os.path.join(currentCSVDirectory, f"{skeletonType}_metadata.csv")
        metadataData = [
            ["name", "value", "lineIndex", "clusterIndex"]
        ]

        for metricFunctionKey in METRIC_FUNCTION_MAP:
            if not isinstance(jsonObject[skeletonType][metricFunctionKey], list) or METRIC_FUNCTION_MAP[metricFunctionKey][functionTypeKey] == imageTypeKey:
                metadataData.append([metricFunctionKey, jsonObject[skeletonType][metricFunctionKey], "", ""])
                continue

            if METRIC_FUNCTION_MAP[metricFunctionKey][functionTypeKey] == imageTypeKey:
                continue

            for i, value in enumerate(jsonObject[skeletonType][metricFunctionKey]):
                line = [metricFunctionKey, value]

                if METRIC_FUNCTION_MAP[metricFunctionKey][functionTypeKey] == lineTypeKey:
                    line.append(i)
                    line.append("")
                else:
                    line.append("")
                    line.append(i)

                metadataData.append(line)
        
        WriteCSV(metadataData, metadataCSVPath)

def WriteCSV(data:list, path:str) -> None:
    with open(path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)