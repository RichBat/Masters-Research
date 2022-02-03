import numpy as np
from scipy import ndimage
from time import time
import pandas as pd
from skimage import measure
from skimage import io

import matplotlib.pyplot as plt

import trimesh

import TiffMetadata

import math

from scipy.spatial.transform import Rotation as R
from scipy.stats import multivariate_normal
from scipy.ndimage import zoom
from scipy import signal
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation
from scipy.stats import moment
from scipy.spatial import ConvexHull

import ImageAnalysis


def labelStack(binarisedImageStack, minVolume=40):
    (ar, countsArr) = np.unique(binarisedImageStack, return_counts=True)
    print(countsArr)
    if len(countsArr) > 2:
        print('You must provide a binarised stack')
        return

    # get all labels (similar to what I tried with watershed)
    labeled, numpatches = ndimage.label(binarisedImageStack)
    
    # since labels start from 1 use this range
    sizes = ndimage.sum(binarisedImageStack/np.max(binarisedImageStack), labeled, range(1, numpatches + 1))
    # print(np.sort(sizes.astype('uint32')))

    # to ensure "black background" is excluded add 1, and labels only start from 1
    filteredIndexes = np.where(sizes >= minVolume)[0] + 1

    filteredBinaryIndexes = np.zeros(numpatches + 1, np.uint8)
    filteredBinaryIndexes[filteredIndexes] = 1
    filteredBinary = filteredBinaryIndexes[labeled]

    labeledStack, numLabels = ndimage.label(filteredBinary)
    print("Initial num labels: {}, num lables after filter: {}".format(numpatches, numLabels))
    
    # sizes = ndimage.sum(filteredBinary/np.max(filteredBinary), labeledStack, range(1, numLabels + 1))
    # print(np.sort(sizes.astype('uint32')))

    return filteredBinary, labeledStack, numLabels


def stack3DTo4D(labeledStack, numLabels):
    print("\nstack3DTo4D")

    if(numLabels > 150):
        print("MEMORY WARNING: More than 150 labels, kernel algorithm could possibly run out of VRAM.")

    sliceArray = []
    label_startTime = time()

    sliceCount = 0
    print("Slice done: ", end='')
    for s in labeledStack:
        frame_label_matrix = np.zeros((numLabels + 1, s.shape[0], s.shape[1], 1), dtype=np.float32)

        for y in range(0, s.shape[0]):
            for x in range(0, s.shape[1]):
                if (s[y, x] != 0):  # not black
                    try:
                        frame_label_matrix[s[y, x], y, x, 0] = 1
                    except:
                        print("Not found: {} at x and y ({},{})".format(s[y, x], x, y))

                        # for i in frame2_label_matrix[0:50]:
        #    plt.imshow(i)
        # split labels, and process segment per segment

        sliceArray.append(frame_label_matrix)

        print(" {} ".format(sliceCount), end='')
        sliceCount += 1

    output = np.swapaxes(np.array(sliceArray)[:, :, :, :, 0], 0, 1)
    print("\nLabel Time: ", time() - label_startTime)

    return output

def fullStackToMesh(stackLabels, scaleVector=None):
    if scaleVector == None:
        scaleVector = [1,1,1,1]

    print(stackLabels.shape)
    properties = measure.regionprops(stackLabels)

    listOfCoords = properties[0].coords
    for index in range(1, len(properties)):
        listOfCoords = np.vstack((listOfCoords, properties[index].coords))

    # print(listOfCoords)
    return trimesh.voxel.base.ops.points_to_marching_cubes(listOfCoords).apply_transform(np.eye(4,4)*scaleVector)

def getMetadata(filename):
    return TiffMetadata.metadata(filename)

def exportMeshAsPng(mesh, path, rotation):
    if not path.lower().endswith('.png'):
        path = path + '.png'
    meshScene = mesh.scene()
    rotationMatrix = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    transformationMatrix = np.eye(4, 4)
    transformationMatrix[0:3, 0:3] = rotationMatrix
    meshScene.apply_transform(transformationMatrix)
    pngBytes = bytes(mesh.scene().save_image((500, 500)))
    file = open(path, "wb")
    file.write(pngBytes)
    file.close()

def exportLabelSummary(MIP_image, mesh_image, infoDict, canCalculateConvexHull):
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(MIP_image)
    axarr[1,0].imshow(mesh_image)
    axarr[1, 0].axis('off')

    label = infoDict["label"]
    if label < 10:
        label = "00" + str(label)
    elif label < 100:
        label = "0" + str(label)
    else:
        label = str(label)

    if canCalculateConvexHull:
        text = ("Axis lengths: {} \n" +
                "AspectRatio3D Main/Middle: {} \n" +
                "AspectRatio3D Main/Minor: {} \n" +
                "AspectRatio3D Middle/Minor: {} \n" +
                "convexity_3D_relative: {} \n" +
                "solidy_3D_relative: {} \n" +
                "form_factor_3D_relative: {} \n").format(
            str(np.round(2*np.sqrt(infoDict['eigenvalues']), 3)),
            np.round(infoDict["AspectRatio3D Main/Middle"], 3),
            np.round(infoDict['AspectRatio3D Main/Minor'], 3),
            np.round(infoDict['AspectRatio3D Middle/Minor'], 3),
            np.round(infoDict['convexity_3D_relative'], 3),
            np.round(infoDict['solidy_3D_relative'], 3),
            np.round(infoDict['form_factor_3D_relative'], 3))
    else:
        text = ("Axis lengths: {} \n" +
                "AspectRatio3D Main/Middle: {} \n" +
                "AspectRatio3D Main/Minor: {} \n" +
                "AspectRatio3D Middle/Minor: {} \n"+
                "form_factor_3D_relative: {} \n").format(
            str(np.round(2*np.sqrt(infoDict['eigenvalues']), 3)),
            np.round(infoDict["AspectRatio3D Main/Middle"], 3),
            np.round(infoDict['AspectRatio3D Main/Minor'], 3),
            np.round(infoDict['AspectRatio3D Middle/Minor'], 3),
            np.round(infoDict['form_factor_3D_relative'], 3))

    axarr[0,1].text(0, 0, text, size="small", backgroundcolor='white', alpha=1.0, ha='left', va='top')
    axarr[0, 1].axis('off')
    axarr[1, 1].axis('off')
    plt.savefig('D:\\PhD\\3DStructureImages\\panels\\{}.png'.format(label))
    plt.close()

def exportMesh(mesh, path, type):
    mesh.export(path, type)

def calculateAllParameters(stackLabels, stackIntensities, metadata, scaleFactor=1, exportStructures3D=False, exportPath='', andExportPanel=False, dataExportFileName=''):
    rescaledStackLabels = ImageAnalysis.rescaleStackXY(stackLabels, 1/scaleFactor, order=0) # scale to original size using nearest neighbour (as to not change labels)
    rescaledStackIntensities = ImageAnalysis.rescaleStackXY(stackIntensities, 1/scaleFactor)
    # print(rescaledStackLabels.dtype)
    # print(rescaledStackIntensities.dtype)
    
    properties3D = measure.regionprops(rescaledStackLabels.astype('uint32'), rescaledStackIntensities)

    labelPropertyList = []
    for pr in properties3D:
        print('Label num:', len(labelPropertyList))

        # convex_image produces an ndarray - 3d?
        propertiesDict = {'area': pr.area,
                          # same as np.sum(stack4D[1]), which is the total number of voxels i.e. volume
                          'bbox': pr.bbox,
                          'bbox_area': pr.bbox_area,
                          'centroid': pr.centroid,
                          'equivalent_diameter': pr.equivalent_diameter,
                          'euler_number': pr.euler_number,
                          'extent': pr.extent,
                          'filled_area': pr.filled_area,
                          # 'inertia_tensor': pr.inertia_tensor,
                          # 'inertia_tensor_eigvals': pr.inertia_tensor_eigvals,
                          'label': pr.label,
                          'major_axis_length': pr.major_axis_length,
                          'minor_axis_length': pr.minor_axis_length,
                          'aspect_ratio_major_minor': np.sqrt(
                              pr.inertia_tensor_eigvals[0] / pr.inertia_tensor_eigvals[-1]),
                          'min_intensity': pr.min_intensity,
                          'max_intensity': pr.max_intensity,
                          'mean_intensity': pr.mean_intensity,
                          # 'moments': pr.moments,
                          # 'moments_central': pr.moments_central,
                          # 'moments_normalized': pr.moments_normalized,
                          }

        canCalculateConvexHull = True
        try:
            propertiesDict['convex_area'] = pr.convex_area
            propertiesDict['solidity'] = pr.solidity
        except:
            print("\tCould not calculate hull, since volume is 2D")
            canCalculateConvexHull = False

        if pr._ndim <= 2:  # Only valid for 2D
            # eccentricity in this case seem to be 1 - aspect ratio
            propertiesDict[
                'eccentricity'] = pr.eccentricity  # how circular is it [0, 1) where 0 is a circle. Sphere? sqrt(1 - major / minor)
            propertiesDict['moments_hu'] = pr.moments_hu
            propertiesDict['perimeter'] = pr.perimeter
            propertiesDict['form_factor'] = pr.area * 4 * math.pi / (pr.perimeter ** 2)

        elif pr._ndim == 3:  # exactly 3D
            scaleZ = (metadata.zVoxelWidth / metadata.xVoxelWidth)
            meshRelative = trimesh.voxel.base.ops.points_to_marching_cubes(pr.coords) \
                .apply_transform(
                np.eye(4, 4) * np.array([scaleZ, 1, 1, 1]))  # since point order (Z, X, Y, W)

            writePath = exportPath
            if exportStructures3D:
                label = pr.label
                if label < 10:
                    label = "00" + str(label)
                elif label < 100:
                    label = "0" + str(label)
                else:
                    label = str(label)

                writePath += "{}.png".format(label)

                try:
                    exportMeshAsPng(meshRelative, writePath, [180, -120, 0])
                except:
                    try:
                        exportMeshAsPng(meshRelative, writePath, [180, -120, 0])
                    except:
                        print("Failed to export twice")


            '''
            # This gives the 'correct' answer for the eigen values, but I think it is less accurate since
            # it stretches the voxels (make them blocky) in z
            binStackImRelative = zoom(pr.image + 0, (scaleZ, 1, 1), prefilter=False)
            prRelative = measure.regionprops(binStackImRelative)[0]
            print(np.flip(meshRelative.principal_inertia_components)/prRelative.inertia_tensor_eigvals)
            '''
            eigenvalues = np.flip(meshRelative.principal_inertia_components)
            propertiesDict['eigenvalues'] = eigenvalues
            # calculate lengths from http://ee263.stanford.edu/lectures/ellipsoids.pdf
            # calcualte lengths from https://math.stackexchange.com/questions/581702/correspondence-between-eigenvalues-and-eigenvectors-in-ellipsoids
            propertiesDict['3D main axis length(unscaled)'] = 2 * np.sqrt(eigenvalues[0])
            propertiesDict['3D middle axis length(unscaled)'] = 2 * np.sqrt(eigenvalues[1])
            propertiesDict['3D minor axis length(unscaled)'] = 2 * np.sqrt(eigenvalues[2])

            propertiesDict['AspectRatio3D Main/Middle'] = (2 * np.sqrt(eigenvalues[0])) / (2 * np.sqrt(eigenvalues[1]))
            propertiesDict['AspectRatio3D Main/Minor'] = (2 * np.sqrt(eigenvalues[0])) / (2 * np.sqrt(eigenvalues[2]))
            propertiesDict['AspectRatio3D Middle/Minor'] = (2 * np.sqrt(eigenvalues[1])) / (2 * np.sqrt(eigenvalues[2]))

            try:
                propertiesDict['surface_area_3D_relative'] = meshRelative.area
                propertiesDict['volume_3D_relative'] = meshRelative.volume
                propertiesDict['extents_3D'] = meshRelative.extents

                if canCalculateConvexHull:
                    propertiesDict['convex_area_3D_relative'] = meshRelative.convex_hull.area
                    propertiesDict['convex_volume_3D_relative'] = meshRelative.convex_hull.volume
                    propertiesDict['convexity_3D_relative'] = meshRelative.convex_hull.area / meshRelative.area
                    propertiesDict['solidy_3D_relative'] = meshRelative.volume / meshRelative.convex_hull.volume

                propertiesDict['form_factor_3D_relative'] = ((36 * math.pi * meshRelative.volume ** 2) / (
                            meshRelative.area ** 3)) ** (1 / 3)

                if exportStructures3D and andExportPanel:
                    mesh_image = io.imread(writePath)
                    MIP_image = np.max(pr.image, axis=0)
                    exportLabelSummary(MIP_image, mesh_image, propertiesDict, canCalculateConvexHull)

            except:
                print('\tCould not extract iso surface')

        labelPropertyList.append(propertiesDict)

    dataFr = pd.DataFrame(labelPropertyList)

    if len(dataExportFileName) != 0:
        if dataExportFileName.endswith('.xlsx'):
            dataFr.to_excel(dataExportFileName)
        else:
            dataFr.to_excel(dataExportFileName + '.xlsx')

    return dataFr


def calculateAllParameters_scales(stackLabels, stackIntensities, XY_voxel_res_um, Z__voxel_res_um, XY_scaleFactor=1, exportStructures3D=False, exportPath='', andExportPanel=False, dataExportFileName=''):
    rescaledStackLabels = ImageAnalysis.rescaleStackXY(stackLabels, 1/XY_scaleFactor, order=0).astype('int32') # scale to original size using nearest neighbour (as to not change labels)
    rescaledStackIntensities = ImageAnalysis.rescaleStackXY(stackIntensities, 1/XY_scaleFactor).astype('int32')
    # print(np.unique(rescaledStackLabels))
    # print(np.unique(rescaledStackLabels.astype('int32')))
    # print(rescaledStackIntensities.dtype)    
    
    properties3D = measure.regionprops(rescaledStackLabels, rescaledStackIntensities)
    print(len(properties3D))

    labelPropertyList = []
    for pr in properties3D:
        # print('Label num:', len(labelPropertyList))

        # convex_image produces an ndarray - 3d?
        propertiesDict = {'area': pr.area,
                          # same as np.sum(stack4D[1]), which is the total number of voxels i.e. volume
                          'bbox': pr.bbox,
                          'bbox_area': pr.bbox_area,
                          'centroid': pr.centroid,
                          'equivalent_diameter': pr.equivalent_diameter,
                          'euler_number': pr.euler_number,
                          'extent': pr.extent,
                          'filled_area': pr.filled_area,
                          # 'inertia_tensor': pr.inertia_tensor,
                          # 'inertia_tensor_eigvals': pr.inertia_tensor_eigvals,
                          'label': pr.label,
                          'major_axis_length': pr.major_axis_length,
                          'minor_axis_length': pr.minor_axis_length,
                          'aspect_ratio_major_minor': np.sqrt(
                              pr.inertia_tensor_eigvals[0] / pr.inertia_tensor_eigvals[-1]),
                          'min_intensity': pr.min_intensity,
                          'max_intensity': pr.max_intensity,
                          'mean_intensity': pr.mean_intensity,
                          # 'moments': pr.moments,
                          # 'moments_central': pr.moments_central,
                          # 'moments_normalized': pr.moments_normalized,
                          }

        canCalculateConvexHull = True
        try:
            propertiesDict['convex_area'] = pr.convex_area
            propertiesDict['solidity'] = pr.solidity
        except:
            # print("\tCould not calculate hull, since volume is 2D")
            canCalculateConvexHull = False

        if pr._ndim <= 2:  # Only valid for 2D
            # eccentricity in this case seem to be 1 - aspect ratio
            propertiesDict[
                'eccentricity'] = pr.eccentricity  # how circular is it [0, 1) where 0 is a circle. Sphere? sqrt(1 - major / minor)
            propertiesDict['moments_hu'] = pr.moments_hu
            propertiesDict['perimeter'] = pr.perimeter
            propertiesDict['form_factor'] = pr.area * 4 * math.pi / (pr.perimeter ** 2)

        elif pr._ndim == 3:  # exactly 3D
            scaleZ = (Z__voxel_res_um / XY_voxel_res_um)
            meshRelative = trimesh.voxel.base.ops.points_to_marching_cubes(pr.coords) \
                .apply_transform(
                np.eye(4, 4) * np.array([scaleZ, 1, 1, 1]))  # since point order (Z, X, Y, W)

            writePath = exportPath
            if exportStructures3D:
                label = pr.label
                if label < 10:
                    label = "00" + str(label)
                elif label < 100:
                    label = "0" + str(label)
                else:
                    label = str(label)

                writePath += "{}.png".format(label)

                try:
                    exportMeshAsPng(meshRelative, writePath, [180, -120, 0])
                except:
                    try:
                        exportMeshAsPng(meshRelative, writePath, [180, -120, 0])
                    except:
                        print("Failed to export twice")


            '''
            # This gives the 'correct' answer for the eigen values, but I think it is less accurate since
            # it stretches the voxels (make them blocky) in z
            binStackImRelative = zoom(pr.image + 0, (scaleZ, 1, 1), prefilter=False)
            prRelative = measure.regionprops(binStackImRelative)[0]
            print(np.flip(meshRelative.principal_inertia_components)/prRelative.inertia_tensor_eigvals)
            '''
            eigenvalues = np.flip(meshRelative.principal_inertia_components)
            propertiesDict['eigenvalues'] = eigenvalues
            # calculate lengths from http://ee263.stanford.edu/lectures/ellipsoids.pdf
            # calcualte lengths from https://math.stackexchange.com/questions/581702/correspondence-between-eigenvalues-and-eigenvectors-in-ellipsoids
            propertiesDict['3D main axis length(unscaled)'] = 2 * np.sqrt(eigenvalues[0])
            propertiesDict['3D middle axis length(unscaled)'] = 2 * np.sqrt(eigenvalues[1])
            propertiesDict['3D minor axis length(unscaled)'] = 2 * np.sqrt(eigenvalues[2])

            propertiesDict['AspectRatio3D Main/Middle'] = (2 * np.sqrt(eigenvalues[0])) / (2 * np.sqrt(eigenvalues[1]))
            propertiesDict['AspectRatio3D Main/Minor'] = (2 * np.sqrt(eigenvalues[0])) / (2 * np.sqrt(eigenvalues[2]))
            propertiesDict['AspectRatio3D Middle/Minor'] = (2 * np.sqrt(eigenvalues[1])) / (2 * np.sqrt(eigenvalues[2]))

            try:
                propertiesDict['surface_area_3D_relative'] = meshRelative.area
                propertiesDict['volume_3D_relative'] = meshRelative.volume # this should be in voxel^3 and since a voxel has length 1x1x1 it could be considered unitless
                propertiesDict['volume_3D_um'] = meshRelative.volume * XY_voxel_res_um * XY_voxel_res_um * XY_voxel_res_um # I think this would give um^3
                propertiesDict['extents_3D'] = meshRelative.extents

                if canCalculateConvexHull:
                    propertiesDict['convex_area_3D_relative'] = meshRelative.convex_hull.area
                    propertiesDict['convex_volume_3D_relative'] = meshRelative.convex_hull.volume
                    propertiesDict['convexity_3D_relative'] = meshRelative.convex_hull.area / meshRelative.area
                    propertiesDict['solidy_3D_relative'] = meshRelative.volume / meshRelative.convex_hull.volume

                propertiesDict['form_factor_3D_relative'] = ((36 * math.pi * meshRelative.volume ** 2) / (
                            meshRelative.area ** 3)) ** (1 / 3)

                if exportStructures3D and andExportPanel:
                    mesh_image = io.imread(writePath)
                    MIP_image = np.max(pr.image, axis=0)
                    exportLabelSummary(MIP_image, mesh_image, propertiesDict, canCalculateConvexHull)

            except:
                print('\tLabel {} Could not extract iso surface'.format(len(labelPropertyList)))

        labelPropertyList.append(propertiesDict)

    dataFr = pd.DataFrame(labelPropertyList)
    # print(dataFr)

    if len(dataExportFileName) != 0:
        if dataExportFileName.endswith('.xlsx'):
            dataFr.to_excel(dataExportFileName)
        else:
            dataFr.to_excel(dataExportFileName + '.xlsx')

    return dataFr