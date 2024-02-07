import xml.etree.ElementTree as ET
import pydicom as PDCM
import numpy as np
import cv2


def XML_to_Image(Path_XML, Path_DCM, uint):
    """
    Convert XML annotations to a binary mask image.

    Args:
        Path_XML (str): The path to the XML file containing the annotations.
        Path_DCM (str): The path to the DICOM image file.
        uint (int): The desired data type for the output mask image (8 or 16).

    Returns:
        numpy.ndarray: The binary mask image.

    """
    meta = PDCM.dcmread(Path_DCM)
    img = meta.pixel_array
    tree = ET.parse(Path_XML)
    root = tree.getroot()

    lesionCounter = 0
    points = []

    mask = np.zeros(img.shape)

    for elem in root:
        new = True
        for subelem in elem:
            if new: #la première ligne dans le xml de chaque lésion ne sert à rien donc on l'utilise pour incrémenter le nombre de lésions
                lesionCounter += 1
                #print("Lesion " + str(lesionCounter))
                new = False
                points = []
            if subelem.text != "0": #un nouveau point du contour de la lésion
                [x, y] = subelem.text.split(",")
                x = x.split(".")[0]
                y = y.split(".")[0]
                points.append(np.array([int(x), int(y)]))
        if not new:
            cv2.polylines(mask, np.int32([points]), True, (255, 255, 255), 1, cv2.LINE_AA) #contour de la lésion
            cv2.fillPoly(mask, np.int32([points]), (255, 255, 255), cv2.LINE_AA) #remplissage de la lésion

    #encode en bit/pixel en fonction de uint
    if uint == 8:
        return mask.astype('uint8')
    elif uint == 16:
        mask[mask > 0] = 2**16 - 1
        return mask.astype('uint16')