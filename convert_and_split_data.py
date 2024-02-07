import os
import configparser
import glob

import tqdm
import numpy as np
from PIL import Image

from core.convert.convert import Dicom_to_png
from core.convert.data_split import split_data_and_create_json
from core.convert.draw_mask import XML_to_Image


if __name__=="__main__":
    
    config = configparser.ConfigParser()    
    config.read("config.ini")
    
    path_dicom = config["PATH"]["img_folder"]
    path_contourage = config["PATH"]["contourage_folder"]

    path_output_image = "data/Images"
    path_output_contourages = "data/Contourages"
    if not os.path.exists(path_output_image):
        os.makedirs(path_output_image)
    if not os.path.exists(path_output_contourages):
        os.makedirs(path_output_contourages)

    width = int(config["IMAGE"]["width"])
    height = int(config["IMAGE"]["height"])
    uint = int(config["IMAGE"]["uint"])
    
    # Get all the dicom files
    dicom_files = glob.glob(os.path.join(path_dicom, "*.dcm"))

    # Convert all the dicom files to png
    for dicom_file in tqdm.tqdm(dicom_files):
        print(dicom_file)
        Dicom_to_png(dicom_file, path_output_image+"/"+os.path.basename(dicom_file).split(".")[0]+".png", width=width, height=height, uint=uint)

    # Get all the png files
    png_files = glob.glob(os.path.join(path_output_image, "*.png"))

    # Create the mask for each png file
    for png_file in tqdm.tqdm(png_files):
        # if the xml file doesn't exist the mask should be a black image with the same size as the png image
        if not os.path.isfile(os.path.join(path_contourage, os.path.basename(png_file).split(".")[0]+".xml")):
            mask = np.zeros((int(height), int(width)), dtype=np.uint8 if uint==8 else np.uint16)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(path_output_contourages, os.path.basename(png_file)))
        else:
            mask = XML_to_Image(os.path.join(path_contourage, os.path.basename(png_file).split(".")[0]+".xml"), os.path.join(path_dicom, os.path.basename(png_file).split(".")[0]+".dcm"), uint)
            mask = Image.fromarray(mask)
            mask = mask.resize((width, height))
            mask.save(os.path.join(path_output_contourages, os.path.basename(png_file)))
            
        
    split_data_and_create_json()
    print("Done")

