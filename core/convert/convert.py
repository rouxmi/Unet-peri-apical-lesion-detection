import cv2
import numpy as np
import pydicom as PDCM
import numpy as np

def bilinear_resize_vectorized(image, height=512, width=1024, uint=8):
    """
    Resize a 2-D numpy array using bilinear interpolation.

    Parameters:
    - image: 2-D numpy array
        The input image to be resized.
    - height: int, optional (default=512)
        The desired height of the resized image.
    - width: int, optional (default=1024)
        The desired width of the resized image.
    - uint: int, optional (default=8)
        The data type of the resized image. Can be either 8 (uint8) or 16 (uint16).

    Returns:
    - resized: 2-D numpy array
        The resized image.

    Note:
    - The input image should be a 2-D numpy array.
    - The resized image will have the specified height and width.
    - The resized image will have the same data type as the input image.
    - Bilinear interpolation is used to calculate the pixel values of the resized image.
    """
    img_height, img_width = image.shape

    image = image.ravel()

    x_ratio = float(img_width - 1) / (width - 1) if width > 1 else 0
    y_ratio = float(img_height - 1) / (height - 1) if height > 1 else 0

    y, x = np.divmod(np.arange(height * width), width)

    x_l = np.floor(x_ratio * x).astype('int32')
    y_l = np.floor(y_ratio * y).astype('int32')

    x_h = np.ceil(x_ratio * x).astype('int32')
    y_h = np.ceil(y_ratio * y).astype('int32')

    x_weight = (x_ratio * x) - x_l
    y_weight = (y_ratio * y) - y_l

    a = image[y_l * img_width + x_l]
    b = image[y_l * img_width + x_h]
    c = image[y_h * img_width + x_l]
    d = image[y_h * img_width + x_h]

    resized = a * (1 - x_weight) * (1 - y_weight) + \
            b * x_weight * (1 - y_weight) + \
            c * y_weight * (1 - x_weight) + \
            d * x_weight * y_weight

    if uint == 8:
        return np.rint(resized.reshape(height, width)).astype(np.uint8)
    else:
        return np.rint(resized.reshape(height, width)).astype(np.uint16)
    
def resize_image(img, uint):
    """
    Resize the input image to a fixed size of 1024x2048 pixels by cropping the image on the sides.

    Args:
        img (numpy.ndarray): The input image as a numpy array.
        uint (int): The desired data type for the output image. Valid values are 8 and 16.

    Returns:
        numpy.ndarray: The resized image as a numpy array with the specified data type.
    """
    img = np.array(img)
    rows, cols = img.shape

    ecartX = int((rows-1024)/2)
    ecartY = int((cols - 2048) / 2)

    res = np.zeros([1024, 2048])

    for i in range(1024):
        for j in range(2048):
            res[i][j] = img[i+ecartX][j+ecartY]

    if uint == 8:
        return res.astype('uint8')
    elif uint == 16:
        return res.astype('uint16')
    
    
def Dicom_to_Image(path, uint):
    """
    Convert a DICOM file to an image array.

    Args:
        Path (str): The path to the DICOM file.
        uint (int): The data type of the output image array (16 for uint16, 8 for uint8).

    Returns:
        tuple: A tuple containing the converted image array and the instance number of the DICOM file.
                If an error occurs during conversion, None is returned for both values.
    """
    try:
        DCM_Img = PDCM.read_file(path)

        rows = DCM_Img.get(0x00280010).value  # Get number of rows from tag (0028, 0010)
        cols = DCM_Img.get(0x00280011).value  # Get number of cols from tag (0028, 0011)

        Instance_Number = int(DCM_Img.get(0x00200013).value)  # Get actual slice instance number from tag (0020, 0013)

        Window_Center = int(DCM_Img.get(0x00281050).value)  # Get window center from tag (0028, 1050)
        Window_Width = int(DCM_Img.get(0x00281051).value)  # Get window width from tag (0028, 1051)

        Window_Max = int(Window_Center + Window_Width / 2)
        Window_Min = int(Window_Center - Window_Width / 2)

        if (DCM_Img.get(0x00281052) is None):
            Rescale_Intercept = 0
        else:
            Rescale_Intercept = int(DCM_Img.get(0x00281052).value)

        if (DCM_Img.get(0x00281053) is None):
            Rescale_Slope = 1
        else:
            Rescale_Slope = int(DCM_Img.get(0x00281053).value)

        if uint == 16:
            New_Img = np.zeros((rows, cols), np.uint16)
            maxValue = 2**16 -1
        else:
            New_Img = np.zeros((rows, cols), np.uint8)
            maxValue = 255

        Pixels = DCM_Img.pixel_array

        for i in range(rows):
            for j in range(cols):
                Pix_Val = Pixels[i][j]
                Rescale_Pix_Val = Pix_Val * Rescale_Slope + Rescale_Intercept

                if (Rescale_Pix_Val > Window_Max):  # if intensity is greater than max window
                    New_Img[i][j] = maxValue
                elif (Rescale_Pix_Val < Window_Min):  # if intensity is less than min window
                    New_Img[i][j] = 0
                else:
                    New_Img[i][j] = int(((Rescale_Pix_Val - Window_Min) / (Window_Max - Window_Min)) * maxValue)

        return New_Img, Instance_Number
    
    except Exception as e:
        print(e)
        return None, None

def Dicom_to_png(path, output_path, uint=8, width=1024, height=512):
    """
    Convert a DICOM file to PNG format.

    Args:
        path (str): The path to the DICOM file.
        uint (int, optional): The number of bits to use for pixel intensity. Defaults to 8.
        width (int, optional): The desired width of the converted image. Defaults to 1024. can't be over 2048.
        height (int, optional): The desired height of the converted image. Defaults to 512. can't be over 1024.

    Returns:
        tuple: A tuple containing the original width and height of the converted image.
    """
    try:
        Output_Image, Instance_Number = Dicom_to_Image(path, uint)
        
        
        if Output_Image is None:
            return None, None
        
        original_height, original_width = Output_Image.shape

        Output_Image = resize_image(Output_Image, uint)

        Output_Image = bilinear_resize_vectorized(Output_Image, height=height, width=width, uint=uint)

        cv2.imwrite(output_path, Output_Image)
        return original_width, original_height
    
    except Exception as e:
        print(e)
        return None, None