import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import restoration
from skimage import measure
from skimage.measure import regionprops
from skimage.filters import threshold_otsu
from skimage.transform import resize

class PreProcess():
    
    def __init__(self, image_location):
        """
        Reads the image in grayscale and thresholds the image.

        Parameters:
        -----------
        image_location: str; full image directory path
        """
        self.full_car_image = rgb2gray(imread(image_location))
        self.full_car_image = self.resize_if_necessary(self.full_car_image)
        self.binary_image = self.threshold(self.full_car_image)
        
    def denoise(self, imgDetails):
        return restoration.denoise_tv_chambolle(imgDetails)
        
    def threshold(self, gray_image):
        """
        Uses the Otsu threshold method to generate a binary image.

        Parameters:
        -----------
        gray_image: 2D array: grayscale image to be thresholded

        Returns:
        --------
        2-D array of the binary image each pixel is either 1 or 0
        """
        thresholdValue = threshold_otsu(gray_image)
        return gray_image > thresholdValue
        
    def get_plate_like_objects(self):
        """
        Uses connected component analysis to find potential plate regions.

        Returns:
        --------
        3-D array of license plate candidate regions.
        """
        self.label_image = measure.label(self.binary_image)
        self.plate_objects_cordinates = []
        threshold = self.binary_image
        plate_dimensions = (
            0.08 * threshold.shape[0], 0.2 * threshold.shape[0], 
            0.15 * threshold.shape[1], 0.4 * threshold.shape[1]
        )
        minHeight, maxHeight, minWidth, maxWidth = plate_dimensions
        plate_like_objects = []

        for region in regionprops(self.label_image):
            if region.area < 10:
                continue

            minRow, minCol, maxRow, maxCol = region.bbox
            regionHeight = maxRow - minRow
            regionWidth = maxCol - minCol

            if (minHeight <= regionHeight <= maxHeight and
                minWidth <= regionWidth <= maxWidth and
                regionWidth > regionHeight):
                plate_like_objects.append(
                    self.full_car_image[minRow:maxRow, minCol:maxCol]
                )
                self.plate_objects_cordinates.append((minRow, minCol, maxRow, maxCol))
                
        return plate_like_objects

    def validate_plate(self, candidates):
        """
        Validates candidate plate regions by column-wise projection.

        Parameters:
        -----------
        candidates: list of 2D arrays

        Returns:
        --------
        2D array of the most likely license plate region.
        """
        license_plate = []
        highest_average = 0

        for each_candidate in candidates:
            height, width = each_candidate.shape
            each_candidate = self.inverted_threshold(each_candidate)

            total_white_pixels = sum(np.sum(each_candidate[:, col]) for col in range(width))
            average = float(total_white_pixels) / width

            if average >= highest_average:
                highest_average = average
                license_plate = each_candidate

        return license_plate

    def inverted_threshold(self, grayscale_image):
        """
        Inverts the threshold to highlight characters over white plates.

        Parameters:
        -----------
        grayscale_image: 2D grayscale image

        Returns:
        --------
        2D binary image
        """
        threshold_value = threshold_otsu(grayscale_image) - 0.05
        return grayscale_image < threshold_value

    def resize_if_necessary(self, image_to_resize):
        """
        Resizes the image to a manageable size while maintaining aspect ratio.

        Parameters:
        -----------
        image_to_resize: 2D or 3D array

        Returns:
        --------
        Resized image (if needed), otherwise original
        """
        height, width = image_to_resize.shape
        ratio = float(width) / height

        if width > 600:
            width = 600
            height = round(width / ratio)
            return resize(image_to_resize, (height, width))

        return image_to_resize
