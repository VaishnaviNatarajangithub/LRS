import os
from preprocess import PreProcess
from deepMachine import DeepMachineLearning
from ocr import OCROnObjects
from textclassification import TextClassification
from datetime import datetime
import wx
import time

from dbAspect import DBConnection

# instantiate the db connection once
db_aspect = DBConnection()

def license_plate_extract(plate_like_objects, pre_process):
    """
    Selects and validates candidate plate regions.

    Args:
        plate_like_objects (list): Candidate plate images.
        pre_process (PreProcess): PreProcess instance with helper methods.

    Returns:
        list or image: Processed license plate image or empty list if none found.
    """
    number_of_candidates = len(plate_like_objects)

    if number_of_candidates == 0:
        wx.MessageBox("License plate could not be located",
                      "Plate Localization", wx.OK | wx.ICON_ERROR)
        return []

    if number_of_candidates == 1:
        license_plate = pre_process.inverted_threshold(plate_like_objects[0])
    else:
        license_plate = pre_process.validate_plate(plate_like_objects)

    return license_plate


def execute_ALPR(imagepath, listResult):
    """
    Runs the full license plate recognition process.
    Function is called when user clicks on the execute button on the GUI.

    Parameters:
        imagepath (str): path to the image file
        listResult (wx.ListCtrl): List control to display results

    Returns:
        bool: True if recognition and DB save succeed, False otherwise.
    """
    start_time = time.time()

    root_folder = os.path.dirname(os.path.realpath(__file__))
    models_folder = os.path.join(root_folder, 'ml_models')

    # Step 1: Preprocess image and find plate-like objects
    try:
        pre_process = PreProcess(imagepath)
    except Exception as e:
        wx.MessageBox(f"Error loading image or preprocessing: {str(e)}",
                      "Preprocessing Error", wx.OK | wx.ICON_ERROR)
        return False

    plate_like_objects = pre_process.get_plate_like_objects()

    license_plate = license_plate_extract(plate_like_objects, pre_process)

    if len(license_plate) == 0:
        return False

    # Step 2: OCR character segmentation
    try:
        ocr_instance = OCROnObjects(license_plate)
    except Exception as e:
        wx.MessageBox(f"Error during OCR segmentation: {str(e)}",
                      "OCR Error", wx.OK | wx.ICON_ERROR)
        return False

    if not ocr_instance.candidates or 'fullscale' not in ocr_instance.candidates:
        wx.MessageBox("No character was segmented",
                      "Character Segmentation", wx.OK | wx.ICON_ERROR)
        return False

    # Step 3: Load SVM model and classify characters
    model_path = os.path.join(models_folder, 'SVC_model', 'SVC_model.pkl')
    if not os.path.exists(model_path):
        wx.MessageBox("SVM model not found!",
                      "Model Error", wx.OK | wx.ICON_ERROR)
        return False

    try:
        deep_learn = DeepMachineLearning()
        text_result = deep_learn.learn(
            ocr_instance.candidates['fullscale'],
            model_path,
            (20, 20)
        )
    except Exception as e:
        wx.MessageBox(f"Error during deep learning classification: {str(e)}",
                      "Classification Error", wx.OK | wx.ICON_ERROR)
        return False

    # Step 4: Text classification and reconstruction
    text_phase = TextClassification()
    scattered_plate_text = text_phase.get_text(text_result)
    plate_text = text_phase.text_reconstruction(
        scattered_plate_text,
        ocr_instance.candidates.get('columnsVal', [])
    )

    if not plate_text:
        wx.MessageBox("No license plate text was recognized.",
                      "Recognition Error", wx.OK | wx.ICON_ERROR)
        return False

    elapsed_time = time.time() - start_time
    print('ALPR process took {:.2f} seconds'.format(elapsed_time))

    # Step 5: Insert recognized plate text into GUI list control
    row_index = listResult.InsertItem(listResult.GetItemCount(), plate_text)

    # Step 6: Save the recognized plate text and timestamp to database
    try:
        db_aspect.save_alpr(plate_text, str(datetime.today()))
    except Exception as e:
        wx.MessageBox(f"Error saving to database: {str(e)}",
                      "Database Error", wx.OK | wx.ICON_ERROR)

    # Step 7: Lookup vehicle info from database or CSV via db_aspect
    vehicle_info = None
    try:
        vehicle_info = db_aspect.get_vehicle_info(plate_text)
    except Exception as e:
        wx.MessageBox(f"Error retrieving vehicle info: {str(e)}",
                      "Information Retrieval Error", wx.OK | wx.ICON_ERROR)

    if vehicle_info:
        wx.MessageBox(
            f"Owner: {vehicle_info.get('owner','N/A')}\n"
            f"Issue Date: {vehicle_info.get('issue_date','N/A')}\n"
            f"Expiry Date: {vehicle_info.get('expiry_date','N/A')}\n"
            f"Chassis: {vehicle_info.get('chassis','N/A')}\n"
            f"Type: {vehicle_info.get('type','N/A')}",
            "Vehicle Information", wx.OK | wx.ICON_INFORMATION
        )

        listResult.SetItem(row_index, 1, vehicle_info.get('owner', ''))
        listResult.SetItem(row_index, 2, vehicle_info.get('issue_date', ''))
        listResult.SetItem(row_index, 3, vehicle_info.get('expiry_date', ''))
        listResult.SetItem(row_index, 4, vehicle_info.get('chassis', ''))
        listResult.SetItem(row_index, 5, vehicle_info.get('type', ''))
    else:
        wx.MessageBox("Vehicle Information could not be retrieved",
                      "Information Retrieval", wx.OK | wx.ICON_WARNING)

    return True
