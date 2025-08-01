import os
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
import re

def show_image_with_scale(title, image, scale=1):
    if scale != 1.0:
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        image = cv2.resize(image, (width, height))
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def convert_to_standard_size(image):
    h, w, _ = image.shape
    scale = 500 / w
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    return resized

def scale_back_to_orignal_image(original_file_path, converted_image, x, y, w, h):
    original_image = cv2.imread(original_file_path)
    if original_image is None:
        print("Error: Original image not found or unable to read.")
        return None
    
    scale_x = original_image.shape[1] / converted_image.shape[1]
    scale_y = original_image.shape[0] / converted_image.shape[0]
    
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)
    
    cropped_image = original_image[y:y+h, x:x+w]
    return cropped_image
    
def preprocessing_and_giving_me_cropped_images(image_dir, show_steps = False):

    image = cv2.imread(image_dir)
    if image is None:
        print("Error: Image not found or unable to read.")
        return
    
    if show_steps: show_image_with_scale('Original Image', image)

    # converting image into same number of width and height.
    #if small then scale it up and if large then scale it down while keeping the aspect ratio.

    resized = convert_to_standard_size(image)
    if show_steps: show_image_with_scale('Resized Image', resized)


    grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    if show_steps: show_image_with_scale('Grey Image', grey)

    bilateral_filtered = cv2.bilateralFilter(grey, 12, 15, 15)
    if show_steps: show_image_with_scale('Bilateral Filtered Image', bilateral_filtered)

    equalized_contrast = cv2.equalizeHist(bilateral_filtered)
    if show_steps: show_image_with_scale('Equalized Contrast Image', equalized_contrast)

    shobel_x = cv2.Sobel(equalized_contrast, cv2.CV_64F, 1, 0, ksize=3)
    shobel_y = cv2.Sobel(equalized_contrast, cv2.CV_64F, 0, 1, ksize=3)
    if show_steps: show_image_with_scale('Sobel X', shobel_x)
    if show_steps: show_image_with_scale('Sobel Y', shobel_y)


    shobel_x = cv2.convertScaleAbs(shobel_x)
    if show_steps: show_image_with_scale('Sobel X (Absolute)', shobel_x)


    dialated_image = cv2.dilate(shobel_x, np.ones((3, 3), np.uint8), iterations=1)
    if show_steps: show_image_with_scale('Dialated Image', dialated_image)

    _, binary_threshold = cv2.threshold(dialated_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show_steps: show_image_with_scale('Binary Threshold Image', binary_threshold)

    contours, _ = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    list_of_images_that_can_be_licanse_plate = []

    resized_copy = resized.copy()

    for contour in contours:
        if cv2.contourArea(contour) > 800 and cv2.contourArea(contour) < 100000:  # Filter contours by area
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if aspect_ratio > 0.7 and aspect_ratio < 2.5:
                cv2.rectangle(resized_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                print(f"area: {cv2.contourArea(contour)}, aspect ratio: {w/h:.2f}, length: {w}, height: {h}")
                if show_steps: show_image_with_scale('Contours on Image', resized_copy)

                cropped_image = resized[y:y + h, x:x + w]
                original_image_cropped = scale_back_to_orignal_image(image_dir, resized, x, y, w, h)
                if original_image_cropped is not None:
                    list_of_images_that_can_be_licanse_plate.append(original_image_cropped)
                    if show_steps: show_image_with_scale('Cropped Image', original_image_cropped)
    return list_of_images_that_can_be_licanse_plate

def extract_texts_from_cropped_images(list_of_images_that_can_be_licanse_plate, file_name = None, save_directory = None):
    for i, image in enumerate(list_of_images_that_can_be_licanse_plate):
        image = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE to enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Threshold (inverted to make text black if needed)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        show_image_with_scale(f'Processed Crop {i+1}', thresh)
        # Try OCR now
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(thresh, lang='ben', config=config)

        # Regular expression for four continuous Bengali digits (০-৯)
        match = re.search(r'[০-৯]{4}', text)
        if match:
            # Create the directory if it doesn't exist
            if save_directory and not os.path.exists(save_directory):
                os.makedirs(save_directory)
            with open(f"{save_directory}/{file_name}.txt", "a", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
