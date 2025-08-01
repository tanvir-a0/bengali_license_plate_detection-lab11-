from functions import show_image_with_scale, convert_to_standard_size, scale_back_to_orignal_image
import re
from functions import preprocessing_and_giving_me_cropped_images
from functions import extract_texts_from_cropped_images


# with open("bengali_ocr_output.txt", "w", encoding="utf-8") as f:
#     f.write("অসদফা\n8758 শি\n")


current_data = "Vehicle643" 
""
list_of_images_that_can_be_licanse_plate = preprocessing_and_giving_me_cropped_images(
    f"data/{current_data}.jpg", 
    show_steps=True)
extract_texts_from_cropped_images(
    list_of_images_that_can_be_licanse_plate,
    file_name=current_data,
    save_directory="data_output"
)
