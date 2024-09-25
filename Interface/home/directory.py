import cv2
import numpy as np
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from mimetypes import guess_type
from ultralytics import YOLO
import os
import glob

# Load the models
model_1 = YOLO('D:/FYP Website - new version/wheaties/models/healthvsdiseased.pt')
model_2 = YOLO('D:/FYP Website - new version/wheaties/models/smutvsrust.pt')
model_3 = YOLO('D:/FYP Website - new version/wheaties/models/brownvsstem.pt')
model_4 = YOLO('D:/FYP Website - new version/wheaties/models/loose smut scoring.pt')
model_5 = YOLO('D:/FYP Website - new version/wheaties/models/brown rust scoring mode.pt')
model_6 = YOLO('D:/FYP Website - new version/wheaties/models/stemrust scoring.pt')

# Model selection dictionary
model_dict = {
    '1': model_1,
    '2': model_2,
    '3': model_3,
    '4': model_4,
    '5': model_5,
    '6': model_6
}

# Label mappings for models
label_mappings = {
    '1': {'healthy': 'Healthy wheat', 'diseased': 'Diseased wheat'},
    '2': {'rust': 'Rust diseased wheat', 'smut': 'Smut diseased wheat'},
    '3': {'brown rust': 'Brown rust diseased wheat', 'stem rust': 'Stem rust diseased wheat'},
    '4': {'low': 'loose smut low', 'medium': 'loose smut medium', 'high': 'loose smut high'},
    '5': {'low': 'brown rust low', 'medium': 'brown rust medium', 'high': 'brown rust high'},
    '6': {'low_sr': 'stem rust low', 'med_sr': 'stem rust medium', 'high_sr': 'stem rust high'}
}

def count_images_in_directory(directory_path, selected_option):
    # Retrieve the selected model
    selected_model = model_dict.get(selected_option)
    if not selected_model:
        return None, None, 'Invalid model selection'

    # Initialize label counts dictionary with mapped labels if necessary
    if selected_option in label_mappings:
        label_counts = {mapped_label: 0 for mapped_label in label_mappings[selected_option].values()}
    else:
        label_counts = {label: 0 for label in selected_model.names.values()}

    image_count = 0

    # Get all images in the directory
    image_paths = glob.glob(os.path.join(directory_path, '*.jpg'))  # Adjust pattern for other image formats if needed

    for image_path in image_paths:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        results = selected_model.predict(image, save=False, imgsz=(height, width))

        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        label = names_dict[np.argmax(probs)]

        # Map the label if necessary
        if selected_option in label_mappings:
            label = label_mappings[selected_option].get(label, label)

        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

        image_count += 1

    return image_count, label_counts, None
