import cv2
import numpy as np
import base64
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from mimetypes import guess_type
from ultralytics import YOLO

# Load the models
model_dict = {
    'health_vs_diseased': YOLO('D:/FYP Website - new version/wheaties/models/healthvsdiseased.pt'),
    'rust_vs_smut': YOLO('D:/FYP Website - new version/wheaties/models/smutvsrust.pt'),
    'brown_vs_stem': YOLO('D:/FYP Website - new version/wheaties/models/brownvsstem.pt'),
    'brown_rust_scoring': YOLO('D:/FYP Website - new version/wheaties/models/brown rust scoring mode.pt'),
    'stem_rust_scoring': YOLO('D:/FYP Website - new version/wheaties/models/stemrust scoring.pt'),
    'smut_scoring': YOLO('D:/FYP Website - new version/wheaties/models/loose smut scoring.pt')
}

# Label mappings for models
label_mappings = {
    'health_vs_diseased': {'healthy': 'Healthy wheat', 'diseased': 'Diseased wheat'},
    'rust_vs_smut': {'rust': 'Rust diseased wheat', 'smut': 'Smut diseased wheat'},
    'brown_vs_stem': {'brown rust': 'Brown rust diseased wheat', 'stem rust': 'Stem rust diseased wheat'},
    'brown_rust_scoring': {'low': 'Brown rust low', 'medium': 'Brown rust medium', 'high': 'Brown rust high'},
    'stem_rust_scoring': {'low_sr': 'Stem rust low', 'med_sr': 'Stem rust medium', 'high_sr': 'Stem rust high'},
    'smut_scoring': {'low': 'Loose smut low', 'medium': 'Loose smut medium', 'high': 'Loose smut high'}
}

def predict_image(model, img, height, width, model_key):
    results = model.predict(img, save=False, imgsz=(height, width))
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    label = names_dict[np.argmax(probs)]
    
    # Map the label if necessary
    if model_key in label_mappings:
        label = label_mappings[model_key].get(label, label)
    
    return label

def handle_image_upload(image_file):
    # Check if the file is an image
    mime_type, _ = guess_type(image_file.name)
    if not mime_type or not mime_type.startswith('image'):
        return None, 'Kindly upload image only'
    
    file_name = default_storage.save(image_file.name, ContentFile(image_file.read()))
    file_path = default_storage.path(file_name)

    # Read the image using OpenCV
    image = cv2.imread(file_path)
    height, width = image.shape[:2]

    if image is None:
        return None, 'Invalid image file'

    # Predict health vs diseased
    health_vs_diseased_label = predict_image(model_dict['health_vs_diseased'], image, height, width, 'health_vs_diseased')
    predicted_label = health_vs_diseased_label

    if health_vs_diseased_label == 'Diseased wheat':
        rust_vs_smut_label = predict_image(model_dict['rust_vs_smut'], image, height, width, 'rust_vs_smut')

        if rust_vs_smut_label == 'Rust diseased wheat':
            brown_vs_stem_label = predict_image(model_dict['brown_vs_stem'], image, height, width, 'brown_vs_stem')

            if brown_vs_stem_label == 'Brown rust diseased wheat':
                brown_rust_scoring_label = predict_image(model_dict['brown_rust_scoring'], image, height, width, 'brown_rust_scoring')
                predicted_label = brown_rust_scoring_label

            elif brown_vs_stem_label == 'Stem rust diseased wheat':
                stem_rust_scoring_label = predict_image(model_dict['stem_rust_scoring'], image, height, width, 'stem_rust_scoring')
                predicted_label = stem_rust_scoring_label

        elif rust_vs_smut_label == 'Smut diseased wheat':
            smut_scoring_label = predict_image(model_dict['smut_scoring'], image, height, width, 'smut_scoring')
            predicted_label = smut_scoring_label

    # Convert the original image to base64 to display in the template
    with open(file_path, 'rb') as f:
        original_image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # Clean up the uploaded image file
    default_storage.delete(file_path)

    return original_image_base64, predicted_label
