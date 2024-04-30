import os
import numpy as np
from google.cloud import vision
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image as keras_image # type: ignore

# Set Google Cloud credentials path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\MAHESH\Downloads\praxis-atrium-416107-f40a5c0c1415.json"

# Function to check if the image contains skin
def is_skin_image(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.safe_search_detection(image=image)

    # Check for likelihood of adult, medical, violence, and spoof content
    likelihood = response.safe_search_annotation
    if (likelihood.adult == 5 or
        likelihood.medical == 5 or
        likelihood.violence == 5 or
        likelihood.spoof == 5):
        return False
    else:
        return True

# Function to detect objects in the image using Google Cloud Vision API
def detect_objects(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.object_localization(image=image)

    objects = []
    for obj in response.localized_object_annotations:
        objects.append(obj.name)

    return objects

# Function to preprocess image for prediction
def preprocess_image(image_path, target_size=(224, 224)):
    img = keras_image.load_img(image_path, target_size=target_size)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the class of the image
def predict_image_class(image_path, model):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    return predicted_class_index

# Map class indices to cancer types
class_map = {
    0: 'akiec', # Actinic keratoses and intraepithelial carcinoma
    1: 'bcc',   # Basal cell carcinoma
    2: 'bkl',   # Benign keratosis-like lesions
    3: 'df',    # Dermatofibroma
    4: 'mel',   # Melanoma
    5: 'nv',    # Melanocytic nevi
    6: 'vasc'   # Vascular lesions
}

# Provide the path to the image you want to detect objects in
image_path = r"C:\Users\MAHESH\Downloads\2.jpg"

# Detect objects in the image
detected_objects = detect_objects(image_path)
if not detected_objects:
    print("Objects not detected in the image: ", detected_objects)
    print("Proceeding with skin cancer detection.")
    # Load the trained model
    model = load_model(r"D:\Visual Studio Code\Projects\Skin_Cancer_Detection\Files\Trained Models\ham10000_cnn_model.h5")
    # Predict the class of the image
    predicted_class_index = predict_image_class(image_path, model)
    predicted_cancer_type = class_map[predicted_class_index]
    print("Predicted the Cancer: ", predicted_cancer_type)

    # Specify the file path to save the result
    output_file = r'D:\Visual Studio Code\Projects\Skin_Cancer_Detection\Files\predicted_cancer_type.txt'
    # Write the predicted cancer type to the text file
    with open(output_file, 'w') as f:
        f.write("Predicted Cancer Type: " + predicted_cancer_type + "\n")
else:
    print("Enter a valid image to check skin cancer")
    print("The image consists: ", detected_objects)
