from flask import Flask, request, jsonify, render_template
from PIL import Image
import matplotlib.pyplot as plt
from extract_bottleneck_features import *
from keras.models import load_model
from keras.src.applications.vgg19 import VGG19
from keras.preprocessing import image
from tqdm import tqdm
import numpy as np
from glob import glob
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50
import cv2
import os


app = Flask(__name__)



# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("../dog_project/dogImages/train/*/"))]

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../dog_project/haarcascades/haarcascade_frontalface_alt.xml')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def predict_dog_breed(image_path):
    # Define the CNN model architecture (choose one: VGG19, Resnet50, InceptionV3, Xception)
    model = VGG19(weights='imagenet', include_top=False)

    # Extract bottleneck features from the image
    bottleneck_features = extract_VGG19(path_to_tensor(image_path))

    # Load the saved model for dog breed prediction
    dog_breed_model = load_model('../dog_project/saved_models/weights.best.VGG19.hdf5')

    # Get the predicted vector for the dog breed
    predicted_vector = dog_breed_model.predict(bottleneck_features)

    # Get the index of the predicted breed
    predicted_breed_index = np.argmax(predicted_vector)

    # Get the corresponding dog breed name
    predicted_breed = dog_names[predicted_breed_index]

    return predicted_breed


def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))


### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def dog_human_resemblance(image_path):
    # Check if a dog is detected
    if dog_detector(image_path):
        breed = predict_dog_breed(image_path)
        return f"This is a dog, and it looks like a {breed}."

    # Check if a human is detected
    if face_detector(image_path):
        breed = predict_dog_breed(image_path)
        return f"This is a human, but they resemble a {breed}."

    # Neither dog nor human detected
    return "Error: Neither a dog nor a human is detected in the image."

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        file = request.files['file']

        if file:
            # Save the uploaded image temporarily
            temp_image_path = 'temp_image.jpg'
            file.save(temp_image_path)

            # Call the functions to make predictions
            dog_or_human = dog_human_resemblance(temp_image_path)

            # Remove the temporary image file
            os.remove(temp_image_path)

            return jsonify({'result': dog_or_human})


        else:
            return jsonify({'error': 'No file uploaded'})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
