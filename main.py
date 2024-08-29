from File import File_im
from PIL import Image
import os
import matplotlib.pyplot as plt

#def display(file_path):
#    image=Image.open(file_path)
#    plt.imshow(image)
#    plt.axis('off')
#    plt.show()
path,files=File_im()

#first_image_path=os.path.join(path,files[1])
#print(display(first_image_path))

import glob
image_directory = r'C:\Users\divya\OneDrive\Desktop\okk\Fasion Recomm\women-fashion\women-fashion'
image_paths_list = [file for file in glob.glob(os.path.join(image_directory, '*.*')) if file.endswith(('.jpg', '.png', '.jpeg', 'webp'))]
print(image_paths_list)
import tensorflow
import keras
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.output)
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array_expanded)

def extract_features(model, preprocessed_img):
    features = model.predict(preprocessed_img)
    flattened_features = features.flatten()
    normalized_features = flattened_features / np.linalg.norm(flattened_features)
    return normalized_features

all_features = []
all_image_names = []

for img_path in image_paths_list:
    preprocessed_img = preprocess_image(img_path)
    features = extract_features(model, preprocessed_img)
    all_features.append(features)
    all_image_names.append(os.path.basename(img_path))


from scipy.spatial.distance import cosine

def recommend_fashion_items_cnn(input_image_path, all_features, all_image_names, model, top_n=5):
    preprocessed_img = preprocess_image(input_image_path)
    input_features = extract_features(model, preprocessed_img)

    similarities = [1 - cosine(input_features, other_feature) for other_feature in all_features]
    similar_indices = np.argsort(similarities)[-top_n:]

    similar_indices = [idx for idx in similar_indices if idx != all_image_names.index(input_image_path)]

    plt.figure(figsize=(15, 10))
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title("Input Image")
    plt.axis('off')

    for i, idx in enumerate(similar_indices[:top_n], start=1):
        image_path = os.path.join(r'C:\Users\divya\OneDrive\Desktop\okk\Fasion Recomm\women-fashion\women fashion', all_image_names[idx])
        plt.subplot(1, top_n + 1, i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"Recommendation {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

input_image_path = r'C:\Users\divya\OneDrive\Desktop\okk\Fasion Recomm\women-fashion\women fashion\black double-breasted blazer with gold buttons, paired with a mid-length skirt.jpg'
recommend_fashion_items_cnn(input_image_path, all_features, image_paths_list, model, top_n=4)
