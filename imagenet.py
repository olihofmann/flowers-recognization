#%% Imports
import numpy as np
import pandas as pd
import io
import glob
import base64
import keras
import warnings
import matplotlib as plt
import cv2

from PIL import Image

from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, GlobalAveragePooling2D, Dropout
from keras.applications import inception_v3
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocessor
from keras.optimizers import Adam

from azureml.core import Dataset, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from zipfile import ZipFile

#%% Download image dataset
subscription_id = '6583b89e-b354-4b5b-9906-047be634ef6b'
resource_group = 'ML'
workspace_name = 'ML-Workspace'

interactive_auth = InteractiveLoginAuthentication()
workspace = Workspace(subscription_id, resource_group, workspace_name, auth=interactive_auth)

flowers_ds = Dataset.get_by_name(workspace, name="Flowers")
flowers_ds.download(target_path=".", overwrite=False)

#%% Extract files
with ZipFile("flowers.zip", "r") as flowers_zip:
    file_names = flowers_zip.namelist()
    for file_name in file_names:
        if not file_name.startswith("_"):
            flowers_zip.extract(file_name, "./flowers")

#%% Reading data and put into a dataframe
def create_csv(path,encoded_category,category):
    img = glob.glob(path)
    img = pd.DataFrame(img)
    img["path"] = img[0]
    img = img.drop([0],axis=1)
    img["label"] = encoded_category
    img["category"] = category
    return(img)

daisy = create_csv("flowers/daisy/*.jpg", 0.0, "daisy")
dandelion = create_csv("flowers/dandelion/*.jpg", 1.0, "dandelion")
rose = create_csv("flowers/rose/*.jpg", 2.0, "rose")
sunflower = create_csv("flowers/sunflower/*.jpg", 3.0, "sunflower")
tulip = create_csv("flowers/tulip/*.jpg", 4.0, "tulip")


#%% Put all images together
images = daisy.append([dandelion, rose, sunflower, tulip], ignore_index=True)
images.head(5)

#%% Build data frames
train_image = []
filename = images["path"]
def load_images(length=200, width=200):
    for filename in images["path"]:
        with open(filename, "rb") as f:
            fname = f.read()
        image_1 = Image.open(io.BytesIO(fname))
        image_1 = image_1.resize((length, width), Image.ANTIALIAS)
        img = image.img_to_array(image_1)
        img = img / 255
        train_image.append(img)
    return train_image

train_image= load_images()
X = np.array(train_image)
y = images["label"].values

#%% Build test and train data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% Convert category to categorical
y_train_encoded = keras.utils.to_categorical(y_train, num_classes=5, dtype="int32")
y_test_encoded = keras.utils.to_categorical(y_test, num_classes=5, dtype="int32")
y_train_encoded

#%% Build base model
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(200, 200, 3))

#%% Define model
layer = base_model.output
layer = GlobalAveragePooling2D()(layer)
layer = Dense(1024, activation="relu")(layer)
layer = Dropout(0.2)(layer)
layer = Dense(1024, activation="relu")(layer)
layer = Dropout(0.2)(layer)
layer = Dense(2048, activation="relu")(layer)
predictions = Dense(5, activation="softmax")(layer)

model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False 

#%% Compile the model
model.compile(Adam(lr=.001), loss="categorical_crossentropy", metrics=["accuracy"])

#%% Train the model
model.fit(X_train, y_train_encoded, batch_size=32, epochs=50, validation_data=[X_test, y_test_encoded])

#%%
