#%% Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rn
import glob
import io

from PIL import Image

from keras.preprocessing import image

#%% Read data
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

images = daisy.append([dandelion, rose, sunflower, tulip], ignore_index=True)

#%% Load Images
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

#%% Display random images
categories = images["category"].values

fig, ax = plt.subplots(5, 2)
fig.set_size_inches(15, 15)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(y))
        ax[i, j].imshow(X[l])
        ax[i, j].set_title("Flower: " + categories[l])

plt.tight_layout()

#%%
