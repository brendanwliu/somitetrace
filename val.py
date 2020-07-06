import model
from somiteData import trainGenerator, testGenerator, saveResult
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import seaborn as sns
import cv2 as cv

from PIL import Image

DATA_PATH = 'datasets/SomiteTraceLibrary/input/'
NO_OF_TEST_IMAGES = len(os.listdir(DATA_PATH + 'test_frames'))

m = model.unet("./model_weights/dice_Model.h5")
testGen = testGenerator(DATA_PATH, "test_frames/", num_image=NO_OF_TEST_IMAGES, target_size=(128,128))

predictions = m.predict_generator(testGen, steps=NO_OF_TEST_IMAGES)

print(predictions.shape)

# mat = predictions[0].reshape(128,128)
# img = Image.fromarray(mat, 'L')
# img.show()

print(np.sum(predictions[5].reshape(128,128)*255))

fig, axes = plt.subplots(2,2,figsize=(9,9))
plt.setp(axes, xticks=[], yticks=[])
for i, ax in enumerate(axes.flat):
    mat = predictions[i].reshape(128,128)
    img = Image.fromarray(np.uint8(mat * 255), 'L')
    ax.imshow(img)
plt.show()