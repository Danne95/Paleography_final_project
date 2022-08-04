
import cv2
import keras
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

const_classes = {"ashkenazi":0, "byzantine":1, "italian":2, "oriental":3, "sephardic":4, "yemenite":5}
const_subclasses = {"cursive":0, "semisquare":1, "square":2}
if (os.getcwd() == '/home/historicalmanuscripts'):
    class_model = tf.keras.models.load_model("paleo_site/myConfig/models/EfficientNetV2B2_final")
    subclass_model = tf.keras.models.load_model("paleo_site/myConfig/models/Xception_SC_final_50")
else:
    class_model = tf.keras.models.load_model("myConfig/models/EfficientNetV2B2_final")
    subclass_model = tf.keras.models.load_model("myConfig/models/Xception_SC_final_50")

def patchifier(img, method = None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    patches = []
    for r in range(0,img.shape[0]-400+1,400):
        for c in range(0,img.shape[1]-400+1,400):
            cropped = img[r:r+400,c:c+400]
            #if patch_viability(cropped) == True:
            if method:
              if method == "inv_otsu":
                cropped = cv2.GaussianBlur(cropped,(5,5),0)
                _,cropped = cv2.threshold(cropped,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                cropped = cv2.cvtColor(cropped,cv2.COLOR_GRAY2BGR)
            patches.append(cropped)
    return patches

def preprocess_image(img, method = None):
    h,w = img.shape[0],img.shape[1]
    img = img[h//10:9*h//10,w//10:9*w//10]
    patches = None
    #if 2-pages image, split into 2
    if img.shape[0]<img.shape[1]:
        left_img = img[0:img.shape[0],0:img.shape[1]//2]
        patches = patchifier(left_img, method)
        right_img = img[0:img.shape[0],img.shape[1]//2+1:img.shape[1]]
        patches += patchifier(right_img, method)
    else:
        patches = patchifier(img, method)
    return np.array(patches)

def predict_image(image, class_model, subclass_model):
    #testing while models files are missing
    if (class_model==1 and subclass_model==2):
        return ("this is a testing print.\ngood job making it so far!\nyou are still missing models paths and/or files")
    if type(image) == str:
        image = cv2.imread(image)
    if type(class_model) == str:
        class_model = keras.models.load_model(class_model)
    if type(subclass_model) == str:
        subclass_model = keras.models.load_model(subclass_model)
    patches = preprocess_image(image, "inv_otsu")
    if not patches.size:
        return -1
    print(patches)
    print(patches.shape)
    class_predictions = class_model.predict(patches)
    class_prediction = np.argmax(class_predictions, axis=1)
    subclass_predictions = subclass_model.predict(patches)
    subclass_prediction = np.argmax(subclass_predictions, axis=1)
    class_bins = np.bincount(class_prediction)
    class_bins = np.divide(class_bins,sum(class_bins))
    subclass_bins = np.bincount(subclass_prediction)
    subclass_bins = np.divide(subclass_bins,sum(subclass_bins))
    return {"class":np.argmax(class_bins), "subclass":np.argmax(subclass_bins), "class_bins" : class_bins, "subclass_bins":subclass_bins}

def revert(prediction):
    return "{0} {1}".format(list(const_classes.keys())[prediction[0]],list(const_subclasses.keys())[prediction[1]])
