import glob
from scipy.spatial.distance import cosine
from keras.models import load_model
from keras_vggface.utils import decode_predictions
from keras_vggface.utils import preprocess_input
from keras import applications
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
import mtcnn
import tensorflow as tf
from camera import VideoCamera
import cv2
from flask import Flask, Response, json, render_template
from werkzeug.utils import secure_filename
from flask import request, json
from os import path, getcwd
import time
import os
import base64

from PIL import Image

app = Flask(__name__)
print(mtcnn.__version__)

model = MTCNN()


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# create a vggface2 model
model = VGGFace(model='resnet50')
# summarize input and output shape
print('Inputs: %s' % model.inputs)
print('Outputs: %s' % model.outputs)


def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False,
                    input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    if score <= thresh:
        return 1
    else:
        return 0
# datapath for recognised faces


# @app.route('/')
# def run_script():
#     print("Starting server")
#     filenames = sorted(glob.glob(r'C:\Users\Shripa\Desktop\minor\images\*'))
#     # filenames = sorted(glob.glob("/content/drive/MyDrive/Colab Notebooks/images/*"))
#     print(filenames)

#     embeddings = get_embeddings(filenames)

#     user = (r'C:\Users\Shripa\Desktop\minor\unseen face\unseen sabudh.jpg')
#     print(user)
#     imageFileName = [user]
#     new = get_embeddings(imageFileName)
#     print(new)

#     embedding = np.array(embeddings)
#     np.save('embedding',embedding)
#     embd=np.load('embedding.npy')
#     embd_keys=['brad', 'google face', 'ruja', 'sabudh' ]
#     res = {embd_keys[i]: embedding[i] for i in range(len(embd_keys))} 
#     print ("Resultant dictionary is : " +  str(res)) 
#     i=0
#     e=0
#     for x in embd:
#         print (x)
#         e = is_match(x,new)
#         if e == 1:
#             #print ('is a match with ')  
#             break
#     i += 1
# #if e == 0 :
#   #print ('the person is not recognized')
# #key = embd.keys(i)
#     if e == 1:
#         k = 0
#         for j in res.keys():
#             if k == i:
#                 key = j 
#             k +=1
   
#         print (key)
#     return "Hello world"


@app.route('/')
def index():
    return render_template('changed.html')

@app.route('/uploadBase64', methods=['GET','POST'])
def uploadBase64():
    if request.method == 'POST':
        print("uplaodBase64")
        bas64Data = request.json['base64']
        print(bas64Data)

        path = "fileImage1.jpg"
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(bas64Data))

        # jpgpath = "fileImage2.jpg"

        # img_png = Image.open(path)

        # img_png.save(jpgpath)
            
        print("Starting server")
        filenames = sorted(glob.glob(r'C:\Users\Shripa\Desktop\minor\images\*'))
        # filenames = sorted(glob.glob("/content/drive/MyDrive/Colab Notebooks/images/*"))
        print(filenames)

        embeddings = get_embeddings(filenames)

        user = path
        #user = r'C:\Users\Shripa\Desktop\minor\unseen face\unseen sabudh.jpg'
        print(user)
        imageFileName = [user]
        new = get_embeddings(imageFileName)
        print(new)

        embedding = np.array(embeddings)
        np.save('embedding',embedding)
        embd=np.load('embedding.npy')
        embd_keys=['brad', 'google face', 'ruja', 'sabudh' ]
        res = {embd_keys[i]: embedding[i] for i in range(len(embd_keys))} 
        print ("Resultant dictionary is : " +  str(res)) 
        i=0
        e=0
        for x in embd:
            print (x)
            e = is_match(x,new)
            if e == 1:
                #print ('is a match with ')  
                break
            i += 1

        print("e")
        print(e)
    #if e == 0 :
    #print ('the person is not recognized')
    #key = embd.keys(i)
        if e == 1:
            k = 0
            for j in res.keys():
                if k == i:
                    key = j 
                k +=1
    
            print(key)
        if e == 0:
            key = 'not recognized'
        print('completed _________')
            
        return key
    return render_template('signup.html')

# if __name__ == "__main__":
#     app.run(debug=True)
# app.run()
