from flask import Flask , Response, json, render_template
from werkzeug.utils import secure_filename
from flask import request
from os import path, getcwd
import time
import os

app = Flask(__name__)
import cv2
from camera import VideoCamera
import tensorflow as tf
import mtcnn
print (mtcnn.__version__)
from keras_vggface.vggface import VGGFace
import matplotlib.pyplot as plt
from numpy import expand_dims
import numpy as np
from matplotlib import pyplot
from numpy import asarray
from PIL import Image
from mtcnn.mtcnn import MTCNN
import cv2
from keras import applications
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import os
from keras.models import load_model
from scipy.spatial.distance import cosine
import glob



#app.config['file_allowed'] = ['image/png', 'image/jpeg']
#app.config['train_img'] = path.join(getcwd(), 'train_img')


#def gen(camera):
    #while True:
        #frame = camera.get_frame()
        #yield (b'--frame\r\n'
               #b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

#@app.route('/video_feed')
#def video_feed():
    #return Response(gen(VideoCamera()),
                    #mimetype='multipart/x-mixed-replace; boundary=frame')

#@app.route('/')
#def index():
    #return render_template('index.html')

#@app.route('/')
#def get_gallery():
   #images = os.listdir(os.path.join(app.static_folder, "capture_image"))
   #return render_template('gallery.html', images=images)
@app.route('/')
def run_script():
    file = open(r'C:\Users\Shripa\Desktop\minor\minor.py', 'r').read()
    return exec(file)

if __name__ == "__main__":
    app.run(debug=True)
app.run()