import cv2
import os
cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
directory = r'C:\Users\acer\Desktop\minor project\camimage'
os.chdir(directory) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break

        img_counter += 1

cam.release()

cv2.destroyAllWindows()

import glob
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import os
from keras_vggface.utils import decode_predictions
from keras_vggface.utils import preprocess_input
from keras import applications
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
from matplotlib import pyplot
import numpy as np
from numpy import expand_dims
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
import mtcnn
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
# import glob
filenames = sorted(glob.glob(r"C:\Users\acer\Desktop\py\recog face\*"))
print(filenames)

embeddings = get_embeddings(filenames)

#user  = input("Enter the name of the image path: ")
#print(user)
#imageFileName = [user]
#new = get_embeddings(imageFileName)
#print(new)
user = ("C:\\Users\\acer\\Desktop\\camimage\\opencv_frame_0.png")
imageFileName = [user]
new = get_embeddings(imageFileName)
embedding = np.array(embeddings)
np.save('embedding', embedding)
embd = np.load('embedding.npy')
embd_keys = ['brad', 'google face', 'ruja', 'sabudh']
res = {embd_keys[i]: embedding[i] for i in range(len(embd_keys))}
print("Resultant dictionary is : " + str(res))
i = 0
e = 0
for x in embd:
    print(x)
    e = is_match(x, new)
    if e == 1:
        #print ('is a match with ')
        break
    i += 1
# if e == 0 :
    #print ('the person is not recognized')
#key = embd.keys(i)
if e == 1:
    k = 0
    for j in res.keys():
        if k == i:
            key = j
        k += 1

    print(key)

# load image from file
    pixels = pyplot.imread(user)
# create the detector, using default weights
    detector = MTCNN()
# detect faces in the image
    results = detector.detect_faces(pixels)
# extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    img = cv2.imread(user)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    name = key
    img = cv2.putText(img, name, (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('123',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if e == 0:
    print('the person is not recognized')
